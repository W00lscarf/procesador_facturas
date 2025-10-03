# app.py
# ------------------------------
# Extractor de Facturas con OCR
# ------------------------------
# - Soporta PDF/imagen
# - Usa OCR con coordenadas (TSV) para leer el valor a la derecha de etiquetas
# - Fallback a texto embebido (pdfplumber) y regex
# - Campos: proveedor, N° factura, forma de pago, fecha emis., fecha venc.,
#           folio, fecha firma, monto total

import io, os, re
from dataclasses import dataclass
from typing import Optional, List, Tuple

import pandas as pd
import streamlit as st
from PIL import Image

# OCR / PDF
import pytesseract
from pdf2image import convert_from_path
import pdfplumber

st.set_page_config(page_title="Extractor de Facturas (OCR+TSV)", layout="wide")

# ------------------------------
# Utilidades de parsing
# ------------------------------
DATE_PAT = r'([0-9]{1,2}[\/\.\- ](?:[0-9]{1,2}|[A-Za-záéíóúñÑ]+)[\/\.\- ](?:[0-9]{2,4}))'
MONEY_PAT = r'(\$?\s*\d{1,3}(?:[.\s]\d{3})*(?:[\,\.]\d{2})?)'
FOLIO_PAT = r'([A-Z0-9\-]{5,})'

def norm_spaces(s: str) -> str:
    return re.sub(r'\s+', ' ', (s or '').strip())

def clean_money(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    s = s.replace(' ', '')
    s = s.replace('$', '').replace('.', '').replace(',', '.')
    m = re.findall(r'[\d\.]+', s)
    try:
        return float(m[0])
    except:
        return None

def to_iso_date(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    x = s.replace('\\', '/').replace('.', '/').replace('-', '/')
    parts = [p for p in re.split(r'[\/ ]', x) if p]
    # Admite "30/09/2025" o "30 septiembre 2025"
    meses = {
        'enero': '01','febrero': '02','marzo': '03','abril': '04','mayo': '05','junio': '06',
        'julio': '07','agosto': '08','septiembre':'09','setiembre':'09','octubre':'10',
        'noviembre':'11','diciembre':'12'
    }
    try:
        if len(parts) == 3 and parts[1].lower() in meses:  # 30 septiembre 2025
            d = int(parts[0]); m = int(meses[parts[1].lower()]); y = int(parts[2])
            if y < 100: y += 2000
            return f"{y:04d}-{m:02d}-{d:02d}"
        # Formatos numéricos
        m = re.match(r'(\d{1,2})/(\d{1,2})/(\d{2,4})', x)
        if m:
            d, mo, y = map(int, m.groups())
            if y < 100: y += 2000
            return f"{y:04d}-{mo:02d}-{d:02d}"
    except:
        pass
    return s  # si no puedo normalizar, devuelvo tal cual

# ------------------------------
# OCR a TSV (palabras con coordenadas)
# ------------------------------
@dataclass
class OcrWord:
    text: str
    left: int
    top: int
    width: int
    height: int
    conf: float
    line_id: Tuple[int,int,int]  # (page_num, block_num, line_num)
    right: int
    mid_y: float

def ocr_tsv(pil_img: Image.Image, page_num: int = 1) -> List[OcrWord]:
    tsv = pytesseract.image_to_data(pil_img, lang="spa+eng", output_type=pytesseract.Output.DATAFRAME)
    words = []
    if tsv is None or len(tsv) == 0:
        return words
    for _, row in tsv.iterrows():
        try:
            if str(row.get("text","")).strip() == "" or int(row.get("conf",-1)) < 0:
                continue
            text = str(row["text"])
            left = int(row["left"]); top = int(row["top"])
            width = int(row["width"]); height = int(row["height"])
            conf = float(row["conf"])
            block = int(row.get("block_num",0)); line = int(row.get("line_num",0))
            right = left + width
            mid_y = top + height/2
            words.append(
                OcrWord(text=text, left=left, top=top, width=width, height=height,
                        conf=conf, line_id=(page_num, block, line), right=right, mid_y=mid_y)
            )
        except:
            continue
    return words

def join_by_line(words: List[OcrWord]) -> dict:
    """Une palabras por línea conservando orden de x."""
    lines = {}
    for w in words:
        lines.setdefault(w.line_id, []).append(w)
    for k in lines:
        lines[k].sort(key=lambda z: z.left)
    return lines

def find_value_right_of(
    words: List[OcrWord],
    label_regex: re.Pattern,
    value_regex: re.Pattern,
    x_gap: int = 10,
    same_line_only: bool = True,
    y_tol: int = 8
) -> Optional[str]:
    """Busca una etiqueta (label_regex) y luego el primer valor que matchee value_regex a la derecha."""
    lines = join_by_line(words)
    for line_id, ws in lines.items():
        # Buscar índice de la etiqueta
        full_line = " ".join([w.text for w in ws])
        if re.search(label_regex, full_line, re.IGNORECASE):
            # posición del último token de la etiqueta
            # (heurística: hallar el token cuyo texto matchee parte final del label)
            label_end_x = None
            for w in ws:
                if re.search(label_regex, w.text, re.IGNORECASE):
                    label_end_x = w.right
            if label_end_x is None:
                continue
            # Buscar a la derecha en la misma línea
            candidates = [w for w in ws if w.left >= label_end_x + x_gap]
            candidate_text = " ".join([w.text for w in candidates])
            m = re.search(value_regex, candidate_text, re.IGNORECASE)
            if m:
                return norm_spaces(m.group(1))
        if not same_line_only:
            # buscar palabras próximas (misma banda Y)
            for i, w in enumerate(ws):
                if re.search(label_regex, w.text, re.IGNORECASE):
                    label_mid = w.mid_y
                    # Buscar en toda la página entre líneas dentro de tolerancia vertical y a la derecha
                    for lid, ws2 in lines.items():
                        for w2 in ws2:
                            if w2.left > w.right and abs(w2.mid_y - label_mid) <= y_tol:
                                # formo texto de la línea derecha
                                tail = " ".join([x.text for x in ws2 if x.left >= w.right + x_gap])
                                m = re.search(value_regex, tail, re.IGNORECASE)
                                if m:
                                    return norm_spaces(m.group(1))
    return None

# ------------------------------
# Extracción por OCR + fallback
# ------------------------------
def pages_from_file(uploaded_file) -> List[Image.Image]:
    content = uploaded_file.read()
    if uploaded_file.name.lower().endswith(".pdf"):
        return convert_from_path(io.BytesIO(content), dpi=200, fmt="png")  # pdf2image admite bytes en algunos entornos
    else:
        img = Image.open(io.BytesIO(content)).convert("RGB")
        return [img]

def extract_with_ocr(pil_img: Image.Image) -> dict:
    words = ocr_tsv(pil_img)
    # Patrones de etiquetas y valores
    lbl_num = re.compile(r'^(N[°º])$|N[°º]\b', re.IGNORECASE)
    lbl_emis = re.compile(r'Fecha\s*Emis\.?', re.IGNORECASE)
    lbl_venc = re.compile(r'Fecha\s*Venc\.?', re.IGNORECASE)
    lbl_folio = re.compile(r'Folio', re.IGNORECASE)
    lbl_formapago = re.compile(r'Forma\s*de\s*pago', re.IGNORECASE)
    lbl_firma = re.compile(r'Firma', re.IGNORECASE)
    # Valores
    val_num = re.compile(r'(\d{3,})')
    val_date = re.compile(DATE_PAT, re.IGNORECASE)
    val_folio = re.compile(FOLIO_PAT)
    val_pago = re.compile(r'([A-Za-zÁÉÍÓÚÑáéíóúñ\. ]{3,})')
    val_money = re.compile(MONEY_PAT)

    out = {
        "n_factura": find_value_right_of(words, lbl_num, val_num),
        "fecha_emis": to_iso_date(find_value_right_of(words, lbl_emis, val_date, same_line_only=False)),
        "fecha_venc": to_iso_date(find_value_right_of(words, lbl_venc, val_date, same_line_only=False)),
        "folio": find_value_right_of(words, lbl_folio, val_folio),
        "forma_pago": norm_spaces(find_value_right_of(words, lbl_formapago, val_pago)),
        "fecha_firma": to_iso_date(find_value_right_of(words, lbl_firma, val_date, same_line_only=False)),
    }

    # Proveedor: buscar líneas con "SPA" y tomar la más alta confianza
    # (o fallback a regex simple del texto completo)
    lines = join_by_line(words)
    spa_candidates = []
    for lid, ws in lines.items():
        line_text = " ".join([w.text for w in ws])
        if re.search(r'\b[A-Z0-9 ]+\s+SPA\b', line_text, re.IGNORECASE):
            spa = re.search(r'([A-Z0-9 ]+\s+SPA)\b', line_text, re.IGNORECASE).group(1)
            spa_candidates.append(norm_spaces(spa.upper()))
    if spa_candidates:
        # heurística: proveedor = candidato más largo
        out["proveedor"] = max(spa_candidates, key=len)
    else:
        out["proveedor"] = None

    # Monto Total: buscar línea con “Total” y luego el valor
    total_line = None
    for lid, ws in lines.items():
        line_text = " ".join([w.text for w in ws])
        if re.search(r'\bTotal(?:\s*a\s*pagar)?\b', line_text, re.IGNORECASE):
            total_line = " ".join([w.text for w in ws])
            m = re.search(MONEY_PAT, total_line, re.IGNORECASE)
            if m:
                out["monto_total"] = clean_money(m.group(1))
                break
    # Fallback: buscar el mayor número con formato de dinero en toda la imagen OCR
    if out.get("monto_total") is None:
        full_text = pytesseract.image_to_string(pil_img, lang="spa+eng")
        money_vals = [clean_money(m) for m in re.findall(MONEY_PAT, full_text)]
        money_vals = [x for x in money_vals if x is not None]
        if money_vals:
            out["monto_total"] = max(money_vals)

    # Normalizaciones
    if out.get("forma_pago"):
        fp = out["forma_pago"]
        fp = fp.replace("Cr", "Crédito").replace("Cred.", "Crédito").replace("Credito", "Crédito")
        out["forma_pago"] = fp.strip()

    return out

def extract_from_pdf_text(file_bytes: bytes) -> dict:
    out = {}
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
            text = "\n".join(pages)
    except:
        text = ""

    if not text.strip():
        return out

    # Regex fallback sobre texto
    out["proveedor"] = None
    m = re.search(r'\b([A-ZÁÉÍÓÚÑ0-9 ]+)\s+SPA\b', text)
    if m: out["proveedor"] = norm_spaces(m.group(1).upper() + " SPA")

    m = re.search(r'N[°º]\s*[:\-]?\s*(\d{3,})', text, re.IGNORECASE)
    if m: out["n_factura"] = m.group(1)

    m = re.search(r'Fecha\s*Emis\.?\s*[:\-]?\s*' + DATE_PAT, text, re.IGNORECASE)
    if m: out["fecha_emis"] = to_iso_date(m.group(1))

    m = re.search(r'Fecha\s*Venc\.?\s*[:\-]?\s*' + DATE_PAT, text, re.IGNORECASE)
    if m: out["fecha_venc"] = to_iso_date(m.group(1))

    m = re.search(r'Folio\s*[:\-]?\s*' + FOLIO_PAT, text, re.IGNORECASE)
    if m: out["folio"] = m.group(1)

    m = re.search(r'Forma\s+de\s+pago\s*[:\-]?\s*([A-Za-zÁÉÍÓÚÑáéíóúñ\. ]+)', text, re.IGNORECASE)
    if m:
        fp = m.group(1)
        fp = fp.replace("Cr", "Crédito").replace("Cred.", "Crédito").replace("Credito", "Crédito")
        out["forma_pago"] = norm_spaces(fp)

    m = re.search(r'Firma.*?' + DATE_PAT, text, re.IGNORECASE|re.DOTALL)
    if m: out["fecha_firma"] = to_iso_date(m.group(1))

    # Total
    m = re.search(r'Total(?:\s*a\s*pagar)?\s*[:\-]?\s*' + MONEY_PAT, text, re.IGNORECASE)
    if m:
        out["monto_total"] = clean_money(m.group(1))
    else:
        # último número grande
        nums = re.findall(MONEY_PAT, text)
        if nums:
            out["monto_total"] = clean_money(nums[-1])

    return out

# ------------------------------
# UI
# ------------------------------
st.title("🧾 Extractor de Facturas (OCR con coordenadas)")
st.caption("Sube PDF(s) o imagen(es). Lee valores a la derecha de las etiquetas (N°, Fecha Emis., Fecha Venc., Folio, Forma de pago) y extrae proveedor, fecha firma y monto total.")

uploads = st.file_uploader("Arrastra uno o varios archivos (PDF/PNG/JPG).", type=["pdf","png","jpg","jpeg"], accept_multiple_files=True)

results = []
if uploads:
    prog = st.progress(0.0)
    for i, f in enumerate(uploads):
        f_bytes = f.read()
        rows = []
        # Primero intentamos OCR con coordenadas (robusto para etiquetas en tablas)
        try:
            pages = convert_from_path(io.BytesIO(f_bytes), dpi=200, fmt="png") if f.name.lower().endswith(".pdf") \
                    else [Image.open(io.BytesIO(f_bytes)).convert("RGB")]
        except Exception as e:
            st.warning(f"{f.name}: No pude rasterizar el PDF. Intento con OCR directo de la 1ª página. Detalle: {e}")
            pages = []
            try:
                img = Image.open(io.BytesIO(f_bytes)).convert("RGB")
                pages = [img]
            except:
                pages = []

        combined = {}
        for p_idx, img in enumerate(pages, start=1):
            ocr_res = extract_with_ocr(img)
            # combina: el primero que trae valor se queda
            for k, v in ocr_res.items():
                if v and not combined.get(k):
                    combined[k] = v

        # Fallback: texto embebido si algo falta
        pdf_res = extract_from_pdf_text(f_bytes)
        for k, v in pdf_res.items():
            if v and not combined.get(k):
                combined[k] = v

        row = {
            "archivo": f.name,
            "proveedor": combined.get("proveedor"),
            "n_factura": combined.get("n_factura"),
            "forma_pago": combined.get("forma_pago"),
            "fecha_emis": combined.get("fecha_emis"),
            "fecha_venc": combined.get("fecha_venc"),
            "folio": combined.get("folio"),
            "fecha_firma": combined.get("fecha_firma"),
            "monto_total": combined.get("monto_total"),
        }
        results.append(row)
        prog.progress((i+1)/len(uploads))

    df = pd.DataFrame(results)
    st.subheader("Tabla extraída")
    st.dataframe(df.fillna(""), use_container_width=True)

    csv = df.to_csv(index=False, encoding="utf-8")
    st.download_button("📥 Descargar CSV", data=csv, file_name="facturas_extraidas.csv", mime="text/csv")
    st.markdown("O copia/pega desde aquí:")
    st.text_area("CSV (copiar)", csv, height=180)

else:
    st.info("Sube archivos para comenzar. Recomendación: PDFs con buena resolución o imágenes nítidas.")

st.markdown("---")
st.caption("Sugerencias: Si una etiqueta no aparece, aumenta el DPI a 250–300 o comparte 2–3 ejemplos para ajustar reglas.")
