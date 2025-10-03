# app.py — Extractor de Facturas (coords PyMuPDF + OCR prepro + reglas Chile)
import io, re
from typing import List, Tuple, Optional
from dataclasses import dataclass

import streamlit as st
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance

import pytesseract
import fitz  # PyMuPDF
import pdfplumber

st.set_page_config(page_title="Extractor de Facturas (robusto)", layout="wide")

# ----------------- Patrones -----------------
MONTHS = {
    'enero':'01','febrero':'02','marzo':'03','abril':'04','mayo':'05','junio':'06',
    'julio':'07','agosto':'08','septiembre':'09','setiembre':'09','octubre':'10',
    'noviembre':'11','diciembre':'12'
}
DATE_TXT = r'(\d{1,2}\s+(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)\s+\d{2,4})'
DATE_NUM = r'(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{2,4})'
DATE_PAT = rf'(?:{DATE_TXT}|{DATE_NUM})'
MONEY_PAT = r'(\$?\s*\d{1,3}(?:[.\s]\d{3})*(?:[\,\.]\d{2})?)'
# evita capturar "Fecha", permite alfanuméricos con guiones y longitud mínima
FOLIO_PAT = r'((?!Fecha)\b[A-Z0-9][A-Z0-9\-]{3,})'
RUT_PAT   = r'\b\d{1,2}\.?\d{3}\.?\d{3}-[0-9Kk]\b'
CORP_SUFFIXES = ['SPA','SpA','S.A.','SA','LTDA','Ltda','EIRL','LIMITADA']
BAD_PREFIXES = ["DATOS DESPACHO", "DATOS DESPACHO O INSTALACION", "DATOS DESPACHO O INSTALACIÓN",
                "DIRECCION", "DIRECCIÓN", "ENVIO", "ENVÍO", "UBICACION", "UBICACIÓN"]

def norm_spaces(s: Optional[str]) -> Optional[str]:
    if s is None: return None
    return re.sub(r'\s+', ' ', s).strip()

def clean_money_to_float(s: Optional[str]) -> Optional[float]:
    if not s: return None
    s = s.replace(' ', '').replace('$', '')
    s = s.replace('.', '').replace(',', '.')
    m = re.findall(r'[\d\.]+', s)
    try: return float(m[0])
    except: return None

def to_iso_date(s: Optional[str]) -> Optional[str]:
    if not s: return None
    t = s.strip().lower().replace('\\','/').replace('.', '/').replace('-', '/')
    m = re.match(r'(\d{1,2})\s+([a-záéíóúñ]+)\s+(\d{2,4})', t)
    if m:
        d, mon, y = m.groups()
        mon = MONTHS.get(mon, None)
        if mon:
            y = int(y);  y = y+2000 if y < 100 else y
            return f"{y:04d}-{int(mon):02d}-{int(d):02d}"
    m = re.match(r'(\d{1,2})/(\d{1,2})/(\d{2,4})', t)
    if m:
        d, mo, y = map(int, m.groups())
        y = y+2000 if y < 100 else y
        return f"{y:04d}-{mo:02d}-{d:02d}"
    return s

def is_good_provider_line(t: str) -> bool:
    T = t.upper().strip(": ").strip()
    if any(T.startswith(p) for p in BAD_PREFIXES):
        return False
    return any(suf in T for suf in ["SPA","S.A.","LTDA","EIRL","SpA","SA","LIMITADA"])

# ----------------- Rasterizar PDF (PyMuPDF) -----------------
def pdf_to_images(pdf_bytes: bytes, dpi: int = 300) -> list[Image.Image]:
    images = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for pno in range(len(doc)):
        page = doc[pno]
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        images.append(img)
    doc.close()
    return images

# ----------------- PREPROCESADO OCR -----------------
def preprocess_for_ocr(img: Image.Image) -> Image.Image:
    # Escala ×1.5, gris, contraste
    w, h = img.size
    img = img.resize((int(w*1.5), int(h*1.5)))
    img = ImageOps.grayscale(img)
    img = ImageEnhance.Contrast(img).enhance(1.5)
    # binarización suave
    img = img.point(lambda x: 0 if x < 200 else 255, mode='1').convert("L")
    return img

# ----------------- OCR TSV -----------------
@dataclass
class Word:
    text: str; left: int; top: int; width: int; height: int; conf: float
    right: int; mid_y: float; line_key: Tuple[int,int,int]

def ocr_words(img: Image.Image, page_no=1) -> List[Word]:
    cfg = "--oem 3 --psm 6"
    df = pytesseract.image_to_data(img, lang="spa+eng", config=cfg, output_type=pytesseract.Output.DATAFRAME)
    out: List[Word] = []
    if df is None or len(df)==0: return out
    for _, r in df.iterrows():
        try:
            txt = str(r.get("text","")).strip()
            conf = float(r.get("conf",-1))
            if not txt or conf < 0: continue
            l, t, w, h = int(r["left"]), int(r["top"]), int(r["width"]), int(r["height"])
            right = l + w; mid_y = t + h/2
            line_key = (page_no, int(r.get("block_num",0)), int(r.get("line_num",0)))
            out.append(Word(txt, l, t, w, h, conf, right, mid_y, line_key))
        except: pass
    return out

def lines_dict(words: List[Word]):
    lines = {}
    for w in words:
        lines.setdefault(w.line_key, []).append(w)
    for k in lines:
        lines[k].sort(key=lambda z: z.left)
    return lines

def find_adjacent_value(words: List[Word], label_regex: re.Pattern, value_regex: re.Pattern,
                        x_gap=6, y_tol=22, prefer_below=False) -> Optional[str]:
    """
    Busca el valor a la derecha o debajo de la etiqueta (OCR).
    - prefer_below=True: intenta debajo primero (útil para 'Folio').
    """
    L = lines_dict(words)

    def right_value():
        for key, ws in L.items():
            line_text = " ".join([w.text for w in ws])
            if label_regex.search(line_text):
                label_end_x = None
                for w in ws:
                    if label_regex.search(w.text):
                        label_end_x = w.right
                if label_end_x is None:
                    continue
                tail = " ".join([w.text for w in ws if w.left >= label_end_x + x_gap])
                m = value_regex.search(tail)
                if m:
                    return norm_spaces(m.group(1))
        return None

    def below_value():
        # busca en líneas siguientes con x cercano y dentro de tolerancia vertical ampliada
        for key, ws in L.items():
            if any(label_regex.search(w.text) for w in ws):
                label_end_x = max([w.right for w in ws if label_regex.search(w.text)])
                label_mid_y = ws[-1].mid_y
                for key2, ws2 in L.items():
                    if key2 == key:
                        continue
                    if (ws2[0].mid_y > label_mid_y) and (ws2[0].left >= label_end_x - 80) and \
                       (ws2[0].left <= label_end_x + 280) and (ws2[0].mid_y - label_mid_y <= y_tol*2.5):
                        candidate = " ".join([w.text for w in ws2])
                        m = value_regex.search(candidate)
                        if m:
                            return norm_spaces(m.group(1))
        return None

    if prefer_below:
        return below_value() or right_value()
    else:
        return right_value() or below_value()

# ----------------- PyMuPDF: texto con coords (sin OCR) -----------------
def extract_by_coords_pdf(pdf_bytes: bytes) -> dict:
    """
    Usa texto con coordenadas (PyMuPDF) para leer valores a la derecha o debajo.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    out = {}
    lbl_num   = re.compile(r'^N[°º]$|N[°º]\b', re.IGNORECASE)
    lbl_emis  = re.compile(r'Fecha\s*Emis\.?', re.IGNORECASE)
    lbl_venc  = re.compile(r'Fecha\s*Venc\.?', re.IGNORECASE)
    lbl_folio = re.compile(r'Folio', re.IGNORECASE)
    lbl_pago  = re.compile(r'Forma\s*de\s*pago', re.IGNORECASE)
    lbl_firma = re.compile(r'Firma', re.IGNORECASE)
    value_date  = re.compile(DATE_PAT, re.IGNORECASE)
    value_num   = re.compile(r'(\d{3,})')
    value_folio = re.compile(FOLIO_PAT, re.IGNORECASE)
    value_pago  = re.compile(r'([A-Za-zÁÉÍÓÚÑáéíóúñ\. ]{3,})')

    def take_right(line_spans, idx_label, regex):
        x_end = line_spans[idx_label]["bbox"][2]
        tail_text = " ".join([s["text"] for s in line_spans if s["bbox"][0] >= x_end + 2])
        m = regex.search(tail_text)
        return norm_spaces(m.group(1)) if m else None

    def take_below(page, span, regex, y_window=40):
        x0, y0, x1, y1 = span["bbox"]
        rect = fitz.Rect(x0, y1, x1 + 250, y1 + y_window)  # zona debajo y un poco a la derecha
        words = page.get_text("words")  # [(x0,y0,x1,y1,"text",block,line,word_no), ...]
        line_texts = []
        for w in words:
            wx0, wy0, wx1, wy1, wtxt, *_ = w
            if fitz.Rect(wx0, wy0, wx1, wy1).intersects(rect):
                line_texts.append((wy0, wtxt))
        if not line_texts:
            return None
        line_texts.sort(key=lambda t: t[0])
        candidate = " ".join([t[1] for t in line_texts])
        m = regex.search(candidate)
        return norm_spaces(m.group(1)) if m else None

    prov_candidates = []

    for page in doc:
        dicttext = page.get_text("dict")
        for b in dicttext.get("blocks", []):
            for l in b.get("lines", []):
                spans = l.get("spans", [])
                if not spans: 
                    continue
                line_text = " ".join([s["text"] for s in spans]).strip()

                # Proveedor: filtra líneas con sufijos corporativos reales (evita "DATOS DESPACHO...")
                if is_good_provider_line(line_text):
                    prov_candidates.append(norm_spaces(line_text).upper())

                # N°
                for i, s in enumerate(spans):
                    if lbl_num.search(s["text"]):
                        val = take_right(spans, i, value_num)
                        if val: out.setdefault("n_factura", val)

                # Fecha Emis y Venc (derecha o debajo)
                for i, s in enumerate(spans):
                    if lbl_emis.search(s["text"]):
                        val = take_right(spans, i, value_date) or take_below(page, s, value_date, y_window=50)
                        if val: out.setdefault("fecha_emis", to_iso_date(val))
                for i, s in enumerate(spans):
                    if lbl_venc.search(s["text"]):
                        val = take_right(spans, i, value_date) or take_below(page, s, value_date, y_window=50)
                        if val: out.setdefault("fecha_venc", to_iso_date(val))

                # Folio: debajo primero
                for i, s in enumerate(spans):
                    if lbl_folio.search(s["text"]):
                        val = take_below(page, s, value_folio, y_window=60) or take_right(spans, i, value_folio)
                        if val: out.setdefault("folio", val)

                # Forma de pago
                for i, s in enumerate(spans):
                    if lbl_pago.search(s["text"]):
                        val = take_right(spans, i, value_pago) or take_below(page, s, value_pago, y_window=60)
                        if val:
                            fp = val
                            # normaliza crédito y dedup
                            fp = re.sub(r'(Crédito)(?:\s*\1)+', r'\1', fp, flags=re.IGNORECASE)
                            fp = fp.replace("Cr", "Crédito").replace("Cred.", "Crédito").replace("Credito","Crédito")
                            out.setdefault("forma_pago", norm_spaces(fp))

                # Total en línea
                if re.search(r'\bTotal(?:\s*a\s*pagar)?\b', line_text, re.IGNORECASE):
                    m = re.search(MONEY_PAT, line_text)
                    if m:
                        out.setdefault("monto_total", clean_money_to_float(m.group(1)))

        # Proveedor por RUT: toma la línea anterior al RUT si existe
        fulltext = page.get_text()
        rutm = re.search(RUT_PAT, fulltext)
        if rutm:
            pos = fulltext.find(rutm.group(0))
            before = fulltext[:pos]
            lines = [l.strip() for l in before.splitlines() if l.strip()]
            if lines and is_good_provider_line(lines[-1]):
                prov_candidates.append(norm_spaces(lines[-1]).upper())

        # Fallback Total (si no se encontró en línea): mayor monto >= 1000
        if out.get("monto_total") is None:
            money_vals = [clean_money_to_float(m) for m in re.findall(MONEY_PAT, fulltext)]
            money_vals = [x for x in money_vals if x and x >= 1000]
            if money_vals:
                out["monto_total"] = max(money_vals)

    doc.close()

    # Escoge proveedor con sufijo corporativo y más largo
    prov_candidates = [c for c in prov_candidates if is_good_provider_line(c)]
    if prov_candidates:
        out["proveedor"] = max(prov_candidates, key=len)

    return out

# ----------------- OCR page extractor -----------------
def extract_ocr_page(img: Image.Image) -> dict:
    imgp = preprocess_for_ocr(img)
    words = ocr_words(imgp)

    lbl_num   = re.compile(r'^N[°º]$|N[°º]\b', re.IGNORECASE)
    lbl_emis  = re.compile(r'Fecha\s*Emis\.?', re.IGNORECASE)
    lbl_venc  = re.compile(r'Fecha\s*Venc\.?', re.IGNORECASE)
    lbl_folio = re.compile(r'Folio', re.IGNORECASE)
    lbl_pago  = re.compile(r'Forma\s*de\s*pago', re.IGNORECASE)
    lbl_firma = re.compile(r'Firma', re.IGNORECASE)

    val_num   = re.compile(r'(\d{3,})')
    val_date  = re.compile(DATE_PAT, re.IGNORECASE)
    val_folio = re.compile(FOLIO_PAT, re.IGNORECASE)
    val_pago  = re.compile(r'([A-Za-zÁÉÍÓÚÑáéíóúñ\. ]{3,})')

    out = {}
    out["n_factura"]  = find_adjacent_value(words, lbl_num,   val_num)
    out["fecha_emis"] = to_iso_date(find_adjacent_value(words, lbl_emis,  val_date))
    out["fecha_venc"] = to_iso_date(find_adjacent_value(words, lbl_venc,  val_date))
    out["folio"]      = find_adjacent_value(words, lbl_folio, val_folio, prefer_below=True)
    fp = find_adjacent_value(words, lbl_pago,  val_pago)
    if fp:
        fp = re.sub(r'(Crédito)(?:\s*\1)+', r'\1', fp, flags=re.IGNORECASE)
        fp = fp.replace("Cr", "Crédito").replace("Cred.", "Crédito").replace("Credito","Crédito")
    out["forma_pago"] = norm_spaces(fp) if fp else None

    # Proveedor por sufijos / RUT (OCR)
    prov = None
    for _, ws in lines_dict(words).items():
        t = " ".join([w.text for w in ws])
        if is_good_provider_line(t):
            prov = norm_spaces(t).upper(); break
    if not prov:
        full_ocr = pytesseract.image_to_string(imgp, lang="spa+eng")
        rutm = re.search(RUT_PAT, full_ocr)
        if rutm:
            pos = full_ocr.find(rutm.group(0))
            before = full_ocr[:pos]
            lines = [l.strip() for l in before.splitlines() if l.strip()]
            if lines and is_good_provider_line(lines[-1]):
                prov = norm_spaces(lines[-1]).upper()
    if prov: out["proveedor"] = prov

    # Total (línea con 'Total'); fallback: mayor monto >= 1000 en OCR completo
    for _, ws in lines_dict(words).items():
        t = " ".join([w.text for w in ws])
        if re.search(r'\bTotal(?:\s*a\s*pagar)?\b', t, re.IGNORECASE):
            m = re.search(MONEY_PAT, t)
            if m:
                out["monto_total"] = clean_money_to_float(m.group(1))
                break
    if out.get("monto_total") is None:
        full_ocr = pytesseract.image_to_string(imgp, lang="spa+eng")
        money_vals = [clean_money_to_float(m) for m in re.findall(MONEY_PAT, full_ocr)]
        money_vals = [x for x in money_vals if x and x >= 1000]
        if money_vals:
            out["monto_total"] = max(money_vals)

    return out

# ----------------- Fallback: texto embebido plano -----------------
def extract_pdf_text_fallback(file_bytes: bytes) -> dict:
    out = {}
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            txt = "\n".join([p.extract_text() or "" for p in pdf.pages])
    except:
        txt = ""
    if not txt.strip(): return out
    m = re.search(r'([A-ZÁÉÍÓÚÑ0-9 ]+\s+(?:SPA|S\.A\.|LTDA|EIRL|SpA|SA|LIMITADA))\b', txt)
    if m and is_good_provider_line(m.group(1)): out["proveedor"] = norm_spaces(m.group(1)).upper()
    m = re.search(r'N[°º]\s*[:\-]?\s*(\d{3,})', txt);           out["n_factura"]  = m.group(1) if m else None
    m = re.search(r'Fecha\s*Emis\.?\s*[:\-]?\s*'+DATE_PAT, txt, re.IGNORECASE)
    out["fecha_emis"] = to_iso_date(m.group(1)) if m else None
    m = re.search(r'Fecha\s*Venc\.?\s*[:\-]?\s*'+DATE_PAT, txt, re.IGNORECASE)
    out["fecha_venc"] = to_iso_date(m.group(1)) if m else None
    m = re.search(r'Folio\s*[:\-]?\s*'+FOLIO_PAT, txt, re.IGNORECASE)
    out["folio"] = m.group(1) if m else None
    m = re.search(r'Forma\s+de\s*pago\s*[:\-]?\s*([A-Za-zÁÉÍÓÚÑáéíóúñ\. ]+)', txt, re.IGNORECASE)
    if m:
        fp = norm_spaces(m.group(1))
        fp = re.sub(r'(Crédito)(?:\s*\1)+', r'\1', fp, flags=re.IGNORECASE)
        fp = fp.replace("Cr", "Crédito").replace("Cred.", "Crédito").replace("Credito","Crédito")
        out["forma_pago"] = fp
    m = re.search(r'Firma.*?'+DATE_PAT, txt, re.IGNORECASE|re.DOTALL)
    out["fecha_firma"] = to_iso_date(m.group(1)) if m else None
    m = re.search(r'Total(?:\s*a\s*pagar)?\s*[:\-]?\s*'+MONEY_PAT, txt, re.IGNORECASE)
    if m: out["monto_total"] = clean_money_to_float(m.group(1))
    return out

# ----------------- UI -----------------
st.title("🧾 Extractor de Facturas (robusto)")
st.caption("Coord. PyMuPDF + OCR con preprocesado. Lee valores a la derecha o DEBAJO de las etiquetas. Reglas para proveedor (sufijos/RUT).")

uploads = st.file_uploader(
    "Sube uno o varios archivos (PDF/PNG/JPG).",
    type=["pdf","png","jpg","jpeg"],
    accept_multiple_files=True
)

rows = []
if uploads:
    pb = st.progress(0.0)
    for i, f in enumerate(uploads):
        content = f.read()

        combined = {}

        # 1) Coordenadas PyMuPDF (sin OCR)
        if f.name.lower().endswith(".pdf"):
            try:
                coord_res = extract_by_coords_pdf(content)
                for k, v in coord_res.items():
                    if v: combined.setdefault(k, v)
            except Exception as e:
                st.warning(f"{f.name}: extract_by_coords_pdf falló ({e}).")

        # 2) OCR con prepro (rasterizar PDF o usar imagen)
        try:
            if f.name.lower().endswith(".pdf"):
                pages = pdf_to_images(content, dpi=300)
            else:
                pages = [Image.open(io.BytesIO(content)).convert("RGB")]
            for img in pages:
                o = extract_ocr_page(img)
                for k, v in o.items():
                    if v and not combined.get(k):
                        combined[k] = v
        except Exception as e:
            st.warning(f"{f.name}: OCR falló ({e}).")

        # 3) Fallback texto embebido plano (pdfplumber)
        t = extract_pdf_text_fallback(content)
        for k, v in t.items():
            if v and not combined.get(k):
                combined[k] = v

        def s(x): 
            if x is None: return ""
            return str(x)

        rows.append({
            "archivo": f.name,
            "proveedor": s(combined.get("proveedor")),
            "n_factura": s(combined.get("n_factura")),
            "forma_pago": s(combined.get("forma_pago")),
            "fecha_emis": s(combined.get("fecha_emis")),
            "fecha_venc": s(combined.get("fecha_venc")),
            "folio": s(combined.get("folio")),
            "fecha_firma": s(combined.get("fecha_firma")),
            "monto_total": s(combined.get("monto_total")),
        })
        pb.progress((i+1)/len(uploads))

    df = pd.DataFrame(rows)
    st.subheader("Tabla extraída (copiable)")
    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False, encoding="utf-8")
    st.download_button("📥 Descargar CSV", data=csv, file_name="facturas_extraidas.csv", mime="text/csv")
    st.text_area("CSV (copiar/pegar)", csv, height=160)
else:
    st.info("Sube archivos para comenzar. Captura 'Folio' debajo de la etiqueta y normaliza fechas tipo '30 septiembre 2025'.")
