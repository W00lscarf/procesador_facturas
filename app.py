# app.py — Extractor de Facturas (OCR robusto con convert_from_bytes)
import io, re
from typing import List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

import streamlit as st
import pandas as pd
from PIL import Image

import pytesseract
from pdf2image import convert_from_bytes
import pdfplumber

st.set_page_config(page_title="Extractor de Facturas (OCR robusto)", layout="wide")

# ---------- Patrones ----------
MONTHS = {
    'enero':'01','febrero':'02','marzo':'03','abril':'04','mayo':'05','junio':'06',
    'julio':'07','agosto':'08','septiembre':'09','setiembre':'09','octubre':'10',
    'noviembre':'11','diciembre':'12'
}
DATE_TXT = r'(\d{1,2}\s+(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)\s+\d{2,4})'
DATE_NUM = r'(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{2,4})'
DATE_PAT = rf'(?:{DATE_TXT}|{DATE_NUM})'
MONEY_PAT = r'(\$?\s*\d{1,3}(?:[.\s]\d{3})*(?:[\,\.]\d{2})?)'
FOLIO_PAT = r'([A-Z0-9\-]{5,})'

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
    # 30 septiembre 2025
    m = re.match(r'(\d{1,2})\s+([a-záéíóúñ]+)\s+(\d{2,4})', t)
    if m:
        d, mon, y = m.groups()
        mon = MONTHS.get(mon, None)
        if mon:
            y = int(y);  y = y+2000 if y < 100 else y
            return f"{y:04d}-{int(mon):02d}-{int(d):02d}"
    # 30/09/2025
    m = re.match(r'(\d{1,2})/(\d{1,2})/(\d{2,4})', t)
    if m:
        d, mo, y = map(int, m.groups())
        y = y+2000 if y < 100 else y
        return f"{y:04d}-{mo:02d}-{d:02d}"
    return s

# ---------- OCR TSV ----------
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

def find_right_value(words: List[Word], label_regex: re.Pattern, value_regex: re.Pattern,
                     x_gap=6, y_tol=18) -> Optional[str]:
    """
    Busca etiqueta en una línea, lee a la derecha; si no, intenta línea vecina en misma banda vertical.
    """
    L = lines_dict(words)
    for key, ws in L.items():
        line_text = " ".join([w.text for w in ws])
        if re.search(label_regex, line_text, re.IGNORECASE):
            # extremo derecho de la etiqueta
            label_end_x = None
            for w in ws:
                if re.search(label_regex, w.text, re.IGNORECASE): label_end_x = w.right
            if label_end_x is None: continue
            # 1) misma línea
            tail = " ".join([w.text for w in ws if w.left >= label_end_x + x_gap])
            m = re.search(value_regex, tail, re.IGNORECASE)
            if m: return norm_spaces(m.group(1))
            # 2) línea vecina (misma banda Y ± y_tol)
            target_mid = [w.mid_y for w in ws][-1]
            for key2, ws2 in L.items():
                if key2 == key: continue
                same_band = abs(ws2[0].mid_y - target_mid) <= y_tol and ws2[0].left > label_end_x
                if same_band:
                    tail2 = " ".join([w.text for w in ws2])
                    m2 = re.search(value_regex, tail2, re.IGNORECASE)
                    if m2: return norm_spaces(m2.group(1))
    return None

# ---------- Extracción ----------
def extract_ocr_page(img: Image.Image) -> dict:
    words = ocr_words(img)
    lbl_num   = re.compile(r'^N[°º]$|N[°º]\b')
    lbl_emis  = re.compile(r'Fecha\s*Emis\.?')
    lbl_venc  = re.compile(r'Fecha\s*Venc\.?')
    lbl_folio = re.compile(r'Folio')
    lbl_pago  = re.compile(r'Forma\s*de\s*pago')
    lbl_firma = re.compile(r'Firma')

    val_num   = re.compile(r'(\d{3,})')
    val_date  = re.compile(DATE_PAT, re.IGNORECASE)
    val_folio = re.compile(FOLIO_PAT)
    val_pago  = re.compile(r'([A-Za-zÁÉÍÓÚÑáéíóúñ\. ]{3,})')

    out = {}
    out["n_factura"]   = find_right_value(words, lbl_num, val_num)
    out["fecha_emis"]  = to_iso_date(find_right_value(words, lbl_emis, val_date))
    out["fecha_venc"]  = to_iso_date(find_right_value(words, lbl_venc, val_date))
    out["folio"]       = find_right_value(words, lbl_folio, val_folio)
    fp = find_right_value(words, lbl_pago, val_pago)
    if fp:
        fp = fp.replace("Cr", "Crédito").replace("Cred.", "Crédito").replace("Credito","Crédito")
    out["forma_pago"]  = norm_spaces(fp) if fp else None

    # Proveedor (línea con "SPA")
    prov = None
    for _, ws in lines_dict(words).items():
        t = " ".join([w.text for w in ws])
        m = re.search(r'([A-Z0-9 ]+\s+SPA)\b', t, re.IGNORECASE)
        if m:
            prov = norm_spaces(m.group(1).upper()); break
    out["proveedor"] = prov

    # Total (línea con "Total")
    total = None
    for _, ws in lines_dict(words).items():
        t = " ".join([w.text for w in ws])
        if re.search(r'\bTotal(?:\s*a\s*pagar)?\b', t, re.IGNORECASE):
            m = re.search(MONEY_PAT, t)
            if m: total = clean_money_to_float(m.group(1)); break
    out["monto_total"] = total
    return out

def extract_pdf_text(file_bytes: bytes) -> dict:
    """Fallback por texto embebido."""
    out = {}
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            txt = "\n".join([p.extract_text() or "" for p in pdf.pages])
    except:
        txt = ""
    if not txt.strip(): return out

    # Proveedor
    m = re.search(r'([A-ZÁÉÍÓÚÑ0-9 ]+\s+SPA)\b', txt)
    if m: out["proveedor"] = norm_spaces(m.group(1).upper())

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
        fp = fp.replace("Cr", "Crédito").replace("Cred.", "Crédito").replace("Credito","Crédito")
        out["forma_pago"] = fp
    m = re.search(r'Firma.*?'+DATE_PAT, txt, re.IGNORECASE|re.DOTALL)
    out["fecha_firma"] = to_iso_date(m.group(1)) if m else None
    m = re.search(r'Total(?:\s*a\s*pagar)?\s*[:\-]?\s*'+MONEY_PAT, txt, re.IGNORECASE)
    if m: out["monto_total"] = clean_money_to_float(m.group(1))
    return out

# ---------- UI ----------
st.title("🧾 Extractor de Facturas (OCR robusto)")
st.caption("Lee valores a la derecha de las etiquetas (N°, Fecha Emis., Fecha Venc., Folio, Forma de pago). Soporta PDF e imágenes. OCR a 300 DPI y regex tolerantes.")

uploads = st.file_uploader("Sube uno o varios archivos (PDF/PNG/JPG).", type=["pdf","png","jpg","jpeg"], accept_multiple_files=True)

rows = []
if uploads:
    pb = st.progress(0.0)
    for i, f in enumerate(uploads):
        content = f.read()
        # Raster a 300 DPI para OCR (convert_from_bytes para PDFs)
        try:
            if f.name.lower().endswith(".pdf"):
                pages = convert_from_bytes(content, dpi=300, fmt="png")
            else:
                pages = [Image.open(io.BytesIO(content)).convert("RGB")]
        except Exception as e:
            st.warning(f"{f.name}: no pude convertir a imagen ({e}). Intento OCR directo.")
            try:
                pages = [Image.open(io.BytesIO(content)).convert("RGB")]
            except Exception as e2:
                st.error(f"{f.name}: tampoco pude abrir como imagen ({e2}).")
                pages = []

        combined = {}
        for p_idx, img in enumerate(pages, start=1):
            o = extract_ocr_page(img)
            for k, v in o.items():
                if v and not combined.get(k):
                    combined[k] = v

        # Fallback de texto embebido si falta algo
        t = extract_pdf_text(content)
        for k, v in t.items():
            if v and not combined.get(k):
                combined[k] = v

        # Normaliza a string para evitar NaN float
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
    st.info("Sube archivos para comenzar. Si la fecha viene como “30 septiembre 2025”, este lector la detecta y la normaliza a ISO (YYYY-MM-DD).")
