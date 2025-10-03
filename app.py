# app.py — Extractor de Facturas (OCR robusto, sin Poppler: usa PyMuPDF)
import io, re
from typing import List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

import streamlit as st
import pandas as pd
from PIL import Image

import pytesseract
import fitz  # PyMuPDF
import pdfplumber

st.set_page_config(page_title="Extractor de Facturas (OCR robusto)", layout="wide")

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

def find_right_value(words: List[Word], label_regex: re.Pattern, value_regex: re.Pattern,
                     x_gap=6, y_tol=18) -> Optional[str]:
    L = lines_dict(words)
    for key, ws in L.items():
        line_text = " ".join([w.text for w in ws])
        if re.search(label_regex, line_text, re.IGNORECASE):
            label_end_x = None
            for w in ws:
                if re.search(label_regex, w.text, re.IGNORECASE): label_end_x = w.right
            if label_end_x is None: continue
            tail = " ".join([w.text for w in ws if w.left >= label_end_x + x_gap])
            m = re.search(value_regex, tail, re.IGNORECASE)
            if m: return norm_spaces(m.group(1))
            target_mid = [w.mid_y for w in ws][-1]
            for key2, ws2 in L.items():
                if key2 == key: continue
                same_band = abs(ws2[0].mid_y - target_mid) <= y_tol and ws2[0].left > label_end_x
                if same_band:
                    tail2 = " ".join([w.text for w in ws2])
                    m2 = re.search(value_regex, tail2, re.IGNORECASE)
                    if m2: return norm_spaces(m2.group(1))
    return None

# ----------------- Rasterizar PDF con PyMuPDF -----------------
def pdf_to_images_bytes(pdf_bytes: bytes, dpi: int = 300) -> list[Image.Image]:
    images = []
    # scale factor for desired dpi (PyMuPDF default ~72 dpi)
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

# ----------------- Extracción -----------------
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

    # Total
    total = None
    for _, ws in lines_dict(words).items():
        t = " ".join([w.text for w in ws])
        if re.search(r'\bTotal(?:\s*a\s*pagar)?\b', t, re.IGNORECASE):
            m = re.search(MONEY_PAT, t)
            if m: total = clean_money_to_float(m.group(1)); break
    out["monto_total"] = total
    return out

def extract_pdf_text(file_bytes: bytes) -> dict:
    out = {}
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            txt = "\n".join([p.extract_text() or "" for p in pdf.pages])
    except:
        txt = ""
    if not txt.strip(): return out
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

# ----------------- UI -----------------
st.title("🧾 Extractor de Facturas (OCR robusto, sin Poppler)")
st.caption("Rasteriza PDFs con PyMuPDF y usa OCR para leer valores a la derecha de las etiquetas (N°, Fecha Emis., Fecha Venc., Folio, Forma de pago).")

uploads = st.file_uploader("Sube uno o varios archivo_
