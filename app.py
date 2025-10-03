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
    m = re.match(r'(\d{1,2})/(\d{1,2})/(\d{2,4})
