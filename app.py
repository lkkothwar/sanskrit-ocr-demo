import streamlit as st

# ============================================
# CRITICAL: set_page_config MUST be the FIRST Streamlit command!
# ============================================
st.set_page_config(page_title="Sanskrit OCR", layout="centered")

# ============================================
# NOW import everything else
# ============================================
import cv2
import numpy as np
import os
import sys
import subprocess

# ---------- 1. INSTALL GDOWN AT RUNTIME (if missing) ----------
try:
    import gdown
except ImportError:
    st.warning("Installing gdown for model download...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
    import gdown

# ---------- 2. DOWNLOAD MODELS FROM GOOGLE DRIVE ----------
MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

# CHANGE THESE FILE IDs TO YOUR ACTUAL GOOGLE DRIVE IDs!
FILES_TO_DOWNLOAD = {
    "ShirorekhaNet_line.pth": "1611G4TwdgyB3zmCJqeeKfVgpEH99wbp6",    # Replace with your ID
    "AksharaNet_best.pth": "1n7AwHKf8tmaBYjUcYccpyKQoeicz-XRL",       # Replace with your ID
    "char_mapping.pkl": "1rbR5ebf4Jwfad2Vyzke1AtRAx1GCHaKw"           # Replace with your ID
}

def download_file(file_id, dest_path):
    """Download using gdown, only if file doesn't exist or is empty."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        return  # Already downloaded (cached)
    
    with st.spinner(f"Downloading {os.path.basename(dest_path)}..."):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, dest_path, quiet=False)

# Download all required files
for filename, file_id in FILES_TO_DOWNLOAD.items():
    download_file(file_id, os.path.join(MODEL_DIR, filename))

# ---------- 3. LOAD YOUR OCR ENGINE ----------
from ocr_engine import SanskritOCREngine

@st.cache_resource
def load_engine():
    return SanskritOCREngine(model_dir=MODEL_DIR)

engine = load_engine()

# ---------- 4. STREAMLIT UI ----------
st.title("📜 Sanskrit OCR Demo")
st.markdown("Upload a scanned printed Sanskrit page.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(img_bgr, channels="BGR", caption="Uploaded Image", width=400)
    
    with st.spinner("Recognizing Sanskrit text..."):
        result = engine.process(img_bgr)
    
    st.success("Recognition Complete!")
    st.text_area("Recognized Text", result["full_text"], height=300)
