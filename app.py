import streamlit as st
import cv2
import numpy as np
import os
import subprocess
import sys

# ---------- DOWNLOAD MODELS FROM GOOGLE DRIVE ----------
MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Map your filenames to Google Drive FILE_IDs
# CHANGE THESE IDs TO YOUR ACTUAL ONES!
FILES_TO_DOWNLOAD = {
    "ShirorekhaNet_line.pth": "1611G4TwdgyB3zmCJqeeKfVgpEH99wbp6",    # Replace with your ID
    "AksharaNet_best.pth": "1n7AwHKf8tmaBYjUcYccpyKQoeicz-XRL",       # Replace with your ID
    "char_mapping.pkl": "1rbR5ebf4Jwfad2Vyzke1AtRAx1GCHaKw"           # Replace with your ID
}

def download_file(file_id, dest_path):
    """Download using gdown, only if file doesn't exist or is empty."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        return  # Already downloaded
    
    try:
        import gdown
        print(f"📥 Downloading {os.path.basename(dest_path)} from Google Drive...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, dest_path, quiet=False)
    except Exception as e:
        st.error(f"Failed to download {dest_path}: {e}")

# Download all required files
for filename, file_id in FILES_TO_DOWNLOAD.items():
    download_file(file_id, os.path.join(MODEL_DIR, filename))

# ---------- LOAD OCR ENGINE ----------
from ocr_engine import SanskritOCREngine

@st.cache_resource
def load_engine():
    return SanskritOCREngine(model_dir=MODEL_DIR)

engine = load_engine()

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Sanskrit OCR", layout="centered")
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
