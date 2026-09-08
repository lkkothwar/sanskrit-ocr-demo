import streamlit as st

# ============================================
# CRITICAL: set_page_config MUST be the FIRST Streamlit command!
# ============================================
st.set_page_config(page_title="Sanskrit OCR", layout="wide")

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
st.markdown("Upload a scanned printed Sanskrit page. The system will detect lines, recognize text, and show you the segmentation overlays.")

# Sidebar for controls
with st.sidebar:
    st.header("Visualization Settings")
    show_line_viz = st.checkbox("Show Line Segmentation Overlay", value=True)
    show_word_viz = st.checkbox("Show Word Segmentation Overlay (slower)", value=False)
    st.markdown("---")
    st.caption("Made with ❤️ using PyTorch & Streamlit")

# Main upload area
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Display original image
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(img_bgr, channels="BGR", caption="Uploaded Image", use_container_width=True)
    
    # Run inference
    with st.spinner("Recognizing Sanskrit text..."):
        result = engine.process(img_bgr, show_line_viz=show_line_viz, show_word_viz=show_word_viz)
    
    # Show visualizations
    with col2:
        if show_line_viz and result["line_viz"] is not None:
            st.image(result["line_viz"], caption="Line Segmentation Overlay", use_container_width=True)
        elif show_word_viz and result["word_viz"] is not None:
            st.image(result["word_viz"], caption="Word Segmentation Overlay", use_container_width=True)
        else:
            st.info("No visualization selected. Enable them in the sidebar.")
    
    # Show recognized text
    st.success("Recognition Complete!")
    st.text_area("Recognized Sanskrit Text", result["full_text"], height=300, key="output_text")
