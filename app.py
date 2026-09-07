import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ocr_engine import SanskritOCREngine  # Same engine file I gave you earlier!

st.set_page_config(page_title="Sanskrit OCR", layout="centered")
st.title("📜 Sanskrit OCR Demo")
st.markdown("Upload a scanned printed Sanskrit page.")

# Cache the model so it only loads once
@st.cache_resource
def load_engine():
    return SanskritOCREngine(model_dir="./models")

engine = load_engine()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    st.image(img_bgr, channels="BGR", caption="Uploaded Image", width=400)
    
    with st.spinner("Recognizing Sanskrit text..."):
        result = engine.process(img_bgr)
    
    st.success("Recognition Complete!")
    st.text_area("Recognized Text", result["full_text"], height=300)
