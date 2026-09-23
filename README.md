**Scenario A: Fresh Start in a New Codespace / New Machine**
# 1. Navigate to the workspace (Codespaces auto-opens here, but just in case)
cd /workspaces/sanskrit-ocr-demo

# 2. (Optional) Verify Python version — should be 3.9 or higher
python --version

# 3. (Optional) Upgrade pip to the latest version
pip install --upgrade pip

# 4. Install all project dependencies from requirements.txt
#    (This installs PyTorch CPU, Streamlit, OpenCV, numpy, Pillow, gdown, etc.)
pip install -r requirements.txt

# 5. (Optional) Verify that the key packages installed correctly
pip show torch streamlit opencv-python-headless gdown

# 6. (Optional) Check that the models folder exists; it will be auto-created
#    at runtime if missing, but you can create it manually:
mkdir -p models

# 7. Launch the Streamlit app
streamlit run app.py

**Scenario B: Resuming After Codespace Auto-Stop**
# 1. Navigate to the workspace
cd /workspaces/sanskrit-ocr-demo

# 2. (Optional) Confirm models are still cached on disk
ls -lh models/

# 3. Restart the app
streamlit run app.py
