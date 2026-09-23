1. Open a Codespace
Go to your GitHub repository.

Click Code → Codespaces → Create codespace on main.

2. Install Dependencies
   
   In the Codespaces terminal :  pip install -r requirements.txt

3. Run the App :  streamlit run app.py

5. Make the Port Public :  Click the Ports tab (next to the Terminal).

Right-click on port 8501.

Select Port Visibility → Public.

Copy the generated public URL (e.g., https://automatic-goldfish-xxxx-8501.app.github.dev/).

Share this URL — anyone can access your running demo while the Codespace is active.

⚠️ Important: GitHub Codespaces automatically stops after 30 minutes of inactivity. Restart it and re-run streamlit run app.py to reactivate the link.
