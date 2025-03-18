import streamlit as st
import streamlit.components.v1 as components
import joblib
import fitz  
import numpy as np
import os
import tempfile

# Load the model and vectorizer
model = joblib.load('resumeClassifier.pkl')
vectorizer = joblib.load('resumeVector.pkl')

# Create a full-page layout
st.set_page_config(page_title="AI Resume Matcher", layout="wide", initial_sidebar_state="collapsed")

# Function to extract text from PDF
def extract_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return ''.join(page.get_text() for page in doc)

# Function to create HTML with predictions
def get_html_content(predictions=None):
    html = '''
    <!DOCTYPE html>
    <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Resume Matcher</title>
            <style>
                body { font-family: Arial, sans-serif; background-color: #f2f2f2; background-image: radial-gradient(circle at center, #e6e6e6, #f2f2f2); margin: 0; display: flex; justify-content: center; align-items: center; min-height: 100vh; overflow: hidden; position: relative; }
                .background-decor { position: absolute; top: -100px; right: -100px; width: 300px; height: 300px; background-color: rgba(255, 255, 255, 0.05); border-radius: 50%; filter: blur(80px); }
                .background-decor2 { position: absolute; bottom: -100px; left: -100px; width: 300px; height: 300px; background-color: rgba(255, 255, 255, 0.05); border-radius: 50%; filter: blur(80px); }
                .container { background-color: #1a1a1a; border-radius: 20px; padding: 40px; width: 480px; box-shadow: 0 12px 30px rgba(0, 0, 0, 0.7); text-align: center; color: #f4f4f4; position: relative; z-index: 2; }
                h1 { margin-bottom: 25px; color: #f4f4f4; font-size: 28px; border-bottom: 2px solid #666; padding-bottom: 10px; letter-spacing: 1px; }
                input { margin-top: 20px; padding: 10px; background-color: #2d2d2d; color: #fff; border: 1px solid #555; border-radius: 5px; width: 90%; display: block; margin-left: auto; margin-right: auto; transition: 0.3s; }
                input:hover { background-color: #444; }
                button { margin-top: 20px; padding: 12px 30px; background-color: #666; color: white; border: none; border-radius: 10px; cursor: pointer; transition: 0.3s; font-weight: bold; display: block; margin-left: auto; margin-right: auto; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.4); }
                button:hover { background-color: #888; transform: scale(1.05); }
                ul { list-style: none; padding: 0; margin-top: 25px; }
                li { background-color: #333; margin-bottom: 12px; padding: 12px 15px; border-radius: 10px; font-weight: bold; display: flex; justify-content: space-between; color: #f4f4f4; transition: 0.3s; }
                li:hover { background-color: #444; }
                .category { text-align: left; font-weight: 600; color: #b5b5b5; }
                .percent { text-align: right; color: #76c7c0; font-weight: bold; }
            </style>
        </head>
    <body>
        <div class="background-decor"></div>
        <div class="background-decor2"></div>
        <div class="container">
            <h1>AI Resume Matcher</h1>
            <form id="upload-form">
                <input type="file" id="resume-file" accept=".pdf" required><br>
                <button type="button" id="analyze-button">Analyze Resume</button>
            </form>
    '''
    
    # Add predictions if available
    if predictions:
        html += '''
            <h2>Top Matching Jobs:</h2>
            <ul>
        '''
        for category, percent in predictions:
            html += f'''
                <li><span class="category">{category}</span><span class="percent">{percent:.2f}%</span></li>
            '''
        html += '''
            </ul>
        '''
    
    # Close HTML tags
    html += '''
        </div>
        <script>
            document.getElementById('analyze-button').addEventListener('click', function() {
                const fileInput = document.getElementById('resume-file');
                if (fileInput.files.length > 0) {
                    // Send a message to Streamlit
                    parent.postMessage({type: "streamlit:setComponentValue", value: true}, "*");
                } else {
                    alert("Please select a PDF file first");
                }
            });
        </script>
    </body>
    </html>
    '''
    return html

# Hide Streamlit elements
hide_st_style = """
<style>
    #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 0rem;}
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# Session state to track file upload and analysis
if 'file_processed' not in st.session_state:
    st.session_state.file_processed = False
    st.session_state.predictions = None

# Handle file upload via Streamlit's native uploader (hidden by default)
uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf", key="resume_uploader", label_visibility="collapsed")

# When a file is uploaded, process it
if uploaded_file and not st.session_state.file_processed:
    resume_text = extract_text(uploaded_file)
    X_input = vectorizer.transform([resume_text])
    y_pred = model.predict_proba(X_input)[0]
    top_indices = np.argsort(y_pred)[-3:][::-1]
    st.session_state.predictions = [(model.classes_[i], y_pred[i] * 100) for i in top_indices]
    st.session_state.file_processed = True

# Create HTML with or without predictions
html_content = get_html_content(st.session_state.predictions)

# Render the HTML
component_value = components.html(html_content, height=700, scrolling=False)

# Reset state if the analyze button was clicked
if component_value:
    st.session_state.file_processed = False
    st.rerun()
