import streamlit as st
import streamlit.components.v1 as components
import joblib
import fitz  
import numpy as np
import base64
from io import BytesIO

# Load the model and vectorizer
model = joblib.load('resumeClassifier.pkl')
vectorizer = joblib.load('resumeVector.pkl')

# Configure the page
st.set_page_config(page_title="AI Resume Matcher", layout="wide", initial_sidebar_state="collapsed")

# Function to extract text from PDF
def extract_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return ''.join(page.get_text() for page in doc)

# Function to get predictions
def get_predictions(file):
    resume_text = extract_text(file)
    X_input = vectorizer.transform([resume_text])
    y_pred = model.predict_proba(X_input)[0]
    top_indices = np.argsort(y_pred)[-3:][::-1]
    return [(model.classes_[i], y_pred[i] * 100) for i in top_indices]

# Session state for tracking
if 'analyze_clicked' not in st.session_state:
    st.session_state.analyze_clicked = False
    st.session_state.predictions = None
    st.session_state.file_uploaded = False

# Create HTML content
def get_html_content(file_name=None):
    has_predictions = st.session_state.predictions is not None
    
    html = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Resume Matcher</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f2f2f2;
                margin: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                overflow: hidden;
            }
            .container {
                background-color: #1a1a1a;
                border-radius: 20px;
                padding: 40px;
                width: 480px;
                box-shadow: 0 12px 30px rgba(0, 0, 0, 0.7);
                text-align: center;
                color: #f4f4f4;
            }
            h1 {
                margin-bottom: 25px;
                color: #f4f4f4;
                font-size: 28px;
                border-bottom: 2px solid #666;
                padding-bottom: 10px;
                letter-spacing: 1px;
            }
            .file-input {
                margin: 20px auto;
                width: 90%;
                text-align: center;
            }
            button {
                margin-top: 20px;
                padding: 12px 30px;
                background-color: #666;
                color: white;
                border: none;
                border-radius: 10px;
                cursor: pointer;
                transition: 0.3s;
                font-weight: bold;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.4);
            }
            button:hover {
                background-color: #888;
                transform: scale(1.05);
            }
            .results {
                margin-top: 20px;
                width: 100%;
            }
            .job-result {
                background-color: #333;
                margin-bottom: 12px;
                padding: 12px 15px;
                border-radius: 10px;
                font-weight: bold;
                display: flex;
                justify-content: space-between;
                color: #f4f4f4;
                transition: 0.3s;
            }
            .job-result:hover {
                background-color: #444;
            }
            .category {
                text-align: left;
                font-weight: 600;
                color: #b5b5b5;
            }
            .percent {
                text-align: right;
                color: #76c7c0;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>AI Resume Matcher</h1>
            
            <div class="file-input">
                <form id="upload-form">
    '''
    
    # Show file name if uploaded
    if file_name:
        html += f'''
                    <input type="file" id="resume-file" accept=".pdf" style="display: none;">
                    <label for="resume-file" style="background-color: #333; padding: 10px; border-radius: 5px; cursor: pointer; display: inline-block; width: 100%; text-align: left;">
                        {file_name}
                    </label>
        '''
    else:
        html += '''
                    <input type="file" id="resume-file" accept=".pdf">
        '''
    
    html += '''
                    <button type="button" id="analyze-button">Analyze Resume</button>
                </form>
            </div>
    '''
    
    # Add results section if we have predictions
    if has_predictions:
        html += '''
            <h2>Top Matching Jobs:</h2>
            <div class="results">
        '''
        
        for category, percent in st.session_state.predictions:
            html += f'''
                <div class="job-result">
                    <span class="category">{category}</span>
                    <span class="percent">{percent:.1f}%</span>
                </div>
            '''
        
        html += '''
            </div>
        '''
    
    html += '''
        </div>
        <script>
            document.getElementById('resume-file').addEventListener('change', function(e) {
                if (this.files.length > 0) {
                    const fileName = this.files[0].name;
                    // Pass the file name to Streamlit
                    parent.postMessage({
                        type: "streamlit:setComponentValue",
                        value: {action: "file_selected", name: fileName}
                    }, "*");
                }
            });
            
            document.getElementById('analyze-button').addEventListener('click', function() {
                const fileInput = document.getElementById('resume-file');
                if (fileInput.files.length > 0) {
                    // Tell Streamlit to analyze the file
                    parent.postMessage({
                        type: "streamlit:setComponentValue",
                        value: {action: "analyze"}
                    }, "*");
                } else {
                    alert("Please select a PDF file first");
                }
            });
        </script>
    </body>
    </html>
    '''
    return html

# Hide all Streamlit elements
hide_streamlit_style = """
<style>
    #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 0rem;}
    header {display: none !important;}
    #MainMenu {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    [data-testid="stToolbar"] {display: none !important;}
    [data-testid="stDecoration"] {display: none !important;}
    [data-testid="stStatusWidget"] {display: none !important;}
    [data-testid="stHeader"] {display: none !important;}
    [data-testid="stSidebar"] {display: none !important;}
    .stApp {
        background-color: white !important;
    }
    div[data-testid="stFileUploadDropzone"] {
        display: none;
    }
    .uploadedFile {
        display: none;
    }
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Create a hidden file uploader that we'll use to actually handle the file
file_uploader = st.file_uploader("Upload your resume", type="pdf", key="resume_file", label_visibility="collapsed")

# Process the file if it exists
if file_uploader is not None and not st.session_state.file_uploaded:
    st.session_state.file_uploaded = True
    st.session_state.file_name = file_uploader.name

# Render the HTML component
file_name = st.session_state.file_name if 'file_name' in st.session_state else None
component_value = components.html(get_html_content(file_name), height=700, scrolling=False)

# Handle component interactions
if component_value:
    if component_value.get('action') == 'file_selected':
        st.session_state.file_name = component_value.get('name')
        st.experimental_rerun()
    
    elif component_value.get('action') == 'analyze' and file_uploader is not None:
        # Reset the file position to the beginning
        file_uploader.seek(0)
        
        # Get predictions
        st.session_state.predictions = get_predictions(file_uploader)
        st.session_state.analyze_clicked = True
        st.experimental_rerun()
