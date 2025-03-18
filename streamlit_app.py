import streamlit as st
import joblib
import fitz  
import numpy as np
import base64

# Load the model and vectorizer
model = joblib.load('resumeClassifier.pkl')
vectorizer = joblib.load('resumeVector.pkl')

# Page configuration
st.set_page_config(page_title="AI Resume Matcher", layout="centered")

# Custom CSS to match the index.html design
st.markdown("""
    <style>
    .stApp {
        background-image: radial-gradient(circle at center, #e6e6e6, #f2f2f2);
    }
    
    .main-container {
        background-color: #1a1a1a;
        border-radius: 20px;
        padding: 40px;
        max-width: 480px;
        margin: auto;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.7);
        color: #f4f4f4;
        position: relative;
        z-index: 2;
    }
    
    .title {
        margin-bottom: 25px;
        color: #f4f4f4;
        font-size: 28px;
        border-bottom: 2px solid #666;
        padding-bottom: 10px;
        letter-spacing: 1px;
        text-align: center;
    }
    
    .file-uploader {
        margin-top: 20px;
        background-color: #2d2d2d;
        border: 1px solid #555;
        border-radius: 5px;
        width: 100%;
        transition: 0.3s;
    }
    
    .stButton > button {
        margin-top: 20px;
        padding: 12px 30px;
        background-color: #666;
        color: white;
        border: none;
        border-radius: 10px;
        cursor: pointer;
        transition: 0.3s;
        font-weight: bold;
        display: block;
        width: 100%;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.4);
    }
    
    .stButton > button:hover {
        background-color: #888;
        transform: scale(1.05);
    }
    
    .result-item {
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
    
    .result-item:hover {
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
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Background decorations */
    .background-decor {
        position: fixed;
        top: -100px;
        right: -100px;
        width: 300px;
        height: 300px;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 50%;
        filter: blur(80px);
        z-index: 1;
    }
    
    .background-decor2 {
        position: fixed;
        bottom: -100px;
        left: -100px;
        width: 300px;
        height: 300px;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 50%;
        filter: blur(80px);
        z-index: 1;
    }
    </style>
    
    <!-- Background decoration elements -->
    <div class="background-decor"></div>
    <div class="background-decor2"></div>
""", unsafe_allow_html=True)

# Custom container to match the centered box in index.html
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown('<h1 class="title">AI Resume Matcher</h1>', unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf", key="resume_uploader")

# Only show the analyze button when a file is uploaded
analyze_clicked = False
if uploaded_file:
    analyze_clicked = st.button("Analyze Resume")

# Process the file and show results
if uploaded_file and analyze_clicked:
    with st.spinner("Analyzing your resume..."):
        try:
            # Extract text from PDF
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            resume_text = ''.join(page.get_text() for page in doc)
            
            # Generate predictions
            X_input = vectorizer.transform([resume_text])
            y_pred = model.predict_proba(X_input)[0]
            top_indices = np.argsort(y_pred)[-3:][::-1]
            
            # Display results
            st.markdown("<h2 style='text-align: center; margin-top: 25px;'>Top Matching Jobs:</h2>", unsafe_allow_html=True)
            
            for i in top_indices:
                job = model.classes_[i]
                percentage = y_pred[i] * 100
                st.markdown(
                    f"""
                    <div class="result-item">
                        <span class="category">{job}</span>
                        <span class="percent">{percentage:.2f}%</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        except Exception as e:
            st.error(f"Error processing the file: {e}")

# Close the main container div
st.markdown('</div>', unsafe_allow_html=True)
