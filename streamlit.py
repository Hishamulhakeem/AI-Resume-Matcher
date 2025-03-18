import streamlit as st
import joblib
import fitz  # PyMuPDF
import numpy as np

# Load the pre-trained model and vectorizer
model = joblib.load('resumeClassifier.pkl')
vectorizer = joblib.load('resumeVector.pkl')

# Streamlit page configuration
st.set_page_config(page_title="AI Resume Matcher", layout="centered")

# Custom styling
st.markdown("""
    <style>
        body {
            background-color: #1a1a1a;
            color: #f4f4f4;
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
        }
        .main-container {
            margin-top: 30px;
            padding: 40px;
            text-align: center;
        }
        h1 {
            margin-top: 20px;
            margin-bottom: 30px;
            font-size: 36px;
            font-weight: bold;
            color: #f4f4f4;
            border-bottom: 2px solid #777;
            padding-bottom: 10px;
            letter-spacing: 1px;
        }
        .container {
            background-color: #262626;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0px 0px 20px rgba(255, 255, 255, 0.1);
            margin-top: 40px;
            margin-bottom: 30px;
            max-width: 500px;
            margin-left: auto;
            margin-right: auto;
        }
        .stButton button {
            background-color: #000;
            color: #fff;
            padding: 10px 30px;
            border-radius: 10px;
            border: 2px solid #444;
            font-weight: bold;
            margin-top: 20px;
            margin-bottom: 20px;
            transition: 0.3s;
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
        .stButton button:hover {
            background-color: #333;
            border-color: #777;
            transform: scale(1.05);
        }
        .category-box {
            background-color: #444;
            margin-bottom: 15px;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        .category-name {
            color: #b5b5b5;
            font-weight: bold;
            font-size: 18px;
            margin-bottom: 5px;
        }
        .percentage-bar {
            background-color: #333;
            border-radius: 10px;
            height: 25px;
            margin-top: 8px;
            position: relative;
            overflow: hidden;
        }
        .percentage-fill {
            background-color: #76c7c0;
            height: 100%;
            border-radius: 10px;
            text-align: right;
            padding-right: 10px;
            line-height: 25px;
            color: #fff;
            font-weight: bold;
            transition: width 1s ease-in-out;
        }
    </style>
""", unsafe_allow_html=True)

# Main container for title
st.markdown('<div class="main-container"><h1>AI Resume Matcher</h1></div>', unsafe_allow_html=True)

# File uploader for resume (PDF format)
uploaded_file = st.file_uploader("Upload your resume (PDF format)", type=['pdf'])

# Function to extract text from the PDF
def extract_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype='pdf')
    return ''.join(page.get_text() for page in doc)

# Analyze button and prediction logic
if uploaded_file is not None:
    analyze_button = st.button("Analyze Resume")

    if analyze_button:
        resume_text = extract_text(uploaded_file)

        # Predicting top 3 categories
        X_input = vectorizer.transform([resume_text])
        y_pred = model.predict_proba(X_input)[0]

        top_indices = np.argsort(y_pred)[-3:][::-1]
        top_predictions = [(model.classes_[i], y_pred[i] * 100) for i in top_indices]

        st.subheader("Top Matching Job Categories:")
        
        # Individual boxes for each category
        for category, confidence in top_predictions:
            st.markdown(f"""
                <div class="category-box">
                    <div class="category-name">{category}</div>
                    <div class="percentage-bar">
                        <div class="percentage-fill" style="width: {confidence}%;">{confidence:.2f}%</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
