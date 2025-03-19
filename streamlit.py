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
        .result-row, .rating-section {
            background-color: #444;
            margin-bottom: 10px;
            padding: 15px;
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.5);
            margin-top: 20px;
            color: #fff;
        }
        .category-text {
            text-align: left;
            color: #b5b5b5;
            font-weight: bold;
            font-size: 16px;
        }
        .percentage-text, .rating-text {
            text-align: right;
            font-weight: bold;
            font-size: 16px;
        }
        .rating-section {
            justify-content: center;
            flex-direction: column;
            text-align: center;
            padding: 20px;
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

# Function to calculate resume rating
def rate(resume_text, y_pred):
    rating = round(np.max(y_pred) * 100, 2)
    color = '#ff4d4d' if rating < 40 else '#ffc107' if rating < 75 else '#4caf50'
    return rating, color

# Analyze button and prediction logic
if uploaded_file is not None:
    analyze_button = st.button("Analyze Resume", help="Click to analyze your resume!", key='analyze_button')

    if analyze_button:
        resume_text = extract_text(uploaded_file)

        # Predicting top 3 categories
        X_input = vectorizer.transform([resume_text])
        y_pred = model.predict_proba(X_input)[0]

        top_indices = np.argsort(y_pred)[-3:][::-1]
        top_predictions = [(model.classes_[i], y_pred[i] * 100) for i in top_indices]

        st.subheader("Top Matching Jobs:")

        # Display top predictions
        for category, confidence in top_predictions:
            st.markdown(f"""
                <div class="result-row">
                    <div class="category-text">{category}</div>
                    <div class="percentage-text" style='color: {'#ff4d4d' if confidence < 40 else '#ffc107' if confidence < 75 else '#4caf50'};'>{confidence:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)

        # Calculate and display resume rating
        rating, color = rate(resume_text, y_pred)
        st.markdown(f"""
        <div class="rating-section">
            <div class="rating-text" style="color: {color};">
                Resume Rating: {rating:.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
