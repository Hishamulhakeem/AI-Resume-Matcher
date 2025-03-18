import streamlit as st
import joblib
import fitz  # PyMuPDF
import numpy as np

# Load the pre-trained model and vectorizer
model = joblib.load('resumeClassifier.pkl')
vectorizer = joblib.load('resumeVector.pkl')

# Streamlit page configuration
st.set_page_config(page_title="AI Resume Matcher", layout="centered")

# Custom styling for a clean black-and-white theme with centered title
st.markdown("""
    <style>
        body {
            background-color: #1a1a1a;
            color: #f4f4f4;
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
        }
        .container {
            background-color: #262626;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0px 0px 20px rgba(255, 255, 255, 0.1);
            text-align: center;
            margin-top: 40px;
            margin-bottom: 30px;
            max-width: 500px;
            margin-left: auto;
            margin-right: auto;
        }
        h1 {
            margin-bottom: 25px;
            font-size: 32px;
            border-bottom: 2px solid #777;
            padding-bottom: 10px;
            letter-spacing: 1px;
            color: #f4f4f4;
            text-align: center;
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
        ul {
            list-style: none;
            padding: 0;
            margin-top: 25px;
        }
        li {
            background-color: #333;
            margin-bottom: 10px;
            padding: 10px;
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            color: #f4f4f4;
            font-weight: bold;
        }
        .category {
            text-align: left;
            color: #b5b5b5;
        }
        .confidence {
            text-align: right;
            color: #76c7c0;
        }
    </style>
""", unsafe_allow_html=True)

# Main container
st.markdown('<div class="container">', unsafe_allow_html=True)
st.markdown('<h1>AI Resume Matcher</h1>', unsafe_allow_html=True)

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
        st.markdown("<ul>", unsafe_allow_html=True)
        for category, confidence in top_predictions:
            st.markdown(f"<li><span class='category'>{category}</span> <span class='confidence'>{confidence:.2f}%</span></li>", unsafe_allow_html=True)
        st.markdown("</ul>", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
