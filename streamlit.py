import streamlit as st
import joblib
import fitz  # PyMuPDF
import numpy as np

# Load the pre-trained model and vectorizer
model = joblib.load('resumeClassifier.pkl')
vectorizer = joblib.load('resumeVector.pkl')

# Streamlit page configuration
st.set_page_config(page_title="AI Resume Matcher", layout="centered")
st.markdown("""
    <style>
        body { font-family: Arial, sans-serif; background-color: #f2f2f2; background-image: radial-gradient(circle at center, #e6e6e6, #f2f2f2); }
        .container { background-color: #1a1a1a; border-radius: 20px; padding: 40px; width: 500px; box-shadow: 0 12px 30px rgba(0, 0, 0, 0.7); text-align: center; color: #f4f4f4; margin-top: 50px; }
        h1 { color: #f4f4f4; margin-bottom: 25px; }
        input { margin-top: 20px; padding: 10px; background-color: #2d2d2d; color: #fff; border: 1px solid #555; border-radius: 5px; width: 90%; display: block; margin-left: auto; margin-right: auto; }
        input:hover { background-color: #444; }
        button { margin-top: 20px; padding: 12px 30px; background-color: #666; color: white; border: none; border-radius: 10px; cursor: pointer; }
        button:hover { background-color: #888; transform: scale(1.05); }
        ul { list-style: none; padding: 0; margin-top: 25px; }
        li { background-color: #333; margin-bottom: 12px; padding: 12px 15px; border-radius: 10px; font-weight: bold; color: #f4f4f4; display: flex; justify-content: space-between; }
        .category { text-align: left; color: #b5b5b5; }
        .percent { text-align: right; color: #76c7c0; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.markdown('<div class="container">', unsafe_allow_html=True)
st.markdown('<h1>AI Resume Matcher</h1>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload your resume (PDF format):", type=['pdf'])

# Function to extract text from a PDF file
def extract_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype='pdf')
    return ''.join(page.get_text() for page in doc)

    # Predict top 3 categories
    X_input = vectorizer.transform([resume_text])
    y_pred = model.predict_proba(X_input)[0]
    top_indices = np.argsort(y_pred)[-3:][::-1]
    top_predictions = [(model.classes_[i], y_pred[i] * 100) for i in top_indices]

    st.subheader("Top Matching Jobs:")
    st.markdown("<ul>", unsafe_allow_html=True)
    for category, confidence in top_predictions:
        st.markdown(f"<li><span class='category'>{category}</span><span class='percent'>{confidence:.2f}%</span></li>", unsafe_allow_html=True)
    st.markdown("</ul>", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
