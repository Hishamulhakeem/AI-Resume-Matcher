import streamlit as st
import joblib
import fitz  # PyMuPDF
import numpy as np

model = joblib.load('resumeClassifier.pkl')
vectorizer = joblib.load('resumeVector.pkl')

st.set_page_config(page_title="AI Resume Matcher", layout="centered")

# Implementing the exact styling and structure from the provided screenshot
st.markdown("""
    <style>
        body { font-family: Arial, sans-serif; background-color: #1a1a1a; color: #f4f4f4; }
        .container { background-color: #262626; border-radius: 20px; padding: 40px; width: 500px; box-shadow: 0 12px 30px rgba(0, 0, 0, 0.7); text-align: center; margin-top: 50px; color: #f4f4f4; }
        h1 { margin-bottom: 25px; color: #f4f4f4; font-size: 32px; padding-bottom: 10px; letter-spacing: 1px; }
        .file-uploader { margin-top: 20px; background-color: #333; padding: 10px; border-radius: 10px; color: #fff; cursor: pointer; border: 2px dashed #555; margin-bottom: 20px; }
        .analyze-button { padding: 12px 30px; background-color: #666; color: white; border: none; border-radius: 10px; cursor: pointer; font-weight: bold; transition: 0.3s; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.4); display: block; margin: 20px auto; }
        .analyze-button:hover { background-color: #888; transform: scale(1.05); }
        ul { list-style: none; padding: 0; margin-top: 25px; }
        li { background-color: #333; margin-bottom: 12px; padding: 12px 15px; border-radius: 10px; display: flex; justify-content: space-between; color: #f4f4f4; transition: 0.3s; }
        li:hover { background-color: #444; }
        .category { font-weight: 600; color: #b5b5b5; }
        .percent { color: #76c7c0; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="container">
        <h1>AI Resume Matcher 📄</h1>
        <div class="file-uploader">Upload your resume (PDF)</div>
    </div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")
analyze_button = st.button("Analyze Resume", key="analyze_button")

if uploaded_file and analyze_button:
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    resume_text = ''.join(page.get_text() for page in doc)
    X_input = vectorizer.transform([resume_text])
    y_pred = model.predict_proba(X_input)[0]
    top_indices = np.argsort(y_pred)[-3:][::-1]

    st.markdown("<h2>Top Matching Jobs:</h2>", unsafe_allow_html=True)
    st.markdown("<ul>", unsafe_allow_html=True)
    for i in top_indices:
        st.markdown(f"<li><span class='category'>{model.classes_[i]}</span><span class='percent'>{y_pred[i] * 100:.2f}%</span></li>", unsafe_allow_html=True)
    st.markdown("</ul>", unsafe_allow_html=True)
