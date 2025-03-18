import streamlit as st
import joblib
import fitz  # PyMuPDF
import numpy as np

model = joblib.load('resumeClassifier.pkl')
vectorizer = joblib.load('resumeVector.pkl')

st.set_page_config(page_title="AI Resume Matcher", layout="centered")

st.markdown("""
    <style>
    body { background-color: #f2f2f2; font-family: Arial, sans-serif; }
    .stApp { background-color: #1a1a1a; border-radius: 20px; padding: 40px; box-shadow: 0 12px 30px rgba(0, 0, 0, 0.7); color: #f4f4f4; max-width: 600px; margin: auto; }
    h1 { color: #f4f4f4; font-size: 28px; margin-bottom: 25px; border-bottom: 2px solid #666; padding-bottom: 10px; }
    input[type="file"] { padding: 8px; background-color: #2d2d2d; color: #fff; border: 1px solid #555; border-radius: 5px; margin-bottom: 20px; }
    button { margin-top: 20px; padding: 10px 20px; background-color: #666; color: white; border: none; border-radius: 8px; cursor: pointer; transition: 0.3s; font-weight: bold; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.4); }
    button:hover { background-color: #888; transform: scale(1.05); }
    </style>
""", unsafe_allow_html=True)

st.title("AI Resume Matcher 📄")
uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")

if uploaded_file:
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    resume_text = ''.join(page.get_text() for page in doc)

    X_input = vectorizer.transform([resume_text])
    y_pred = model.predict_proba(X_input)[0]
    top_indices = np.argsort(y_pred)[-3:][::-1]

    st.write("### Top Matching Jobs:")
    for i in top_indices:
        st.write(f"{model.classes_[i]}: {y_pred[i] * 100:.2f}%")
