import streamlit as st
import joblib
import fitz  
import numpy as np

model = joblib.load('resumeClassifier.pkl')
vectorizer = joblib.load('resumeVector.pkl')

st.set_page_config(page_title="AI Resume Matcher", layout="centered")

st.markdown("""
    <style>
    body { font-family: Arial, sans-serif; background-color: #f2f2f2; background-image: radial-gradient(circle at center, #e6e6e6, #f2f2f2); margin: 0; display: flex; justify-content: center; align-items: center; min-height: 100vh; overflow: hidden; position: relative; }
    .stApp { background-color: #1a1a1a; border-radius: 20px; padding: 40px; box-shadow: 0 12px 30px rgba(0, 0, 0, 0.7); color: #f4f4f4; max-width: 600px; margin: 80px auto; position: relative; z-index: 2; }
    h1 { color: #f4f4f4; font-size: 28px; margin-bottom: 25px; border-bottom: 2px solid #666; padding-bottom: 10px; letter-spacing: 1px; }
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
""", unsafe_allow_html=True)

st.title("AI Resume Matcher ")
uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")

analyze_button = st.button("Analyze Resume")

if uploaded_file and analyze_button:
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    resume_text = ''.join(page.get_text() for page in doc)

    X_input = vectorizer.transform([resume_text])
    y_pred = model.predict_proba(X_input)[0]
    top_indices = np.argsort(y_pred)[-3:][::-1]

    st.write("### Top Matching Jobs:")
    for i in top_indices:
        st.write(f"{model.classes_[i]}: {y_pred[i] * 100:.2f}%")
