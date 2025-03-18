import streamlit as st
import joblib
import fitz  
import numpy as np

model = joblib.load('resumeClassifier.pkl')
vectorizer = joblib.load('resumeVector.pkl')

st.title("AI Resume Matcher")
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