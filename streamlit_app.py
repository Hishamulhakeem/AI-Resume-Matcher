import streamlit as st
import joblib
import fitz
import numpy as np

# Load model and vectorizer
model = joblib.load('resumeClassifier.pkl')
vectorizer = joblib.load('resumeVector.pkl')

# Set page config
st.set_page_config(page_title="AI Resume Matcher", layout="centered")

# Render index.html
with open("index.html", "r") as f:
    html_string = f.read()
st.components.v1.html(html_string, height=600)

# Rest of your code...
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
