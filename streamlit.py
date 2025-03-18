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
            /* Removed the border-bottom that created the line */
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
        .result-row {
            background-color: #444;
            margin-bottom: 10px;
            padding: 15px;
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .category-text {
            text-align: left;
            color: #b5b5b5;
            font-weight: bold;
            font-size: 16px;
        }
        .percentage-text {
            text-align: right;
            color: #76c7c0;
            font-weight: bold;
            font-size: 16px;
        }
        /* Hide the default Streamlit file uploader label */
        .uploadedFile {
            display: none;
        }
        
    </style>
""", unsafe_allow_html=True)

# Main container for title (removed the border-bottom in the CSS above)
st.markdown('<div class="main-container"><h1>AI Resume Matcher</h1></div>', unsafe_allow_html=True)




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

        st.subheader("Top Matching Jobs:")
        
        # Simple rows with category on left and percentage on right
        for category, confidence in top_predictions:
            st.markdown(f"""
                <div class="result-row">
                    <div class="category-text">{category}</div>
                    <div class="percentage-text">{confidence:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)
