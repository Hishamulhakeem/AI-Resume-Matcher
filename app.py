from flask import Flask, request, render_template
import joblib
import fitz  # PyMuPDF
import numpy as np

app = Flask(__name__)

# Load the pre-trained model and vectorizer
model = joblib.load('resumeClassifier.pkl')
vectorizer = joblib.load('resumeVector.pkl')

# Function to extract text from a PDF file
def extract_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype='pdf')
    return ''.join(page.get_text() for page in doc)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'resume' not in request.files or request.files['resume'].filename == '':
        return render_template('index.html', error="Please upload a valid PDF file.")

    file = request.files['resume']
    resume_text = extract_text(file)

    # Transform the resume text for prediction
    X_input = vectorizer.transform([resume_text])
    y_pred = model.predict_proba(X_input)[0]

    # Get top 3 predicted job categories
    top_indices = np.argsort(y_pred)[-3:][::-1]
    top_predictions = [(model.classes_[i], y_pred[i] * 100) for i in top_indices]

    return render_template('index.html', predictions=top_predictions)

if __name__ == '__main__':
    app.run(debug=True)
