import streamlit as st
import joblib
import fitz  
import numpy as np

# Load the model and vectorizer
model = joblib.load('resumeClassifier.pkl')
vectorizer = joblib.load('resumeVector.pkl')

# Page configuration
st.set_page_config(page_title="AI Resume Matcher", layout="wide", initial_sidebar_state="collapsed")

# Function to extract text from PDF
def extract_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return ''.join(page.get_text() for page in doc)

# Initialize session state
if 'analyze_clicked' not in st.session_state:
    st.session_state.analyze_clicked = False
    st.session_state.predictions = None

# Hide all Streamlit elements
hide_streamlit_style = """
<style>
    #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 0rem;}
    header {display: none !important;}
    #MainMenu {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    [data-testid="stToolbar"] {display: none !important;}
    [data-testid="stDecoration"] {display: none !important;}
    [data-testid="stStatusWidget"] {display: none !important;}
    [data-testid="stHeader"] {display: none !important;}
    [data-testid="stSidebar"] {display: none !important;}
    .stApp {
        background-color: white !important;
    }
    div[data-testid="stFileUploadDropzone"] {
        display: none;
    }
    .uploadedFile {
        display: none;
    }
    
    /* Custom CSS for our app */
    .resume-container {
        background-color: #1a1a1a;
        border-radius: 20px;
        padding: 40px;
        width: 480px;
        margin: 100px auto;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.7);
        text-align: center;
        color: #f4f4f4;
    }
    .resume-title {
        margin-bottom: 25px;
        color: #f4f4f4;
        font-size: 28px;
        border-bottom: 2px solid #666;
        padding-bottom: 10px;
        letter-spacing: 1px;
    }
    .file-input-container {
        margin: 20px auto;
        width: 90%;
    }
    .analyze-button {
        margin-top: 20px;
        padding: 12px 30px;
        background-color: #666;
        color: white;
        border: none;
        border-radius: 10px;
        cursor: pointer;
        transition: 0.3s;
        font-weight: bold;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.4);
        width: 100%;
    }
    .analyze-button:hover {
        background-color: #888;
        transform: scale(1.05);
    }
    .job-result {
        background-color: #333;
        margin-bottom: 12px;
        padding: 12px 15px;
        border-radius: 10px;
        font-weight: bold;
        display: flex;
        justify-content: space-between;
        color: #f4f4f4;
    }
    .job-result:hover {
        background-color: #444;
    }
    .category {
        text-align: left;
        font-weight: 600;
        color: #b5b5b5;
    }
    .percent {
        text-align: right;
        color: #76c7c0;
        font-weight: bold;
    }
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Create a container for our app
st.markdown('<div class="resume-container">', unsafe_allow_html=True)
st.markdown('<h1 class="resume-title">AI Resume Matcher</h1>', unsafe_allow_html=True)

# Create a file uploader (hidden but functional)
uploaded_file = st.file_uploader("Upload your resume", type="pdf", key="resume_file", label_visibility="collapsed")

# Show the file input field
st.markdown('<div class="file-input-container">', unsafe_allow_html=True)

# Create our own file input display
if uploaded_file is not None:
    file_name = uploaded_file.name
else:
    file_name = "Choose File"

# Add a button to trigger analysis
def analyze_resume():
    st.session_state.analyze_clicked = True

# Display the analyze button
analyze_button = st.button("Analyze Resume", key="analyze_button", on_click=analyze_resume)

st.markdown('</div>', unsafe_allow_html=True)

# Process the file if button is clicked
if st.session_state.analyze_clicked and uploaded_file is not None:
    # Reset the file position to the beginning
    uploaded_file.seek(0)
    
    # Extract text and get predictions
    try:
        resume_text = extract_text(uploaded_file)
        X_input = vectorizer.transform([resume_text])
        y_pred = model.predict_proba(X_input)[0]
        top_indices = np.argsort(y_pred)[-3:][::-1]
        
        # Store predictions
        st.session_state.predictions = [(model.classes_[i], y_pred[i] * 100) for i in top_indices]
    except Exception as e:
        st.error(f"Error processing the file: {e}")

# Display results if we have predictions
if st.session_state.predictions:
    st.markdown('<h2>Top Matching Jobs:</h2>', unsafe_allow_html=True)
    
    for category, percent in st.session_state.predictions:
        st.markdown(
            f"""
            <div class="job-result">
                <span class="category">{category}</span>
                <span class="percent">{percent:.1f}%</span>
            </div>
            """,
            unsafe_allow_html=True
        )

# Close the container
st.markdown('</div>', unsafe_allow_html=True)

# Custom JavaScript to make the file input work better
st.markdown(
    """
    <script>
    const fileInput = document.querySelector('input[type="file"]');
    const customFileInput = document.querySelector('.file-input-container');
    
    customFileInput.addEventListener('click', function() {
        fileInput.click();
    });
    </script>
    """,
    unsafe_allow_html=True
)
