.rating-section { background-color: #333; margin-bottom: 10px; padding: 20px; border-radius: 8px; display: flex; justify-content: center; align-items: center; box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.5); margin-top: 20px; color: #fff; font-weight: bold; }w_html=True)

# Main container for title
st.markdown('<div class="main-container"><h1>AI Resume Matcher</h1></div>', unsafe_allow_html=True)

# File uploader for resume (PDF format)
uploaded_file = st.file_uploader("Upload your resume (PDF format)", type=['pdf'])

# Function to extract text from the PDF
def extract_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype='pdf')
    return ''.join(page.get_text() for page in doc)

# Function to calculate resume rating
def rate(resume_text, y_pred):
    rating = round(np.max(y_pred) * 100, 2)
    color = '#ff4d4d' if rating < 40 else '#ffc107' if rating < 75 else '#4caf50'
    return rating, color

# Analyze button and prediction logic
if uploaded_file is not None:
    analyze_button = st.markdown("<div style='display: flex; justify-content: center; margin-top: 20px;'><button style='background-color: #000; color: #fff; padding: 10px 30px; border-radius: 10px; border: 2px solid #444; font-weight: bold; cursor: pointer;'>Analyze Resume</button></div>", unsafe_allow_html=True)

    if analyze_button:
        resume_text = extract_text(uploaded_file)

        # Predicting top 3 categories
        X_input = vectorizer.transform([resume_text])
        y_pred = model.predict_proba(X_input)[0]

        top_indices = np.argsort(y_pred)[-3:][::-1]
        top_predictions = [(model.classes_[i], y_pred[i] * 100) for i in top_indices]

        st.subheader("Top Matching Jobs:")

        # Display top predictions
        for category, confidence in top_predictions:
            st.markdown(f"""
                <div class="result-row">
                    <div class="category-text">{category}</div>
                    <div class="percentage-text" style='color: {'#ff4d4d' if confidence < 40 else '#ffc107' if confidence < 75 else '#4caf50'};'>{confidence:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)

        # Calculate and display resume rating
        rating, color = rate(resume_text, y_pred)
        st.markdown(f"""
        <div class="rating-section">
            <div class="rating-text" style="color: {color};">
                Resume Rating: {rating:.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
