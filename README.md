AI Resume Matcher 📄🔍

AI Resume Matcher is a Flask-based web application that uses machine learning to classify resumes into different job categories based on their content. It processes PDF resumes, extracts text, and predicts the top 3 most relevant job categories.

⭕ Note: The app.py and index.html files were generated using ChatGPT and then used in this project.

🚀 Features

🏗 Upload a PDF resume and get job category predictions.

🔍 Uses TF-IDF vectorization and a trained ML model.

📊 Displays the top 3 predicted categories with probabilities.

🖥️ Simple and visually appealing user interface.

📂 Project Structure

AI-Resume-Matcher/
│── app.py               # Flask application backend (Generated using ChatGPT)

│── index.html           # Frontend UI (Generated using ChatGPT)

│── AI Matcher.ipynb     # Jupyter Notebook for model training

│── Resume_dataset.csv   # Dataset used for training

│── resumeClassifier.pkl # Trained ML model

│── resumeVector.pkl     # TF-IDF vectorizer

│── requirements.txt     # Required dependencies

│── uploads/             # Folder to store uploaded resumes

🧠 Model Training

The model is trained using:

Dataset: Resume_dataset.csv

Feature Extraction: TF-IDF Vectorization

Classifier: Scikit-Learn's RandomForestClassifier (or another algorithm used)

Training Notebook: AI Matcher.ipynb

Accuracy Achieved: 99.48196%

🔥 How It Works

Upload a resume (only pdf file).

The system extracts text using PyMuPDF.

The extracted text is transformed into features using TF-IDF.

The trained classifier predicts the top 3 job categories.

Results are displayed with probability scores.

🎨 UI Preview

The web UI is designed with a clean and modern look.

🤝 Contributing

Feel free to submit issues or pull requests. Any improvements to the UI or model are welcome!

📜 License

This project is licensed under the MIT License.
