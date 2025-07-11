# 🧠 Fake Job Detection AI Dashboard

This project is an AI-powered job screening tool that predicts whether a job posting is **Real**, **Fake**, or **Suspicious** using Natural Language Processing and Machine Learning. Built with a Flask backend, Random Forest classifier, and a clean, interactive dashboard.

---

## 🚀 Features

- ✅ Predict job authenticity from real-time user input
- 📋 Intelligent input validation (required fields only)
- 📊 Confidence-based predictions with visual feedback (pie chart)
- 📈 Graphs for Fake vs Real job trends
- 🧠 Model trained on Kaggle dataset with class imbalance handled
- 🎨 Responsive and attractive frontend (HTML + CSS)
- 🔐 Focus on real-world deployment and model reliability

---

## ⚙️ Tech Stack

- **Backend**: Python, Flask
- **ML/NLP**: scikit-learn, TF-IDF, Random Forest
- **Frontend**: HTML, CSS (internal), Chart.js
- **Tools**: Jupyter, VS Code

---

## 📁 Dataset Source

This project uses the official [Fake Job Postings dataset from Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting).

> 💡 Download the dataset and place it in the root folder as:
> `data/fake_job_postings.csv`


---

## ▶️ How to Run Locally

```bash
# Step 1: Install all required packages
pip install -r requirements.txt

# Step 2: Train the model (only once, or skip if pkl files provided)
python train_model.py

# Step 3: Launch the web app
python app.py

Open your browser and go to:
http://127.0.0.1:5000/

project/
├── app.py                   # Flask web app
├── train_model.py           # ML training pipeline
├── predict_job.py           # Input transformation logic
├── real_model.pkl           # Trained Random Forest model
├── real_vectorizer.pkl      # TF-IDF vectorizer
├── real_encoders.pkl        # Label encoders for categorical fields
├── fake_job_detection.csv   # Dataset (Kaggle)
├── templates/
│   └── index.html           # HTML frontend
├── requirements.txt         # Python libraries
├── README.md

## 📎 License
This project is open-source and free to use for learning and educational purposes.
Attribution is appreciated — feel free to fork, improve, and share 🙌
