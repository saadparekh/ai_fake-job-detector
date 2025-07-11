from flask import Flask, render_template, request
import joblib
import numpy as np
import re

app = Flask(__name__)

# Load model and tools
model = joblib.load("real_model.pkl")
vectorizer = joblib.load("real_vectorizer.pkl")
encoders = joblib.load("real_encoders.pkl")

# Red flag keywords
red_flags = [
    'earn', 'no experience', 'apply now', 'bonus', 'instant', 'hiring fast', 'work from home', 'click here',
    'confidential projects', 'available now', 'urgent hiring', 'start immediately', 'flexible hours',
    'remote position', 'fast money', 'quick money', 'training provided', 'guaranteed income',
    'unlimited earnings', 'no interview', 'limited seats', '100% payout', 'easy money', 'weekly payout'
]

def count_red_flags(text):
    return sum(1 for word in red_flags if word in text.lower())

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        title = request.form['title']
        company_profile = request.form['company_profile']
        description = request.form['description']
        requirements = request.form['requirements']
        benefits = request.form['benefits']
        emp_type = request.form['employment_type']
        experience = request.form['required_experience']
        education = request.form['required_education']

        # Red flag features
        has_email = int(bool(re.search(r"\S+@\S+", description)))
        has_url = int(bool(re.search(r"http[s]?://", description)))
        has_salary = int(bool(re.search(r"\$\d+", description)))
        desc_len = len(description.split())
        red_flag_count = count_red_flags(description)

        # Combine text for vectorization
        combined_text = f"{title} {company_profile} {description} {requirements} {benefits}"
        text_vector = vectorizer.transform([combined_text]).toarray()

        # Encode categorical
        cat_features = []
        for col, val in zip(['employment_type', 'required_experience', 'required_education'],
                            [emp_type, experience, education]):
            le = encoders[col]
            try:
                encoded_val = le.transform([val])[0]
            except:
                encoded_val = 0  # handle unknown category
            cat_features.append(encoded_val)
        
        # Final input
        extra_features = np.array([[has_email, has_url, has_salary, desc_len, red_flag_count]])
        final_input = np.hstack((text_vector, np.array(cat_features).reshape(1, -1), extra_features))

        # Predict
        proba = model.predict_proba(final_input)[0][1] * 100  # as percentage
        if proba >= 60:
            prediction_label = "FAKE"
        elif proba <= 40:
            prediction_label = "REAL"
        else:
            prediction_label = "⚠️ Suspicious / Unsure"

        return render_template("index.html", prediction=prediction_label, confidence=f"{proba:.2f}")
    
    except Exception as e:
        return f"⚠️ Something went wrong during prediction: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
