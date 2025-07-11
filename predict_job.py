import joblib
import numpy as np
import re

# Load model, vectorizer, and encoders
model = joblib.load("real_model.pkl")
vectorizer = joblib.load("real_vectorizer.pkl")
encoders = joblib.load("real_encoders.pkl")

print("\n🎯 Fake Job Detection System (Real-World Version)")
print("🔐 Fill required fields. Press Enter to skip optional fields.\n")

# Helper for required input
def get_required(prompt):
    val = input(prompt).strip()
    while not val:
        val = input(f"❗ This is required.\n{prompt}").strip()
    return val

# Helper for optional input
def get_optional(prompt):
    return input(prompt).strip() or "Unknown"

# Get inputs
title = get_required("Enter Job Title: ")
company_profile = get_required("Enter Company Profile: ")
description = get_required("Enter Job Description: ")
requirements = get_optional("Enter Job Requirements (optional): ")
benefits = get_optional("Enter Job Benefits (optional): ")

employment_type = get_optional("Employment Type (e.g., Full-time, Part-time): ")
required_experience = get_optional("Required Experience (e.g., Entry level): ")
required_education = get_optional("Required Education (e.g., High School, Bachelor): ")

# Text features
full_text = title + " " + company_profile + " " + description + " " + requirements + " " + benefits
X_text = vectorizer.transform([full_text]).toarray()

# Derived features
has_email = int(bool(re.search(r'\S+@\S+', description)))
has_url = int(bool(re.search(r'http[s]?://', description)))
has_salary = int(bool(re.search(r'\$\d+', description)))

# Encode categorical (optional) features
def encode_feature(col_name, value):
    encoder = encoders.get(col_name)
    if encoder is None:
        return 0
    if value not in encoder.classes_:
        encoder.classes_ = np.append(encoder.classes_, value)
    return encoder.transform([value])[0]

et_encoded = encode_feature('employment_type', employment_type)
exp_encoded = encode_feature('required_experience', required_experience)
edu_encoded = encode_feature('required_education', required_education)

# Combine features
X_other = np.array([[has_email, has_url, has_salary, et_encoded, exp_encoded, edu_encoded]])
X_final = np.concatenate((X_text, X_other), axis=1)

# Predict
pred = model.predict(X_final)[0]
proba = model.predict_proba(X_final)[0][pred]

# Output
print("\n🔍 Prediction Result:")
if pred == 1:
    print(f"⚠️ This job is likely **FAKE** (Confidence: {round(proba * 100, 2)}%)")
else:
    print(f"✅ This job is likely **REAL** (Confidence: {round(proba * 100, 2)}%)")
