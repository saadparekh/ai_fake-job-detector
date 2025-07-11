import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib

print("🔄 Loading dataset...")
df = pd.read_csv("fake_job_detection.csv", encoding='latin1')

# ✅ Inject smart fake jobs
print("📌 Injecting smart fake examples...")
smart_fakes = pd.DataFrame([
    {
        'title': 'Junior Data Analyst',
        'company_profile': 'Fast-growing global analytics firm.',
        'description': 'Urgent requirement for work-from-home analyst. Remote & flexible. Apply fast!',
        'requirements': 'No experience needed. Must have internet access.',
        'benefits': 'Flexible timing, Weekly payout',
        'employment_type': 'Contract',
        'required_experience': 'Internship',
        'required_education': 'High School or equivalent',
        'fraudulent': 1
    },
    {
        'title': 'Remote Writer',
        'company_profile': 'Online content marketing group.',
        'description': 'Write from home! Guaranteed income, no prior experience required.',
        'requirements': 'English fluency. Laptop. Available to start now.',
        'benefits': 'Get paid per task',
        'employment_type': 'Part-time',
        'required_experience': 'Internship',
        'required_education': 'None',
        'fraudulent': 1
    }
])

# Fill missing columns if needed
for col in df.columns:
    if col not in smart_fakes.columns:
        smart_fakes[col] = ""

df = pd.concat([df, smart_fakes], ignore_index=True)

# ✅ Fill missing values
text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits',
             'employment_type', 'required_experience', 'required_education']
for col in text_cols:
    df[col] = df[col].fillna("")

# ✅ Feature engineering
df['has_email'] = df['description'].apply(lambda x: int(bool(re.search(r"\S+@\S+", x))))
df['has_url'] = df['description'].apply(lambda x: int(bool(re.search(r"http[s]?://", x))))
df['has_salary'] = df['description'].apply(lambda x: int(bool(re.search(r"\$\d+", x))))
df['desc_len'] = df['description'].apply(lambda x: len(x.split()))

# ✅ Red flag count
red_flags = [
    'earn', 'no experience', 'apply now', 'bonus', 'instant', 'hiring fast', 'work from home', 'click here',
    'confidential projects', 'available now', 'urgent hiring', 'start immediately', 'flexible hours',
    'remote position', 'fast money', 'quick money', 'training provided', 'guaranteed income',
    'unlimited earnings', 'no interview', 'limited seats', '100% payout', 'easy money', 'weekly payout'
]

def count_red_flags(text):
    count = sum(1 for word in red_flags if word in text.lower())
    return count

df['red_flag_count'] = df['description'].apply(count_red_flags)

# ✅ Combine text for TF-IDF
combined_text = df['title'] + " " + df['company_profile'] + " " + df['description'] + " " + df['requirements'] + " " + df['benefits']
tfidf = TfidfVectorizer(max_features=1000)
X_text = tfidf.fit_transform(combined_text).toarray()

# ✅ Encode categorical features
cat_features = ['employment_type', 'required_experience', 'required_education']
X_cat = []
encoders = {}

for col in cat_features:
    le = LabelEncoder()
    encoded = le.fit_transform(df[col])
    encoders[col] = le
    X_cat.append(encoded.reshape(-1, 1))

X_cat = np.hstack(X_cat)

# ✅ Numeric features
custom_feats = df[['has_email', 'has_url', 'has_salary', 'desc_len', 'red_flag_count']].values

# ✅ Final dataset
X = np.hstack((X_text, X_cat, custom_feats))
y = df['fraudulent']
print(f"📊 Final feature shape: {X.shape}")

# ✅ Balance using SMOTE
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)
print("🔁 Data balanced using SMOTE.")

# ✅ Split train-test
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# ✅ Train RandomForest
model = RandomForestClassifier(n_estimators=200, random_state=42)
print("🧠 Training model...")
model.fit(X_train, y_train)

# ✅ Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_pred)

print(f"\n✅ Accuracy: {acc*100:.2f} %")
print(f"✅ ROC AUC Score: {roc:.4f}")
print("\n📄 Classification Report:\n", classification_report(y_test, y_pred))

# ✅ Save model
joblib.dump(model, 'real_model.pkl')
joblib.dump(tfidf, 'real_vectorizer.pkl')
joblib.dump(encoders, 'real_encoders.pkl')
print("💾 Model, vectorizer, and encoders saved.")
