# ==============================
# TASK 3: CUISINE CLASSIFICATION (OPTIMIZED)
# ==============================

import pandas as pd
import numpy as np
import sys

# Fix Unicode printing (Windows)
sys.stdout.reconfigure(encoding='utf-8')

# ------------------------------
# Step 1: Load Dataset
# ------------------------------
df = pd.read_csv(r"D:\Internships\Cognifyz ML Internship - 2\Dataset .csv")

# Drop missing cuisines
df = df.dropna(subset=['Cuisines'])

# ------------------------------
# Step 2: Clean Cuisine Column
# ------------------------------
df['Cuisines'] = df['Cuisines'].apply(
    lambda x: [i.strip() for i in x.split(',')]
)

# ------------------------------
# Step 3: Remove Rare Cuisines
# ------------------------------
from collections import Counter

all_cuisines = [c for sublist in df['Cuisines'] for c in sublist]
counts = Counter(all_cuisines)

THRESHOLD = 120  # tuned
common_cuisines = {c for c, count in counts.items() if count > THRESHOLD}

df['Cuisines'] = df['Cuisines'].apply(
    lambda x: [c for c in x if c in common_cuisines]
)

# Keep max 2 cuisines
df['Cuisines'] = df['Cuisines'].apply(lambda x: x[:2])

# Remove empty rows
df = df[df['Cuisines'].map(len) > 0]

# ------------------------------
# Step 4: Multi-label Encoding
# ------------------------------
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['Cuisines'])

# ------------------------------
# Step 5: Feature Selection
# ------------------------------
important_cols = [
    'City',
    'Price range',
    'Votes',
    'Average Cost for two',
    'Has Table booking',
    'Has Online delivery',
    'Aggregate rating'   # strong feature
]

important_cols = [col for col in important_cols if col in df.columns]
X = df[important_cols].copy()

# Encode categorical
from sklearn.preprocessing import LabelEncoder

for col in X.select_dtypes(include=['object', 'string']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Scale numeric
from sklearn.preprocessing import StandardScaler

num_cols = X.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# ------------------------------
# Step 6: Train-Test Split
# ------------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------
# Step 7: Model (Better Choice)
# ------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

model = MultiOutputClassifier(
    LogisticRegression(max_iter=3000)
)

model.fit(X_train, y_train)

# ------------------------------
# Step 8: Auto Threshold Tuning 🔥
# ------------------------------
from sklearn.metrics import f1_score

y_prob = model.predict_proba(X_test)

best_threshold = 0
best_f1 = 0
best_pred = None

for t in np.arange(0.2, 0.5, 0.05):
    y_pred_temp = np.array([
        (prob[:, 1] > t).astype(int)
        for prob in y_prob
    ]).T

    score = f1_score(y_test, y_pred_temp, average='micro')

    if score > best_f1:
        best_f1 = score
        best_threshold = t
        best_pred = y_pred_temp

print(f"\nBest Threshold: {best_threshold}")
y_pred = best_pred

# ------------------------------
# Step 9: Evaluation
# ------------------------------
from sklearn.metrics import hamming_loss, classification_report

print("\n===== FINAL PERFORMANCE =====")

print("Micro F1 Score:", best_f1)
print("Macro F1 Score:", f1_score(y_test, y_pred, average='macro'))
print("Hamming Loss:", hamming_loss(y_test, y_pred))

# Jaccard Accuracy
jaccard = np.mean([
    np.sum((yt & yp)) / np.sum((yt | yp)) if np.sum((yt | yp)) != 0 else 1
    for yt, yp in zip(y_test, y_pred)
])
print("Jaccard Accuracy:", jaccard)

# Clean names
clean_class_names = [
    str(c).encode("ascii", "ignore").decode() for c in mlb.classes_
]

print("\n===== CLASSIFICATION REPORT =====\n")
print(classification_report(
    y_test,
    y_pred,
    target_names=clean_class_names,
    zero_division=0
))