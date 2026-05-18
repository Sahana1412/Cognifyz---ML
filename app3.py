# ==============================
# RESTAURANT CUISINE CLASSIFICATION DASHBOARD
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(page_title="Cuisine Classification Dashboard", layout="wide")

st.title("Restaurant Cuisine Classification")
st.caption("Predict restaurant cuisines using machine learning")

# ------------------------------
# Load Data
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(r"D:\Internships\Cognifyz ML Internship - 2\Dataset .csv")
    df = df.dropna(subset=['Cuisines'])
    df['Cuisines'] = df['Cuisines'].apply(lambda x: [i.strip() for i in x.split(',')])
    return df

df = load_data()

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.header("Controls")
threshold = st.sidebar.slider("Prediction Threshold", 0.1, 0.5, 0.3, 0.05)
min_freq = st.sidebar.slider("Minimum Cuisine Frequency", 50, 200, 120)

# ------------------------------
# Filter Cuisines
# ------------------------------
all_cuisines = [c for sublist in df['Cuisines'] for c in sublist]
counts = Counter(all_cuisines)

common = {c for c, count in counts.items() if count > min_freq}

df['Cuisines'] = df['Cuisines'].apply(lambda x: [c for c in x if c in common])
df['Cuisines'] = df['Cuisines'].apply(lambda x: x[:2])
df = df[df['Cuisines'].map(len) > 0]

# ------------------------------
# Encoding
# ------------------------------
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['Cuisines'])

# ------------------------------
# Features
# ------------------------------
cols = [
    'City',
    'Price range',
    'Votes',
    'Average Cost for two',
    'Aggregate rating'
]

cols = [c for c in cols if c in df.columns]
X = df[cols].copy()

# ------------------------------
# Encode categorical
# ------------------------------
from sklearn.preprocessing import LabelEncoder

encoders = {}
for col in X.select_dtypes(include=['object']):
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

# ------------------------------
# Scale
# ------------------------------
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------
# Split
# ------------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ------------------------------
# Model
# ------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

model = MultiOutputClassifier(LogisticRegression(max_iter=2000))
model.fit(X_train, y_train)

# ------------------------------
# Predictions
# ------------------------------
y_prob = model.predict_proba(X_test)

y_pred = np.array([
    (prob[:, 1] > threshold).astype(int)
    for prob in y_prob
]).T

# ------------------------------
# Metrics
# ------------------------------
from sklearn.metrics import accuracy_score, precision_score, recall_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='micro', zero_division=0)
recall = recall_score(y_test, y_pred, average='micro', zero_division=0)

st.subheader("Model Performance")

c1, c2, c3 = st.columns(3)
c1.metric("Accuracy", round(accuracy, 3))
c2.metric("Precision", round(precision, 3))
c3.metric("Recall", round(recall, 3))

# ------------------------------
# Feature Importance
# ------------------------------
st.subheader("Feature Importance")

feature_names = cols

coef_list = []
for estimator in model.estimators_:
    coef_list.append(estimator.coef_[0])

coef_array = np.array(coef_list)
importance = np.mean(np.abs(coef_array), axis=0)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
}).sort_values("Importance", ascending=True)

import plotly.express as px

fig = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h"
)

st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# Cuisine Distribution
# ------------------------------
st.subheader("Top Cuisines")

import matplotlib.pyplot as plt

top_cuisines = Counter([
    c for sublist in df['Cuisines'] for c in sublist
]).most_common(10)

labels, values = zip(*top_cuisines)

fig2, ax = plt.subplots()
ax.barh(labels, values)
st.pyplot(fig2)

# ------------------------------
# Prediction UI
# ------------------------------
st.subheader("Cuisine Prediction")

input_data = {}

price_map = {
    1: "₹0–500",
    2: "₹500–1000",
    3: "₹1000–2000",
    4: "₹2000+"
}
reverse_price_map = {v: k for k, v in price_map.items()}

for col in cols:
    if col == "Price range":
        selected = st.selectbox(col, list(price_map.values()))
        input_data[col] = reverse_price_map[selected]

    elif col in encoders:
        input_data[col] = st.selectbox(col, encoders[col].classes_)

    elif col == "Average Cost for two":
        input_data[col] = st.slider(col, 100.0, 3000.0, 1000.0, step=100.0)

    else:
        input_data[col] = st.slider(
            col,
            float(df[col].min()),
            float(df[col].max()),
            float(df[col].mean())
        )

# ------------------------------
# Predict (FIXED)
# ------------------------------
if st.button("Predict Cuisine"):
    input_df = pd.DataFrame([input_data])

    for col in encoders:
        input_df[col] = encoders[col].transform(input_df[col])

    input_scaled = scaler.transform(input_df)

    probs = model.predict_proba(input_scaled)

    prob_values = np.array([p[0][1] for p in probs])

    pred = [
        mlb.classes_[i]
        for i, p in enumerate(prob_values)
        if p > threshold
    ]

    # FIX: always show something
    if not pred:
        top_idx = np.argsort(prob_values)[-2:]
        pred = [mlb.classes_[i] for i in top_idx]

    st.markdown(f"""
<div style="
    background-color:#0a0a0a;
    border:1px solid #2a2a2a;
    padding:18px;
    border-left:4px solid #00b8d9;
    font-family:'IBM Plex Mono', monospace;
    color:#faf9f6;
    margin-top:10px;
">
    <div style="font-size:12px; letter-spacing:0.12em; color:#00b8d9; margin-bottom:6px;">
        PREDICTED CUISINES
    </div>
    <div style="font-size:16px; font-weight:500;">
        {", ".join(pred)}
    </div>
</div>
""", unsafe_allow_html=True)