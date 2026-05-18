import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Restaurant Rating Predictor", layout="wide")

st.title("Restaurant Rating Predictor")
st.caption("Estimate restaurant ratings using key features")

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    return pd.read_csv(r"D:\Internships\Cognifyz ML Internship - 2\Dataset .csv")

df = load_data()

# ---------- CLEANING ----------
cols_to_drop = ['Restaurant ID', 'Restaurant Name', 'Country Code', 'Address',
                'Locality', 'Locality Verbose', 'Rating color', 'Rating text',
                'Switch to order menu', 'Currency', 'Is delivering now']

df = df.drop([c for c in cols_to_drop if c in df.columns], axis=1)

target = "Aggregate rating"

numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(target)
cat_cols = df.select_dtypes(include=['object']).columns

# ---------- PREPROCESS ----------
imputer = SimpleImputer(strategy="median")
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

X = df.drop(target, axis=1)
y = df[target]

# ---------- MODEL ----------
@st.cache_resource
def train_model():
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model()

# ---------- PRICE RANGE MAPPING ----------
price_map = {
    "₹0–500": 1,
    "₹500–1000": 2,
    "₹1000–2000": 3,
    "₹2000+": 4
}

# ---------- INPUT UI ----------
st.subheader("Enter Restaurant Details")

col1, col2 = st.columns(2)
user_input = {}

for i, col in enumerate(X.columns):

    # PRICE RANGE DROPDOWN
    if col == "Price range":
        if i % 2 == 0:
            selected_price = col1.selectbox(col, list(price_map.keys()))
        else:
            selected_price = col2.selectbox(col, list(price_map.keys()))

        user_input[col] = price_map[selected_price]

    # NUMERIC INPUT
    elif col in numeric_cols:
        if i % 2 == 0:
            user_input[col] = col1.number_input(
                col,
                value=float(df[col].mean())
            )
        else:
            user_input[col] = col2.number_input(
                col,
                value=float(df[col].mean())
            )

    # CATEGORICAL INPUT
    else:
        options = list(encoders[col].classes_)
        if i % 2 == 0:
            user_input[col] = col1.selectbox(col, options)
        else:
            user_input[col] = col2.selectbox(col, options)

# ---------- PREDICTION ----------
if st.button("Predict Rating"):
    new_df = pd.DataFrame([user_input])

    for col in cat_cols:
        new_df[col] = encoders[col].transform(new_df[col])

    new_df[numeric_cols] = imputer.transform(new_df[numeric_cols])

    pred = model.predict(new_df)[0]

    if pred >= 4.5:
        label = "Excellent"
    elif pred >= 4:
        label = "Very Good"
    elif pred >= 3:
        label = "Good"
    else:
        label = "Average"

    st.success(f"Predicted Rating: {pred:.2f} ({label})")

# ---------- MODEL INSIGHTS ----------
st.markdown("---")
st.subheader("Model Insights")

y_pred = model.predict(X)

rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

colA, colB = st.columns(2)
colA.metric("RMSE", f"{rmse:.3f}")
colB.metric("R² Score", f"{r2:.3f}")

# Feature Importance
importances = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=True)

fig = px.bar(
    importances.tail(10),
    x="importance",
    y="feature",
    orientation="h",
    title="Top Factors Influencing Rating"
)

st.plotly_chart(fig, use_container_width=True)

# ---------- ABOUT ----------
with st.expander("About this model"):
    st.write("""
    This application uses a Random Forest Regressor to predict restaurant ratings.

    The model learns patterns from features such as cost, location, cuisine, and customer engagement (votes).
    Random Forest works well here because it captures non-linear relationships and reduces overfitting.

    The feature importance chart shows which inputs most influence the predicted rating.
    """)