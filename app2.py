import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# ---------- PAGE ----------
st.set_page_config(page_title="Restaurant Recommender", layout="wide")
st.title("Restaurant Recommendation System")
st.caption("Find restaurants based on your preferences")

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    return pd.read_csv(r"D:\Internships\Cognifyz ML Internship - 2\Dataset .csv")

df_original = load_data()
df = df_original.copy()

# ---------- CLEAN ----------
cols_to_drop = ['Country Code', 'Address', 'Locality', 'Locality Verbose',
                'Rating color', 'Rating text', 'Switch to order menu',
                'Currency', 'Is delivering now']

df = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)

# ---------- PRICE MAPPING ----------
price_map = {
    1: "₹0–500",
    2: "₹500–1000",
    3: "₹1000–2000",
    4: "₹2000+"
}
reverse_price_map = {v: k for k, v in price_map.items()}

# ---------- FEATURES ----------
rec_features = ['City', 'Cuisines', 'Price range',
                'Average Cost for two', 'Aggregate rating']
rec_features = [f for f in rec_features if f in df.columns]

numeric_rec = df[rec_features].select_dtypes(include=[np.number]).columns
cat_rec = df[rec_features].select_dtypes(include=['object']).columns

# ---------- PREPROCESS ----------
imputer = SimpleImputer(strategy='median')
df[numeric_rec] = imputer.fit_transform(df[numeric_rec])

le_dict = {}
df_encoded = df.copy()

for col in cat_rec:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col].astype(str).fillna('missing'))
    le_dict[col] = le

rec_df = df_encoded[rec_features].fillna(0)

scaler = StandardScaler()
rec_scaled = scaler.fit_transform(rec_df)

# ---------- USER INPUT ----------
st.subheader("Choose Your Preferences")

col1, col2, col3 = st.columns(3)

city = col1.selectbox("City", sorted(df['City'].dropna().unique()))
cuisine = col2.selectbox("Cuisine", sorted(df['Cuisines'].dropna().unique()))

price_display = col3.selectbox(
    "Price Range",
    ["₹0–500", "₹500–1000", "₹1000–2000", "₹2000+"]
)

price = reverse_price_map[price_display]

# ---------- RECOMMEND ----------
if st.button("Get Recommendations"):

    # FILTER BY CITY + CUISINE
    city_mask = df_original['City'] == city
    cuisine_mask = df_original['Cuisines'].str.contains(cuisine, case=False, na=False)

    df_city = df_encoded[city_mask & cuisine_mask]

    # fallback if no cuisine match
    if df_city.empty:
        st.warning("No exact cuisine match found. Showing similar restaurants in this city.")
        df_city = df_encoded[city_mask]

    if df_city.empty:
        st.warning("No restaurants found for this city.")
        st.stop()

    rec_city = df_city[rec_features].fillna(0)
    rec_scaled_city = scaler.transform(rec_city)

    # USER PROFILE
    user_input = pd.DataFrame([{
        "City": city,
        "Cuisines": cuisine,
        "Price range": price
    }])

    for col in cat_rec:
        try:
            user_input[col] = le_dict[col].transform(user_input[col])
        except:
            user_input[col] = 0

    user_profile = user_input.reindex(columns=rec_features, fill_value=0)

    if len(numeric_rec) > 0:
        user_profile[numeric_rec] = imputer.transform(user_profile[numeric_rec])

    user_scaled = scaler.transform(user_profile)

    # SIMILARITY
    sim_scores = cosine_similarity(user_scaled, rec_scaled_city)[0]

    city_indices = df_city.index

    scores_df = pd.DataFrame({
        "index": city_indices,
        "score": sim_scores
    })

    top_indices = scores_df.nlargest(20, "score")["index"].tolist()

    # UNIQUE RESULTS
    results = []
    seen = set()

    for idx in top_indices:
        name = df_original.loc[idx, "Restaurant Name"]

        if name not in seen:
            row = df_original.loc[idx]

            results.append({
                "Name": name,
                "Cuisine": row["Cuisines"],
                "Price": price_map.get(row["Price range"], row["Price range"]),
                "Rating": row["Aggregate rating"],
                "Votes": row.get("Votes", 0),
                "Similarity": round(scores_df[scores_df["index"] == idx]["score"].values[0], 3),
                "Latitude": row.get("Latitude", None),
                "Longitude": row.get("Longitude", None)
            })

            seen.add(name)

        if len(results) == 5:
            break

    rec_df_display = pd.DataFrame(results)

    # ---------- OUTPUT ----------
    st.subheader(f"Top Recommendations in {city}")
    st.dataframe(rec_df_display, use_container_width=True)

    # ---------- MAP ----------
    if "Latitude" in rec_df_display.columns and rec_df_display["Latitude"].notnull().any():
        st.subheader("Locations")
        st.map(rec_df_display.rename(columns={
            "Latitude": "lat",
            "Longitude": "lon"
        }))