import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=== Task 2: Restaurant Recommendation (Fixed Readable Output) ===")

# Load ORIGINAL data + create copy for display
df_original = pd.read_csv(r"D:\Internships\Cognifyz ML Internship - 2\Dataset .csv")
df = df_original.copy()
print("Dataset loaded:", df.shape)

# Drop irrelevant (keep Restaurant Name for display)
cols_to_drop = ['Country Code', 'Address', 'Locality', 'Locality Verbose', 
                'Rating color', 'Rating text', 'Switch to order menu', 'Currency', 'Is delivering now']
df = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)

# Key features
rec_features = ['City', 'Cuisines', 'Price range', 'Average Cost for two', 'Aggregate rating']
rec_features = [f for f in rec_features if f in df.columns]
print("Features:", rec_features)

# Numeric/categorical
numeric_rec = df[rec_features].select_dtypes(include=[np.number]).columns
cat_rec = df[rec_features].select_dtypes(include=['object']).columns

# Impute numeric
imputer_num = SimpleImputer(strategy='median')
df[numeric_rec] = pd.DataFrame(imputer_num.fit_transform(df[numeric_rec]), columns=numeric_rec, index=df.index)

# Encode for similarity
le_dict = {}
df_encoded = df.copy()
for col in cat_rec:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col].astype(str).fillna('missing'))
    le_dict[col] = le

# Recommendation matrix
rec_df = df_encoded[rec_features].fillna(0)
scaler = StandardScaler()
rec_scaled = scaler.fit_transform(rec_df)

print("Matrix ready. Testing users...\n" + "="*70)

# User tests
user_tests = [
    {"name": "Budget Foodie", "City": "Makati City", "Cuisines": "Filipino", "Price_range": 1},
    {"name": "Fine Dining", "City": "Mandaluyong City", "Cuisines": "French, Japanese", "Price_range": 4},
    {"name": "Family", "City": "Pasay City", "Cuisines": "Asian", "Has_Table_booking": "Yes", "Price_range": 3}
]

for user_pref in user_tests:
    print(f"User: {user_pref['name']}")
    print(f"Prefs: City={user_pref.get('City')}, Cuisines={user_pref.get('Cuisines')}, Price={user_pref.get('Price_range', '?')}")
    
    # User profile (encoded)
    user_encoded = pd.DataFrame([{k: v for k, v in user_pref.items() if k != 'name' and k not in ['max_cost']}])
    for col in cat_rec:
        if col in user_encoded.columns:
            try:
                user_encoded[col] = le_dict[col].transform([str(user_encoded[col].iloc[0])])
            except:
                user_encoded[col] = le_dict[col].transform(['missing'])
        else:
            user_encoded[col] = le_dict[col].transform(['missing'])
    
    user_profile = user_encoded.reindex(columns=rec_features, fill_value=0).fillna(0)
    if len(numeric_rec) > 0:
        user_profile[numeric_rec] = imputer_num.transform(user_profile[numeric_rec])
    
    user_scaled = scaler.transform(user_profile)
    
    # Similarity scores
    sim_scores = cosine_similarity(user_scaled, rec_scaled)[0]
    
    # Get diverse top recs (remove duplicates by restaurant name)
    scores_df = pd.DataFrame({'index': range(len(sim_scores)), 'score': sim_scores})
    top_indices = scores_df.nlargest(20, 'score')['index'].tolist()  # More candidates for diversity
    
    # Filter unique restaurants, sort by score
    unique_recs = []
    seen_names = set()
    for idx in top_indices:
        name = df_original.iloc[idx]['Restaurant Name']
        if name not in seen_names:
            unique_recs.append((idx, sim_scores[idx]))
            seen_names.add(name)
        if len(unique_recs) >= 5:
            break
    
    print("Top 5 UNIQUE Recommendations:")
    for i, (idx, score) in enumerate(unique_recs):
        row = df_original.iloc[idx]
        print(f"  {i+1}. {row['Restaurant Name']} | {row['Cuisines'][:50]}... | "
              f"City: {row['City']} | Price: {row['Price range']} | Rating: {row['Aggregate rating']:.1f} | Sim: {score:.3f}")
    
    print()

print("Complete!")