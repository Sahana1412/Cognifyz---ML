import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("Task 1: Predict Restaurant Ratings")
df = pd.read_csv(r"D:\Internships\Cognifyz ML Internship - 2\Dataset .csv")  # e.g., 'zomato.csv'
print("Dataset loaded. Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nFirst 3 rows:")
print(df.head(3))
cols_to_drop = ['Restaurant ID', 'Restaurant Name', 'Country Code', 'Address', 'Locality', 'Locality Verbose', 
                'Rating color', 'Rating text', 'Switch to order menu', 'Currency', 'Is delivering now']
df = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)
target_col = 'Aggregate rating'
if target_col not in df.columns:
    print(f"ERROR: '{target_col}' not found! Available: {list(df.columns)}")
    exit()
numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(target_col)
cat_cols = df.select_dtypes(include=['object', 'string']).columns

print("\nNumeric features (%d): %s" % (len(numeric_cols), list(numeric_cols)))
print("Categorical features (%d): %s" % (len(cat_cols), list(cat_cols)))
imputer_num = SimpleImputer(strategy='median')
if len(numeric_cols) > 0:
    df[numeric_cols] = pd.DataFrame(imputer_num.fit_transform(df[numeric_cols]), 
                                    columns=numeric_cols, index=df.index)
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str).fillna('missing'))
    le_dict[col] = le
X = df.drop(target_col, axis=1)
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining set: %s, Test set: %s" % (X_train.shape, X_test.shape))

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model trained!")

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nModel Performance:")
print("MSE: %.4f" % mse)
print("R2: %.4f (%.1f%% explained)" % (r2, r2*100))

importances = pd.DataFrame({
    'feature': X.columns, 
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Features:")
print(importances.head(10).round(4))
plt.figure(figsize=(10, 6))
top10 = importances.head(10)
plt.barh(range(len(top10)), top10['importance'])
plt.yticks(range(len(top10)), top10['feature'])
plt.xlabel('Importance')
plt.title('Top 10 Features Affecting Ratings')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("\nEXAMPLE: New restaurant prediction")
new_restaurant = pd.DataFrame({
    'City': ['Makati City'],
    'Cuisines': ['French, Japanese, Desserts'],
    'Average Cost for two': [1100],
    'Has Table booking': ['Yes'],
    'Has Online delivery': ['No'],
    'Price range': [3],
    'Votes': [300],
    'Longitude': [121.027],
    'Latitude': [14.565]
})

print("New restaurant:")
print(new_restaurant)

for col in cat_cols:
    val = 'missing'
    if col in new_restaurant.columns:
        val = new_restaurant[col].iloc[0]
    new_restaurant[col] = [le_dict[col].transform([str(val)])[0]]

new_restaurant = new_restaurant.reindex(columns=X.columns, fill_value=0)
new_restaurant[numeric_cols] = imputer_num.transform(new_restaurant[numeric_cols])

prediction = model.predict(new_restaurant)[0]
rating_desc = "Excellent" if prediction > 4.5 else "Very Good" if prediction > 4 else "Good"
print("\nPredicted Rating: %.2f/5 (%s)" % (prediction, rating_desc))

print("\nComplete!")