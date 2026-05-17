# Restaurant Analytics & Intelligence System

A comprehensive data science project for restaurant analysis, prediction, and recommendation — built with Python, Scikit-learn, Streamlit, and modern web technologies.

---

## Project Overview

This project consists of four independent tasks that together form a full-stack restaurant intelligence system:

| Task | Description | Interface |
|------|-------------|-----------|
| Task 1 | Rating Prediction | Streamlit (`app.py`) |
| Task 2 | Recommendation System | Streamlit (`app2.py`) |
| Task 3 | Cuisine Classification | Streamlit (`app3.py`) |
| Task 4 | Location-Based Analysis | Browser (`index.html`) |


## Tasks

### Task 1 — Restaurant Rating Prediction

A regression model that predicts the aggregate rating of a restaurant based on features such as cost, location, and customer engagement.

**Key Details:**
- Data preprocessing including missing value handling and feature encoding
- 80–20 train-test split
- Model: Random Forest Regressor
- Evaluation metrics: RMSE and R² score
- Feature importance analysis
- Interactive prediction interface built with Streamlit

---

### Task 2 — Restaurant Recommendation System

A content-based recommendation system that suggests restaurants based on user preferences.

**Key Details:**
- Input parameters: city, cuisine, and price range
- Feature scaling and cosine similarity for matching
- Filtering and ranking based on similarity scores
- Graceful handling of missing or unmatched preferences
- Displays top recommendations with location visualization

---

### Task 3 — Cuisine Classification

A multi-label classification model that predicts possible cuisines for a restaurant.

**Key Details:**
- Multi-label classification using MultiOutput Logistic Regression
- Feature encoding and scaling
- Evaluation metrics: accuracy, precision, and recall
- Adjustable prediction threshold
- Supports multiple cuisine predictions per input

---

### Task 4 — Location-Based Analysis Dashboard

A data visualization dashboard that analyzes the geographical distribution of restaurants and identifies patterns across cities.

**Key Details:**
- Restaurant location visualization using latitude and longitude
- City-level aggregation of:
  - Number of restaurants
  - Average rating
  - Price range
  - Online delivery availability
- Comparative analysis across cities using tables and charts
- Global cuisine distribution visualization
- Key insights derived from aggregated data
- Built with HTML, CSS, Chart.js, and D3.js — no server required

---

## Technologies Used

**Backend & ML**
- Python, Pandas, NumPy
- Scikit-learn
- Streamlit, Plotly, Matplotlib

**Frontend (Task 4)**
- HTML, CSS, JavaScript
- Chart.js, D3.js

---

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Streamlit Applications

Each task runs as a separate Streamlit app:

```bash
# Task 1 — Rating Prediction
streamlit run app.py

# Task 2 — Recommendation System
streamlit run app2.py

# Task 3 — Cuisine Classification
streamlit run app3.py
```

### 3. Run the Location Analysis Dashboard

Open `index.html` directly in any modern web browser — no server setup required.

---

## Notes

- All models are trained using an 80–20 train-test split.
- Categorical features are encoded using label encoding.
- Numerical features are scaled where required.
- For the location analysis dashboard, data is pre-aggregated at the city level for efficient rendering.
