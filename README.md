Tasks Implemented
Task 1: Restaurant Rating Prediction
A regression model that predicts the aggregate rating of a restaurant based on features such as cost, location, and customer engagement.
Key points:
•	Data preprocessing with missing value handling and encoding 
•	Model built using Random Forest Regressor 
•	Train-test split (80–20) for evaluation 
•	Performance measured using RMSE and R² score 
•	Feature importance analysis to identify key influencing factors 
•	Interactive interface for user-based predictions 
________________________________________
Task 2: Restaurant Recommendation System
A content-based recommendation system that suggests restaurants based on user preferences such as city, cuisine, and price range.
Key points:
•	Filtering based on user-selected preferences 
•	Feature scaling and cosine similarity for recommendations 
•	Handles missing or unmatched inputs gracefully 
•	Displays top recommendations with similarity scores 
•	Includes location-based visualization of recommended restaurants 
________________________________________
Task 3: Cuisine Classification
A multi-label classification model that predicts possible cuisines for a restaurant.
Key points:
•	Multi-label classification using MultiOutput Logistic Regression 
•	Preprocessing of categorical and numerical features 
•	Evaluation using accuracy, precision, and recall 
•	Adjustable prediction threshold for tuning results 
•	Handles multiple cuisine predictions per input 
________________________________________
Task 4: Location-Based Analysis
A data visualization dashboard that analyzes the geographical distribution of restaurants and identifies patterns across cities.
Key points:
•	Visualization of restaurant locations using latitude and longitude on a world map 
•	Aggregation of data at the city level to compute statistics such as: 
o	number of restaurants 
o	average rating 
o	price range 
o	delivery availability 
•	Comparative analysis across cities using tables and charts 
•	Global cuisine distribution analysis 
•	Identification of key insights and patterns based on location data 
•	Interactive dashboard built using HTML, CSS, Chart.js, and D3.js 
________________________________________
Technologies Used
•	Python 
•	Pandas, NumPy 
•	Scikit-learn 
•	Streamlit 
•	Plotly and Matplotlib 
•	HTML, CSS, JavaScript 
•	Chart.js and D3.js 
________________________________________
How to Run
1.	Install required dependencies: 
pip install -r requirements.txt
2.	Run the Streamlit applications: 
streamlit run app.py
streamlit run app2.py
streamlit run app3.py
3.	Open the location analysis dashboard: 
•	Simply open index.html in a browser 
4.	Ensure the dataset file is placed in the correct directory where required. 
________________________________________
Notes
•	Models are trained using an 80–20 train-test split. 
•	Categorical features are encoded using label encoding. 
•	Numerical features are scaled where required. 
•	For the location dashboard, data is pre-aggregated at the city level for efficient visualization. 
