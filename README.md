# Lifestyle Health Risk Predictor

A machine learning-based web application that predicts an individual’s health risk level (Low / Medium / High) based on lifestyle factors such as sleep, stress, screen time, diet, and exercise.

# Project Overview

This project uses machine learning algorithms to analyse lifestyle patterns and classify users into different health risk categories. It also provides personalised suggestions based on user inputs.

The model is deployed using Streamlit, allowing real-time interaction through a simple web interface.

# Features
1. Predicts health risk level (Low, Medium, High)
2. Uses multiple ML models: Logistic Regression, K-Nearest Neighbours (KNN), Support Vector Machine (SVM)
3. Displays: Model comparison, Confusion matrices, Feature importance
4. Provides personalised health suggestions
5. Interactive Streamlit web app

# Dataset Features

The model is trained on lifestyle-related attributes:

1. Age
2. Sleep Duration
3. Stress Level
4. Daily Screen Time
5. Exercise Frequency
6. Diet Type
7. Gender

# Tech Stack
- Python
- Pandas, NumPy (data processing)
- Scikit-learn (machine learning)
- Matplotlib, Seaborn (visualization)
- Streamlit (web app)

# Machine Learning Workflow
1. Data Cleaning & Preprocessing
2. Handling Missing Values
3. Encoding Categorical Features
4. Feature Scaling
5. Train-Test Split (80–20)
6. Model Training (LR, KNN, SVM)
7. Hyperparameter Tuning (GridSearchCV)
8. Model Evaluation:
9. Accuracy
10. Confusion Matrix
11. Classification Report

# Results
- SVM achieved the best performance (~98% accuracy)
- KNN performed well after tuning
- Logistic Regression served as a baseline model

# Key Insights
- High stress and low sleep strongly increase health risk
- Increased screen time correlates with a higher risk
- Regular exercise significantly reduces the risk

# Streamlit App

The app allows users to:

- Input lifestyle details
- Get instant health risk prediction
- Receive personalised suggestions
## How to Run Locally
1. Clone the repository
git clone (https://github.com/tanni19p/lifestyle-health-risk-predictor)
2. cd lifestyle-health-risk-predictor

3. Install dependencies
pip install -r requirements.txt

4. Run the app
streamlit run app.py

# Project Structure
- ├── app.py
- ├── train.py
- ├── model.pkl
- ├── scaler.pkl
- ├── label_encoder.pkl
- ├── columns.pkl
- ├── data.csv
- ├── test_data_500.csv
- ├── *.png (visualizations)

# Future Improvements
- Add more features (BMI, habits, etc.)
- Improve model performance with advanced techniques
- Enhance UI/UX

# Acknowledgment

This project was built as part of a machine learning journey to understand real-world data analysis and model deployment.
