import streamlit as st
import pandas as pd
import pickle

# Load files
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

st.set_page_config(page_title="Health Risk Predictor", layout="wide")

st.title("Lifestyle Health Risk Predictor")

st.markdown("""
This application predicts **lifestyle-based health risk** using machine learning.

Adjust your lifestyle factors in the sidebar and click **Predict Risk**.
""")

st.sidebar.header("Enter Your Details")

age = st.sidebar.slider("Age", 10, 80, 25)
sleep = st.sidebar.slider("Sleep Duration (hours)", 3.0, 10.0, 7.0)
stress = st.sidebar.slider("Stress Level (1-10)", 1, 10, 5)
screen = st.sidebar.slider("Daily Screen Time (hours)", 1.0, 15.0, 5.0)

exercise = st.sidebar.selectbox("Exercise Frequency", ["Low", "Medium", "High"])
diet = st.sidebar.selectbox("Diet Type", ["Vegetarian", "Non-Vegetarian"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

input_data = {
    'Age': age,
    'Sleep_Duration': sleep,
    'Stress_Level': stress,
    'Daily_Screen_Time': screen,
    'Exercise_Frequency': exercise,
    'Diet_Type': diet,
    'Gender': gender
}

input_df = pd.DataFrame([input_data])

input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=columns, fill_value=0)

input_scaled = scaler.transform(input_df)

st.subheader("Your Inputs")
st.table(pd.DataFrame([input_data]))

st.subheader("Prediction Result")

if st.button("Predict Risk"):
    pred = model.predict(input_scaled)
    result = le.inverse_transform(pred)[0]

    if result == "High":
        st.error("⚠️ High Risk")
    elif result == "Medium":
        st.warning("⚠️ Medium Risk")
    else:
        st.success("✅ Low Risk")

    st.markdown("### 💡 Suggestions")

    if stress > 7:
        st.write("- Reduce stress (meditation, breaks, exercise)")
    if sleep < 6:
        st.write("- Improve sleep duration (aim for 7-8 hours)")
    if screen > 8:
        st.write("- Reduce screen time")
    if exercise == "Low":
        st.write("- Increase physical activity")
    if diet == "Non-Vegetarian":
        st.write("- Maintain balanced diet with more nutrients")

with st.expander("About Model"):
    st.write("""
    - Models Used: Logistic Regression, KNN, SVM  
    - Best Model: SVM (~98% accuracy)  
    - Features: Age, Sleep Duration, Stress Level, Screen Time, Exercise, Diet  
    """)

st.markdown("---")
st.markdown("Built using Machine Learning + Streamlit")