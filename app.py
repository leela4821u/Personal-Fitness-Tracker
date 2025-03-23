import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

# App Title and Intro
st.title("Personal Fitness Tracker")
st.write("Track your fitness journey and predict your calories burned accurately by entering your vital stats.")

# Sidebar Header
st.sidebar.header("User Input Parameters")

def user_input_features():
    age = st.sidebar.slider("Age: ", 10, 100, 30)
    height = st.sidebar.slider("Height (cm): ", 100, 220, 170)
    weight = st.sidebar.slider("Weight (kg): ", 30, 150, 70)
    bmi = round(weight / ((height / 100) ** 2), 2)
    duration = st.sidebar.slider("Duration (min): ", 0, 35, 15)
    heart_rate = st.sidebar.slider("Heart Rate: ", 60, 130, 80)
    body_temp = st.sidebar.slider("Body Temperature (C): ", 36, 42, 38)
    gender_button = st.sidebar.radio("Gender: ", ("Male", "Female"))

    gender = 1 if gender_button == "Male" else 0

    data_model = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": gender
    }
    features = pd.DataFrame(data_model, index=[0])
    return features

df = user_input_features()

st.write("---")
st.header("Your Input Parameters")
st.write(df)

# Load and preprocess data
@st.cache_data

def load_data():
    calories = pd.read_csv("calories.csv")
    exercise = pd.read_csv("exercise.csv")
    exercise_df = exercise.merge(calories, on="User_ID")
    exercise_df.drop(columns="User_ID", inplace=True)

    exercise_df.dropna(inplace=True)
    exercise_df.drop_duplicates(inplace=True)
    
    for data in [exercise_df]:
        data["BMI"] = round(data["Weight"] / ((data["Height"] / 100) ** 2), 2)
    
    return exercise_df

exercise_df = load_data()

exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)
exercise_train_data = pd.get_dummies(exercise_train_data, drop_first=True)
exercise_test_data = pd.get_dummies(exercise_test_data, drop_first=True)

X_train = exercise_train_data.drop("Calories", axis=1)
y_train = exercise_train_data["Calories"]

X_test = exercise_test_data.drop("Calories", axis=1)
y_test = exercise_test_data["Calories"]

# Train the model
GB_reg = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, max_depth=3, random_state=1)
GB_reg.fit(X_train, y_train)

# Align input columns
df = df.reindex(columns=X_train.columns, fill_value=0)

# Prediction
prediction = GB_reg.predict(df)

st.write("---")
st.header("Predicted Calories Burned")
st.success(f"{round(prediction[0], 2)} kilocalories")

st.write("---")
st.header("Similar Results")
calorie_range = [prediction[0] - 10, prediction[0] + 10]
similar_data = exercise_df[(exercise_df["Calories"] >= calorie_range[0]) & (exercise_df["Calories"] <= calorie_range[1])]
st.write(similar_data.sample(5) if not similar_data.empty else "No similar results found.")

st.write("---")
st.header("Custom Visual Insights")
parameter_choice = st.selectbox("Select a parameter to visualize:", ["Age", "Duration", "Heart_Rate", "Body_Temp", "BMI"])

fig, ax = plt.subplots(figsize=(8, 6))
sn.histplot(exercise_df[parameter_choice], bins=30, kde=True, color="lightcoral", ax=ax)
ax.set_title(f"Distribution of {parameter_choice}")
ax.set_xlabel(parameter_choice)
ax.set_ylabel("Frequency")
st.pyplot(fig)

st.write("---")
st.header("Model Performance Metrics")
y_pred = GB_reg.predict(X_test)
st.write(f"Mean Absolute Error: {round(metrics.mean_absolute_error(y_test, y_pred), 2)}")
st.write(f"Mean Squared Error: {round(metrics.mean_squared_error(y_test, y_pred), 2)}")
st.write(f"R-squared Score: {round(metrics.r2_score(y_test, y_pred) * 100, 2)}%")
