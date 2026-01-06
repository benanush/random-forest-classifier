import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Titanic Survival Predictor",
    layout="wide"
)

st.title("🚢 Titanic Survival Predictor")
st.write("This app predicts survival using a Random Forest model.")

# -----------------------------
# Load & Train Model
# -----------------------------
@st.cache_resource
def load_and_train():
    try:
        df = pd.read_csv("titanic.csv")  # ✅ RELATIVE PATH
    except FileNotFoundError:
        st.error("❌ titanic.csv not found. Upload it to the GitHub repo.")
        st.stop()

    required_cols = ['p_class', 'sex', 'age', 'sib_sp', 'parch', 'fare', 'survived']

    try:
        df = df[required_cols].dropna()
    except KeyError as e:
        st.error(f"❌ Column mismatch: {e}")
        st.info(f"CSV columns: {df.columns.tolist()}")
        st.stop()

    le = LabelEncoder()
    df['sex'] = le.fit_transform(df['sex'])

    X = df.drop('survived', axis=1)
    y = df['survived']

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X, y)

    return model, le, df

model, le, dataframe = load_and_train()

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("🧍 Passenger Details")

p_class = st.sidebar.selectbox("Ticket Class", [1, 2, 3])
sex_str = st.sidebar.radio("Gender", ['male', 'female'])
age = st.sidebar.slider("Age", 1, 80, 30)
sib_sp = st.sidebar.number_input("Siblings / Spouses", 0, 8, 0)
parch = st.sidebar.number_input("Parents / Children"_
