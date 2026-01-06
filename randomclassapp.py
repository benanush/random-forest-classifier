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
        df = pd.read_csv("")
        df.columns = df.columns.str.lower()
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
parch = st.sidebar.number_input("Parents / Children", 0, 6, 0)
fare = st.sidebar.slider("Fare Paid", 0.0, 500.0, 30.0)

sex_encoded = le.transform([sex_str])[0]

input_data = np.array([[p_class, sex_encoded, age, sib_sp, parch, fare]])

# -----------------------------
# Prediction
# -----------------------------
if st.sidebar.button("Predict Survival"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    st.divider()
    if prediction[0] == 1:
        st.success("### ✅ Survived")
        st.write(f"Confidence: **{probability[0][1]*100:.2f}%**")
    else:
        st.error("### ❌ Did Not Survive")
        st.write(f"Confidence: **{probability[0][0]*100:.2f}%**")

# -----------------------------
# Visualizations
# -----------------------------
st.divider()
st.subheader("📊 Dataset Insights")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.barplot(
        x=model.feature_importances_,
        y=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'],
        ax=ax
    )
    ax.set_title("Feature Importance")
    st.pyplot(fig)

with col2:
    fig2, ax2 = plt.subplots()
    dataframe['survived'].value_counts().plot.pie(
        autopct='%1.1f%%',
        labels=['Did Not Survive', 'Survived'],
        ax=ax2
    )
    ax2.set_ylabel("")
    ax2.set_title("Survival Distribution")
    st.pyplot(fig2)





