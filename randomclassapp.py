import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- Page Configuration ---
# FIXED: Using 'page_icon' instead of 'icon'
st.set_page_config(page_title="Titanic Survival Predictor", layout="wide")

st.title("Titanic Survival Predictor")
st.write("This app predicts survival using the Random Forest logic from your notebook.")

# --- 1. Load and Preprocess Data ---
@st.cache_resource
def load_and_train():
    # Use a relative path (ensure titanic.csv is in the same folder as this script)
    csv_filename = 'titanic.csv'
    
    try:
        df = pd.read_csv(r"C:\Users\benan\Documents\Data_Scientist\Streamlit\titanic.csv")
    except FileNotFoundError:
        file_path = r'C:\Users\benan\Documents\Data_Scientist\Streamlit\titanic.csv'
        st.error(f"❌ '{file_path}' not found in the current directory.")
        st.stop()

    # Define columns based on your specific randomclass.ipynb headers
    # Note: These are lowercase to match your specific dataset error
    required_cols = ['p_class', 'sex', 'age', 'sib_sp', 'parch', 'fare', 'survived']
    
    try:
        df = df[required_cols].dropna()
    except KeyError as e:
        st.error(f"❌ Column mismatch: {e}")
        st.info(f"Your CSV has these columns: {df.columns.tolist()}")
        st.stop()

    # Label Encoding for Gender
    le = LabelEncoder()
    df['sex'] = le.fit_transform(df['sex']) # male usually becomes 1, female 0

    X = df.drop('survived', axis=1)
    y = df['survived']

    # Train Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, le, df

# Execute the training
model, le, dataframe = load_and_train()

# --- 2. Sidebar User Inputs ---
st.sidebar.header("📋 Passenger Details")

# Ensure these keys match the names used in X
p_class = st.sidebar.selectbox("Ticket Class", options=[1, 2, 3])
sex_str = st.sidebar.radio("Gender", options=['male', 'female'])
age = st.sidebar.slider("Age", 0, 80, 28)
sib_sp = st.sidebar.number_input("Siblings/Spouses Aboard", 0, 8, 0)
parch = st.sidebar.number_input("Parents/Children Aboard", 0, 6, 0)
fare = st.sidebar.slider("Fare Paid", 0.0, 500.0, 32.0)

# --- 3. Prediction ---
# Convert the radio button string to the numerical format the model expects
sex_encoded = le.transform([sex_str])[0]

# Create the input array in the exact same order as X
input_data = np.array([[p_class, sex_encoded, age, sib_sp, parch, fare]])

if st.button("Predict Survival Status"):
    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)

    st.divider()
    if prediction[0] == 1:
        st.success(f"### Result: **Survived** 🟢")
        st.write(f"Model Confidence: {prob[0][1]*100:.1f}%")
    else:
        st.error(f"### Result: **Did Not Survive** 🔴")
        st.write(f"Model Confidence: {prob[0][0]*100:.1f}%")

# --- 4. Visualizations ---
st.divider()
st.subheader("📊 Insights from the Dataset")

col1, col2 = st.columns(2)

with col1:
    # Feature Importance Plot
    
    importances = model.feature_importances_
    feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    
    fig, ax = plt.subplots()
    sns.barplot(x=importances, y=feature_names, palette="magma", ax=ax)
    ax.set_title("Key Factors in Survival Prediction")
    st.pyplot(fig)

with col2:
    # Data Distribution
    fig2, ax2 = plt.subplots()
    dataframe['survived'].value_counts().plot.pie(
        labels=['Deceased', 'Survived'], autopct='%1.1f%%', colors=['#ff9999','#66b3ff'], ax=ax2
    )
    ax2.set_ylabel('')
    st.pyplot(fig2)