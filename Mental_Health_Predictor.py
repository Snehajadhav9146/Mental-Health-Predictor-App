import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import time  # For progress bar simulation

# Sample dataset
data = {
    'Age': [25, 45, 35, 50, 23, 34, 42, 25, 23, 54],
    'StressLevel': [3, 8, 5, 9, 2, 4, 7, 3, 2, 8],
    'SleepHours': [7, 5, 6, 4, 8, 7, 5, 8, 7, 5],
    'ExerciseHours': [2, 1, 3, 0, 4, 2, 1, 3, 2, 1],
    'WorkHours': [8, 10, 9, 12, 7, 8, 10, 9, 8, 11],
    'SocialSupport': [3, 2, 4, 1, 5, 3, 2, 4, 3, 2],
    'FinancialStress': [2, 4, 3, 5, 1, 3, 4, 2, 3, 4],
    'DietQuality': [4, 3, 5, 2, 5, 4, 3, 4, 4, 3],
    'AlcoholConsumption': [1, 3, 2, 4, 1, 2, 3, 1, 2, 3],
    'SmokingHabit': [0, 1, 0, 1, 0, 0, 1, 0, 0, 1],
    'MentalHealthIssue': [0, 1, 0, 1, 0, 0, 1, 0, 0, 1]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Split dataset
X = df.drop('MentalHealthIssue', axis=1)
y = df['MentalHealthIssue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Function for login page
def login():
    st.title('🔐 Login')
    with st.form(key="login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")

    if submit_button:
        if username == 'Sneha' and password == 'Pratya':
            st.session_state['loggedIn'] = True
            st.success("🎉 Login Successful!")
            st.rerun()  # ✅ Fixed: Use st.rerun() instead of st.experimental_rerun()
        else:
            st.error("❌ Invalid username or password")

# Function for main application
def main_page():
    st.title('🧠 Mental Health Predictor')

    # Sidebar for user input
    st.sidebar.header('📊 Input Features')
    age = st.sidebar.slider('Age', 0, 100, 25)
    stress_level = st.sidebar.slider('Stress Level (1-10)', 1, 10, 5)
    sleep_hours = st.sidebar.slider('Sleep Hours', 0, 12, 7)
    exercise_hours = st.sidebar.slider('Exercise Hours per Week', 0, 14, 3)
    work_hours = st.sidebar.slider('Work Hours per Week', 0, 80, 40)
    social_support = st.sidebar.slider('Social Support (1-5)', 1, 5, 3)
    financial_stress = st.sidebar.slider('Financial Stress (1-5)', 1, 5, 3)
    diet_quality = st.sidebar.slider('Diet Quality (1-5)', 1, 5, 4)
    alcohol_consumption = st.sidebar.slider('Alcohol Consumption (0-5)', 0, 5, 2)
    smoking_habit = st.sidebar.selectbox('Smoking Habit', [0, 1], format_func=lambda x: 'Non-Smoker' if x == 0 else 'Smoker')

    # Function to predict mental health
    def predict_mental_health():
        input_data = [[age, stress_level, sleep_hours, exercise_hours, work_hours, social_support, financial_stress, diet_quality, alcohol_consumption, smoking_habit]]
        prediction = clf.predict(input_data)
        confidence = clf.predict_proba(input_data)[0]

        return 'Mental Health Issue' if prediction[0] == 1 else 'No Mental Health Issue', confidence

    # User input details
    st.subheader('📌 User Input Features')
    user_inputs = {
        "Age": age, "Stress Level": stress_level, "Sleep Hours": sleep_hours,
        "Exercise Hours": exercise_hours, "Work Hours": work_hours,
        "Social Support": social_support, "Financial Stress": financial_stress,
        "Diet Quality": diet_quality, "Alcohol Consumption": alcohol_consumption,
        "Smoking Habit": "Non-Smoker" if smoking_habit == 0 else "Smoker"
    }
    st.json(user_inputs)

    # Predict and show result with a progress bar
    if st.button('🔍 Predict'):
        with st.spinner('🔄 Processing...'):
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.02)
                progress.progress(i + 1)
        
        result, confidence = predict_mental_health()
        st.success(f'**Prediction:** {result}')
        
        # Show confidence scores in a bar chart
        st.subheader("📊 Prediction Confidence")
        confidence_df = pd.DataFrame({"Mental Health Issue": [confidence[1]], "No Issue": [confidence[0]]})
        st.bar_chart(confidence_df)

    # Display dataset
    if st.checkbox("📂 Show Dataset"):
        st.subheader('🔍 Dataset')
        st.write(df)

# Session state for login
if 'loggedIn' not in st.session_state:
    st.session_state['loggedIn'] = False

# Run the app
if st.session_state['loggedIn']:
    main_page()
else:
    login()

st.sidebar.info('🚀 This app is maintained by **Sneha**')
