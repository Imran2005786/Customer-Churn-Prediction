
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Application title
st.title("📊 Customer Churn Prediction")
st.write("Predict whether a customer may leave a service using Machine Learning.")

# Sample dataset
@st.cache_data
def load_data():
    return pd.DataFrame({
        "Tenure": [
            2, 5, 12, 24, 36, 48, 60, 3, 18, 42,
            8, 30, 55, 6, 15, 50, 10, 40, 20, 58,
            1, 25, 33, 45, 14, 52, 7, 28, 38, 57
        ],
        "MonthlyCharges": [
            90, 85, 70, 60, 50, 45, 40, 95, 75, 48,
            88, 55, 42, 92, 78, 43, 80, 47, 65, 41,
            98, 58, 52, 46, 82, 44, 91, 57, 49, 39
        ],
        "SupportCalls": [
            5, 4, 3, 2, 1, 0, 0, 6, 3, 1,
            4, 2, 0, 5, 3, 1, 4, 0, 2, 0,
            6, 2, 1, 0, 4, 1, 5, 2, 1, 0
        ],
        "Churn": [
            1, 1, 0, 0, 0, 0, 0, 1, 0, 0,
            1, 0, 0, 1, 1, 0, 1, 0, 0, 0,
            1, 0, 0, 0, 1, 0, 1, 0, 0, 0
        ]
    })


data = load_data()

# Display dataset
st.subheader("📋 Sample Customer Data")
st.dataframe(data, use_container_width=True)

# Prepare features and target
X = data[["Tenure", "MonthlyCharges", "SupportCalls"]]
y = data["Churn"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Train the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

st.subheader("🤖 Machine Learning Model")
st.metric("Test Accuracy", f"{accuracy:.1%}")

# Customer input section
st.subheader("🔍 Predict Customer Churn")

col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.slider(
        "Customer Tenure (months)",
        min_value=1,
        max_value=72,
        value=12
    )

with col2:
    charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=500.0,
        value=70.0
    )

with col3:
    calls = st.slider(
        "Number of Support Calls",
        min_value=0,
        max_value=10,
        value=2
    )

# Make prediction
if st.button("Predict Churn", type="primary"):
    customer = pd.DataFrame({
        "Tenure": [tenure],
        "MonthlyCharges": [charges],
        "SupportCalls": [calls]
    })

    prediction = model.predict(customer)[0]
    probability = model.predict_proba(customer)[0][1]

    st.metric(
        "Estimated Churn Probability",
        f"{probability:.1%}"
    )

    if prediction == 1:
        st.error(
            "⚠️ The model predicts that this customer may leave."
        )
    else:
        st.success(
            "✅ The model predicts that this customer may stay."
        )

# Churn distribution chart
st.subheader("📈 Churn Distribution")

churn_counts = (
    data["Churn"]
    .value_counts()
    .reindex([0, 1], fill_value=0)
)

churn_counts.index = ["Stayed", "Left"]

st.bar_chart(churn_counts)

# Feature importance chart
st.subheader("📊 Feature Importance")

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

st.bar_chart(importance)

# Footer
st.info(
    "Educational demonstration using a small synthetic dataset. "
    "Predictions and accuracy are illustrative, not reliable "
    "estimates of real-world customer churn."
)

