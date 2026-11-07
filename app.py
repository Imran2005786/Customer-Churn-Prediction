from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Sample dataset
data = pd.DataFrame({
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

# Prepare features and target
X = data[["Tenure", "MonthlyCharges", "SupportCalls"]]
y = data["Churn"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Train Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Calculate accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Feature importance
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    probability = None

    tenure = 12
    charges = 70.0
    calls = 2

    if request.method == "POST":

        tenure = int(request.form.get("tenure", 12))
        charges = float(request.form.get("charges", 70))
        calls = int(request.form.get("calls", 2))

        customer = pd.DataFrame({
            "Tenure": [tenure],
            "MonthlyCharges": [charges],
            "SupportCalls": [calls]
        })

        prediction = int(model.predict(customer)[0])

        probability = model.predict_proba(customer)[0][1]

    churn_counts = data["Churn"].value_counts()

    stayed = int(churn_counts.get(0, 0))
    left = int(churn_counts.get(1, 0))

    return render_template(
        "index.html",
        accuracy=f"{accuracy:.1%}",
        prediction=prediction,
        probability=probability,
        tenure=tenure,
        charges=charges,
        calls=calls,
        stayed=stayed,
        left=left,
        importance=importance.to_dict()
    )


if __name__ == "__main__":
    app.run(debug=True)