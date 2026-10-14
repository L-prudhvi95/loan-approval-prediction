import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv("data/train.csv")


# --------------------------------------------------
# 2. Remove unnecessary column
# --------------------------------------------------

if "Loan_ID" in df.columns:
    df = df.drop("Loan_ID", axis=1)


# --------------------------------------------------
# 3. Convert target variable
# --------------------------------------------------

df["Loan_Status"] = df["Loan_Status"].map({
    "Y": 1,
    "N": 0
})


# --------------------------------------------------
# 4. Separate features and target
# --------------------------------------------------

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]


# --------------------------------------------------
# 5. Define columns
# --------------------------------------------------

categorical_columns = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area"
]

numerical_columns = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History"
]


# --------------------------------------------------
# 6. Numerical preprocessing
# --------------------------------------------------

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


# --------------------------------------------------
# 7. Categorical preprocessing
# --------------------------------------------------

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)


# --------------------------------------------------
# 8. Combine preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numerical_pipeline,
            numerical_columns
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_columns
        )
    ]
)


# --------------------------------------------------
# 9. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 10. Logistic Regression
# --------------------------------------------------

logistic_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)


logistic_model.fit(X_train, y_train)

logistic_predictions = logistic_model.predict(X_test)

logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)


print("Logistic Regression Accuracy:",
      logistic_accuracy)


# --------------------------------------------------
# 11. Random Forest
# --------------------------------------------------

random_forest_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)


random_forest_model.fit(X_train, y_train)

rf_predictions = random_forest_model.predict(X_test)


rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)


print("\nRandom Forest Accuracy:",
      rf_accuracy)


# --------------------------------------------------
# 12. Detailed evaluation
# --------------------------------------------------

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        rf_predictions
    )
)


print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        rf_predictions
    )
)


# --------------------------------------------------
# 13. Save the complete model
# --------------------------------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(
    random_forest_model,
    "models/loan_model.pkl"
)


print("\nModel saved successfully!")
print("Location: models/loan_model.pkl")