import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load dataset
df = pd.read_csv("data/train.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns)

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nStatistical summary:")
print(df.describe())

print("\nLoan Status distribution:")
print(df["Loan_Status"].value_counts())


# Loan status visualization
sns.countplot(data=df, x="Loan_Status")
plt.title("Loan Approval Distribution")
plt.show()


# Credit history vs loan status
sns.countplot(
    data=df,
    x="Credit_History",
    hue="Loan_Status"
)

plt.title("Credit History vs Loan Status")
plt.show()