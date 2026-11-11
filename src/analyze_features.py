from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "cleaned_data.csv"

df = pd.read_csv(DATA_PATH)

# Separate numerical and categorical columns
numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = df.select_dtypes(include=["str"]).columns

print("Numerical Features:")
print(list(numerical_cols))

print("\nCategorical Features:")
print(list(categorical_cols))

# Check unique values
print("\nUnique values in categorical columns:")
for col in categorical_cols:
    print(f"{col}: {df[col].nunique()} unique values")


# Numerical Feature Analysis
print("\nNumerical Feature Statistics:")

print(
    df[numerical_cols].describe().round(2)
)


import matplotlib.pyplot as plt

# Numerical Feature Distribution
for col in ["tenure", "MonthlyCharges", "TotalCharges"]:

    plt.figure(figsize=(7, 4))

    plt.hist(df[col], bins=30, edgecolor="black")

    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Number of Customers")

    plt.tight_layout()
    plt.show()
# Boxplot Analysis
for row in ["tenure", "MonthlyCharges", "TotalCharges"]:

    plt.figure(figsize=(7, 4))

    plt.boxplot(df[row])

    plt.title(f"Boxplot of {row}")
    plt.ylabel(row)

    plt.tight_layout()
    plt.show()
    
# Churn vs Tenure Analysis

plt.figure(figsize=(8, 5))

for churn_status in ["Yes", "No"]:
    plt.hist(
        df[df["Churn"] == churn_status]["tenure"],
        bins=30,
        alpha=0.5,
        label=f"Churn: {churn_status}"
    )

plt.title("Customer Churn vs Tenure")
plt.xlabel("Tenure (Months)")
plt.ylabel("Number of Customers")
plt.legend()

plt.tight_layout()
plt.show()




# Churn vs Contract Analysis

contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn"]
)

print("\nChurn vs Contract:")
print(contract_churn)

contract_churn.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Customer Churn vs Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.legend(title="Churn")

plt.tight_layout()
plt.show()

