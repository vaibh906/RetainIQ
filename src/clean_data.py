from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "Telco-Customer-Churn.csv"

df = pd.read_csv(DATA_PATH)

# Convert TotalCharges into numeric format
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Fill missing values with 0
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Check missing values
print("Missing values after cleaning:")
print(df["TotalCharges"].isnull().sum())

# Check data type
print("\nData type:")
print(df["TotalCharges"].dtype)

# Save cleaned dataset
OUTPUT_PATH = PROJECT_ROOT / "data" / "cleaned_data.csv"
df.to_csv(OUTPUT_PATH, index=False)

print("\nCleaned dataset saved successfully!")