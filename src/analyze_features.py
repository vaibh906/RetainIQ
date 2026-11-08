from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "cleaned_data.csv"

df = pd.read_csv(DATA_PATH)

# Separate numerical and categorical columns
numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = df.select_dtypes(include=["object"]).columns

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
