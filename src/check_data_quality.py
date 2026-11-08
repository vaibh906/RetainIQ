from pathlib import Path
import pandas as pd

# Find project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load dataset
DATA_PATH = PROJECT_ROOT / "data" / "Telco-Customer-Churn.csv"
df = pd.read_csv(DATA_PATH)

# 1. Check data types and non-null values
print("----- DATA INFO -----")
df.info()

# 2. Check missing values
print("\n----- MISSING VALUES -----")
print(df.isnull().sum())

# 3. Check duplicate rows
print("\n----- DUPLICATE ROWS -----")
print("Duplicate count:", df.duplicated().sum())

# 4. Check statistical summary
print("\n----- NUMERICAL SUMMARY -----")
print(df.describe())