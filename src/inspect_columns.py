from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "Telco-Customer-Churn.csv"

df = pd.read_csv(DATA_PATH)

print("Data Types:")
print(df.dtypes)

print("\nTotalCharges blank values:")
print((df["TotalCharges"].str.strip() == "").sum())


# Find rows where TotalCharges is blank
blank_rows = df[df["TotalCharges"].str.strip() == ""]

print("Blank TotalCharges records:")
print(blank_rows[["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]])