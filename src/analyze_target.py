from pathlib import Path
import pandas as pd

# Find project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load dataset
DATA_PATH = PROJECT_ROOT / "data" / "Telco-Customer-Churn.csv"
df = pd.read_csv(DATA_PATH)

# Check target variable
target = "Churn"

print("Target column:", target)

print("\nClass counts:")
print(df[target].value_counts())

print("\nClass percentages:")
print(df[target].value_counts(normalize=True) * 100)