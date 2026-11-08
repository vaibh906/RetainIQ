from pathlib import Path
import pandas as pd

# Find the project root folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Set the dataset file path
DATA_PATH = PROJECT_ROOT / "data" / "Telco-Customer-Churn.csv"

# Load the dataset
df = pd.read_csv(DATA_PATH)

# Display basic information
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())