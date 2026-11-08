from pathlib import Path
import pandas as pd

# Project root folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset location
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# IBM dataset URL
DATA_URL = (
    "https://raw.githubusercontent.com/IBM/"
    "telco-customer-churn-on-icp4d/master/data/"
    "Telco-Customer-Churn.csv"
)

# Download and save dataset
df = pd.read_csv(DATA_URL)

file_path = DATA_DIR / "Telco-Customer-Churn.csv"
df.to_csv(file_path, index=False)

print("Dataset downloaded successfully!")
print("Dataset shape:", df.shape)
print("Saved at:", file_path)