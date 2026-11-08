from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "cleaned_data.csv"

df = pd.read_csv(DATA_PATH)

# Find categorical columns
categorical_cols = df.select_dtypes(include=["str"]).columns

# Display unique categories
for col in categorical_cols:
    print(f"\n{col}:")
    print(df[col].unique())