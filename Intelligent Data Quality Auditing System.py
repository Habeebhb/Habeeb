import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)
n = 1000  # Number of records

# Generate synthetic data
data = {
    "TransactionID": [f"TX{1000 + i}" for i in range(n)],
    "Date": pd.date_range(start="2023-01-01", periods=n).strftime("%Y-%m-%d"),
    "Amount": np.random.normal(500, 200, n).round(2),
    "IsFlagged": np.random.choice([True, False, np.nan], n, p=[0.1, 0.85, 0.05]),
    "AccountID": np.random.choice(["A123", "B456", "C789", "INVALID"], n)
}

# Convert dictionary to DataFrame
df = pd.DataFrame(data)

# Introduce missing values in Amount
missing_indices = np.random.choice(n, 50, replace=False)
df.loc[missing_indices, "Amount"] = np.nan  # Introduce NaNs

# Introduce outliers in Amount, ensuring no NaNs are modified
outlier_indices = np.random.choice(df.dropna(subset=["Amount"]).index, 20, replace=False)
df.loc[outlier_indices, "Amount"] *= 10  # Multiply outliers

# Generate Description correctly
df["Description"] = np.where(
    np.random.rand(n) > 0.1,
    ["TX-" + desc for desc in np.random.choice(["PAYMENT", "INVOICE", "REFUND", "UNKNOWN"], n)],
    np.nan  # Assign NaN for 10% of values
)

# Save to CSV
df.to_csv("financial_data.csv", index=False)

print("Data generation complete. CSV saved successfully.")
# Data Extraction and Pre Processing 
# Load data
# Import pandas
import pandas as pd

# Load data
df = pd.read_csv("financial_data.csv")

# Connect to SQLite database
from sqlalchemy import create_engine
engine = create_engine("sqlite:///financial.db")
df.to_sql("transactions", engine, if_exists="replace", index=False)

# SQL query to analyze missing data
query = """
SELECT 
    SUM(CASE WHEN Amount IS NULL THEN 1 ELSE 0 END) AS Missing_Amount,
    SUM(CASE WHEN Description IS NULL THEN 1 ELSE 0 END) AS Missing_Description,
    SUM(CASE WHEN IsFlagged IS NULL THEN 1 ELSE 0 END) AS Missing_IsFlagged
FROM transactions;
"""
missing_data = pd.read_sql(query, engine)
print(missing_data)
#Statistical Analysis & Anomaly Detection
from sklearn.ensemble import IsolationForest

# Detect outliers in Amount
clf = IsolationForest(contamination=0.05)
outliers = clf.fit_predict(df[["Amount"]].dropna())
df["IsOutlier"] = False
df.loc[df["Amount"].notna(), "IsOutlier"] = (outliers == -1)

# Validate Boolean field
invalid_boolean = ~df["IsFlagged"].isin([True, False])
print(f"Invalid Boolean entries: {invalid_boolean.sum()}")

# Check Description consistency - fixed to handle non-string values
# Convert to string type first and handle NaN values properly
df["Description"] = df["Description"].astype(str)
invalid_desc = ~df["Description"].str.startswith("TX-")
# Replace "nan" strings (from NaN conversions) back to False in results
invalid_desc = invalid_desc & (df["Description"] != "nan")
print(f"Invalid Descriptions: {invalid_desc.sum()}")
#Machine Learning for Missing Data
from sklearn.impute import KNNImputer

# Impute missing numerical values (Amount)
imputer = KNNImputer(n_neighbors=5)
# Fix: Extract the values from the returned array (which is 2D) into the new column
df["Amount_Imputed"] = imputer.fit_transform(df[["Amount"]]).flatten()
df.to_csv("audited_financial_data.csv", index=False)
# Read the file back into a new DataFrame
audited_df = pd.read_csv("audited_financial_data.csv")

# Display the first few rows
print(audited_df.head())

# For a more comprehensive overview
print(audited_df.info())
print(audited_df.describe())
import os
print("File saved at:", os.path.abspath("audited_financial_data.csv"))