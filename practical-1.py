"""Practical 1: Data Pre-processing and Data Quality Assessment
Objective: Load a dataset, inspect its structure, identify missing/duplicate/inconsistent data, and perform basic remediation.
Explanation: Real-world data is messy — it has missing values (NaN), duplicate rows, wrong data types, and outliers. Before any ML model can be trained, we must:
Inspect structure (shape, dtypes, info(), describe())
Detect missing values (isnull().sum())
Detect duplicates (duplicated())
Handle missing values (drop or impute with mean/median/mode)
Fix inconsistent data (e.g., mismatched casing, wrong types, outliers)
Re-verify the cleaned dataset"""

import pandas as pd
import numpy as np

# Create a sample messy dataset (replace with pd.read_csv('file.csv'))
data = {
    'Name': ['Alice', 'Bob', 'charlie', 'David', 'Alice', np.nan, 'Frank'],
    'Age': [25, np.nan, 30, 22, 25, 40, -5],          # -5 is inconsistent (invalid age)
    'Salary': [50000, 60000, np.nan, 45000, 50000, 55000, 62000],
    'Department': ['HR', 'IT', 'IT ', 'hr', 'HR', 'Finance', 'IT']  # inconsistent casing/spacing
}
df = pd.DataFrame(data)

print("---- Original Data ----")
print(df)

# 1. Inspect structure
print("\nShape:", df.shape)
print("\nInfo:")
print(df.info())
print("\nStatistical Summary:\n", df.describe(include='all'))

# 2. Identify missing values
print("\nMissing values per column:\n", df.isnull().sum())

# 3. Identify duplicate rows
print("\nDuplicate rows:\n", df[df.duplicated()])

# 4. Identify inconsistent data (invalid age, inconsistent text casing)
print("\nInvalid ages (negative):\n", df[df['Age'] < 0])

# ---- Data Remediation ----

# Drop exact duplicate rows
df_clean = df.drop_duplicates()

# Fix text inconsistency: strip spaces, standardize case
df_clean['Department'] = df_clean['Department'].str.strip().str.upper()
df_clean['Name'] = df_clean['Name'].str.strip().str.title()

# Handle missing Name -> fill with 'Unknown'
df_clean['Name'] = df_clean['Name'].fillna('Unknown')

# Handle missing/invalid Age -> replace negative with NaN, then impute with median
df_clean.loc[df_clean['Age'] < 0, 'Age'] = np.nan
df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].median())

# Handle missing Salary -> impute with mean
df_clean['Salary'] = df_clean['Salary'].fillna(df_clean['Salary'].mean())

print("\n---- Cleaned Data ----")
print(df_clean)

print("\nMissing values after cleaning:\n", df_clean.isnull().sum())
print("\nDuplicates after cleaning:", df_clean.duplicated().sum())