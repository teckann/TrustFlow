import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load a small sample for exploration
train_transaction = pd.read_csv('train_transaction.csv', nrows=100000)
train_identity = pd.read_csv('train_identity.csv')

print(f"Transaction shape: {train_transaction.shape}")
print(f"Identity shape: {train_identity.shape}")

# Merge data
train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
print(f"Merged shape: {train.shape}")

# Check for class imbalance
fraud_counts = train['isFraud'].value_counts(normalize=True)
print("\nClass Imbalance:")
print(fraud_counts)

# Check for missing values (top 20 columns)
missing_vals = train.isnull().sum() / len(train)
print("\nTop 20 columns with missing values:")
print(missing_vals.sort_values(ascending=False).head(20))

# Basic statistics for transaction amount
print("\nTransaction Amount Statistics:")
print(train['TransactionAmt'].describe())

# Save basic info to a file for reference
with open('data_summary.txt', 'w') as f:
    f.write(f"Transaction shape: {train_transaction.shape}\n")
    f.write(f"Identity shape: {train_identity.shape}\n")
    f.write(f"Merged shape: {train.shape}\n\n")
    f.write("Class Imbalance:\n")
    f.write(str(fraud_counts) + "\n\n")
    f.write("Top 20 missing values:\n")
    f.write(str(missing_vals.sort_values(ascending=False).head(20)) + "\n")
