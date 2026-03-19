import pandas as pd
import numpy as np

# Load a sample of training data
train_transaction = pd.read_csv('train_transaction.csv', nrows=50000)
train_identity = pd.read_csv('train_identity.csv')
train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')

# Find some fraud examples
fraud_examples = train[train['isFraud'] == 1].head(10)

# Get top features from our previous importance check
# V30, C8, V317, V287, V70, C4, V69, C14, V95, V94
top_cols = ['isFraud', 'TransactionAmt', 'card1', 'card2', 'card4', 'card6', 'P_emaildomain', 'C8', 'V30', 'V317', 'V287']

print("Fraud Examples Top Features:")
print(fraud_examples[top_cols])

# Compare with Non-Fraud
non_fraud_examples = train[train['isFraud'] == 0].head(10)
print("\nNon-Fraud Examples Top Features:")
print(non_fraud_examples[top_cols])

# Calculate means for fraud vs non-fraud
print("\nFeature Means (Fraud vs Non-Fraud):")
comparison = train.groupby('isFraud')[['TransactionAmt', 'C8', 'V30', 'V317', 'V287']].mean()
print(comparison)
