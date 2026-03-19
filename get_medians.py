import pandas as pd
import numpy as np
import pickle

# Load training data to get medians
train_transaction = pd.read_csv('train_transaction.csv', nrows=100000)
train_identity = pd.read_csv('train_identity.csv')
train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')

# Get numeric medians
numeric_cols = train.select_dtypes(include=[np.number]).columns
medians = train[numeric_cols].median().to_dict()

# Save medians for the dashboard
with open('feature_medians.pkl', 'wb') as f:
    pickle.dump(medians, f)

print("Medians saved for all numeric features.")
