import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, precision_recall_curve, confusion_matrix
from imblearn.over_sampling import SMOTE
import pickle
import os

def preprocess_data(train_transaction_path, train_identity_path, nrows=200000):
    print("Loading data...")
    train_transaction = pd.read_csv(train_transaction_path, nrows=nrows)
    train_identity = pd.read_csv(train_identity_path)
    
    # Merge
    train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
    del train_transaction, train_identity
    
    print("Preprocessing data...")
    # 1. Handle missing values
    # Drop columns with > 50% missing values
    missing_info = train.isnull().sum() / len(train)
    cols_to_drop = missing_info[missing_info > 0.5].index
    train = train.drop(columns=cols_to_drop)
    
    # Fill remaining missing values
    # Numeric with median, categorical with mode
    for col in train.columns:
        if train[col].dtype == 'object':
            train[col] = train[col].fillna(train[col].mode()[0] if not train[col].mode().empty else 'unknown')
        else:
            train[col] = train[col].fillna(train[col].median())
            
    # 2. Encode categorical variables
    cat_cols = train.select_dtypes(include=['object']).columns
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str))
        le_dict[col] = le
        
    return train, le_dict

def train_fraud_model(train):
    print("Preparing features and target...")
    X = train.drop(['isFraud', 'TransactionID', 'TransactionDT'], axis=1)
    y = train['isFraud']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Original class distribution: {np.bincount(y_train)}")
    
    # 3. Handle Imbalance with SMOTE
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"Resampled class distribution: {np.bincount(y_train_res)}")
    
    # 4. Train XGBoost
    print("Training XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=9,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method='hist', # Faster
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(X_train_res, y_train_res)
    
    # 5. Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save model and features
    print("Saving model and feature list...")
    model_data = {
        'model': model,
        'features': X.columns.tolist()
    }
    with open('fraud_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    return model, X.columns.tolist()

if __name__ == "__main__":
    train_df, le_dict = preprocess_data('train_transaction.csv', 'train_identity.csv', nrows=200000)
    # Save label encoders for later use in API
    with open('label_encoders.pkl', 'wb') as f:
        pickle.dump(le_dict, f)
        
    model, features = train_fraud_model(train_df)
    print("Training complete!")
