import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
with open('fraud_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    features = model_data['features']

# Get feature importance
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'feature': features, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

print("Top 20 Features by Importance:")
print(feature_importance_df.head(20))

# Save to file
feature_importance_df.to_csv('feature_importance.csv', index=False)
