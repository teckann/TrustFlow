import pickle
import pandas as pd

# Load model data and medians
with open('fraud_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    features = model_data['features']

with open('feature_medians.pkl', 'rb') as f:
    medians = pickle.load(f)

def check_scenario(name, amount, c8, v317, v287):
    input_data = {}
    for feat in features:
        if feat in medians:
            input_data[feat] = medians[feat]
        else:
            input_data[feat] = 0
            
    input_data['TransactionAmt'] = amount
    input_data['C8'] = c8
    input_data['V317'] = v317
    input_data['V287'] = v287
    
    df_input = pd.DataFrame([input_data])[features]
    prob = model.predict_proba(df_input)[0, 1]
    print(f"Scenario: {name}")
    print(f"  Amt: {amount}, C8: {c8}, V317: {v317}, V287: {v287}")
    print(f"  Risk Score: {prob:.4f}")
    print("-" * 20)

check_scenario("Normal with medians", 50.0, 1, 0, 0)
check_scenario("High Frequency with medians", 50.0, 150, 0, 0)
check_scenario("High Volume Session with medians", 50.0, 1, 1800, 0)
check_scenario("Fraud Example with medians", 150.0, 100, 1500, 5)
