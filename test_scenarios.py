import pickle
import pandas as pd

# Load model
with open('fraud_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    features = model_data['features']

def check_scenario(name, amount, c8, v317, v287):
    input_data = {feat: 0 for feat in features}
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

check_scenario("Normal", 50.0, 1, 0, 0)
check_scenario("High Frequency", 50.0, 150, 0, 0)
check_scenario("High Volume Session", 50.0, 1, 1800, 0)
check_scenario("Fraud Example", 150.0, 100, 1500, 5)
