import pandas as pd
import numpy as np
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time

app = FastAPI(title="Real-Time Fraud Shield API")

# Load model, features, and label encoders
with open('fraud_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    features = model_data['features']

with open('label_encoders.pkl', 'rb') as f:
    le_dict = pickle.load(f)

# Mock baseline data for missing features (e.g., median values)
# In a real system, these would be retrieved from a database/cache
baseline_data = {} # To be filled with some default values if needed

class TransactionRequest(BaseModel):
    TransactionID: int
    TransactionAmt: float
    ProductCD: str = 'W'
    card1: int = 1000
    card2: float = 300.0
    card3: float = 150.0
    card4: str = 'visa'
    card5: float = 226.0
    card6: str = 'debit'
    addr1: float = 300.0
    addr2: float = 87.0
    P_emaildomain: str = 'gmail.com'
    # Adding some common identity features
    id_31: str = 'chrome'
    DeviceType: str = 'desktop'
    DeviceInfo: str = 'Windows'

@app.get("/")
def read_root():
    return {"message": "Fraud Detection API is running"}

@app.post("/v1/risk-score")
async def get_risk_score(req: TransactionRequest):
    start_time = time.time()
    
    # Create a full feature vector with defaults
    input_data = {feat: 0 for feat in features}
    
    # Update with request values
    req_dict = req.dict()
    for k, v in req_dict.items():
        if k in input_data:
            input_data[k] = v
            
    # Preprocess categorical features using saved label encoders
    for col, le in le_dict.items():
        if col in input_data:
            val = str(input_data[col])
            # Handle unknown labels
            if val in le.classes_:
                input_data[col] = le.transform([val])[0]
            else:
                # Default to the first class or 'unknown' if it exists
                if 'unknown' in le.classes_:
                    input_data[col] = le.transform(['unknown'])[0]
                else:
                    input_data[col] = 0
    
    # Convert to DataFrame for model
    df_input = pd.DataFrame([input_data])[features]
    
    # Predict
    prob = model.predict_proba(df_input)[0, 1]
    
    # Decision Logic
    # 0 - 0.3: Approve
    # 0.3 - 0.7: Flag (Review)
    # 0.7 - 1.0: Block
    
    if prob < 0.3:
        status = "Approve"
        action = "None"
    elif prob < 0.7:
        status = "Flag"
        action = "Manual Review Required"
    else:
        status = "Block"
        action = "Transaction Denied"
        
    latency_ms = (time.time() - start_time) * 1000
    
    return {
        "transaction_id": req.TransactionID,
        "risk_score": float(prob),
        "status": status,
        "action": action,
        "latency_ms": round(latency_ms, 2)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
