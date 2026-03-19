import requests
import json
import time

def test_risk_api(transaction_id, amount, card_type, email_domain, extra_features={}):
    url = "http://localhost:8000/v1/risk-score"
    payload = {
        "TransactionID": transaction_id,
        "TransactionAmt": amount,
        "card4": card_type,
        "P_emaildomain": email_domain,
        **extra_features
    }
    
    start_time = time.time()
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        print(f"Transaction ID: {transaction_id}")
        print(f"Amount: {amount}")
        print(f"Risk Score: {data['risk_score']:.4f}")
        print(f"Status: {data['status']}")
        print(f"Action: {data['action']}")
        print(f"Latency: {data['latency_ms']} ms")
        print("-" * 30)
    except Exception as e:
        print(f"Error testing API: {e}")

if __name__ == "__main__":
    print("Testing Fraud Shield API with various scenarios...\n")
    
    # Scenario 1: Normal everyday transaction
    print("Scenario 1: Normal Transaction (Small amount, known card)")
    test_risk_api(1001, 25.50, "visa", "gmail.com")
    
    # Scenario 2: High value transaction
    print("Scenario 2: High Value Transaction (Large amount)")
    test_risk_api(1002, 5000.0, "mastercard", "yahoo.com")
    
    # Scenario 3: Suspicious counts (C8 is high)
    # C8 is one of the top features by importance
    print("Scenario 3: Suspicious Patterns (High card counts/frequency)")
    test_risk_api(1003, 100.0, "american express", "unknown", extra_features={"C8": 50, "V30": 5})
    
    # Scenario 4: Cross-border/Unknown domain
    print("Scenario 4: Contextual Anomaly (Unknown email domain)")
    test_risk_api(1004, 75.0, "discover", "anonymous.com")
