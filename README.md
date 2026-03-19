#  TrustFlow: 30ms Real-Time Digital Integrity Engine 🛡️

## 🌐 Project Context
Digital payment adoption in ASEAN is skyrocketing, particularly through "Super Apps." However, millions of new users are "unbanked" or have low digital literacy. For a gig worker in the Philippines or a rural merchant in Thailand, losing their digital wallet balance to a fraudulent transaction is catastrophic and destroys trust in the digital economy.

This project, developed for **VHack 2026 - Case Study 2: Digital Trust – Real-Time Fraud Shield for the Unbanked**, provides an AI-powered solution to detect and prevent fraud in real-time, specifically designed for digital wallet ecosystems.

## 🚀 Problem Statement
Conventional rule-based fraud detection systems are often inadequate in identifying sophisticated or evolving fraud patterns. There is an urgent need for AI models that can analyze transaction behavior, detect anomalies in real-time, and prevent fraudulent activities without disrupting genuine transactions for legitimate users.

## 🛠️ Key Technical Features
- **Behavioral Profiling**: Analyzes user patterns (frequency, amount, location, time) to build a "normal" behavior baseline.
- **Real-Time Anomaly Scoring**: Optimized XGBoost model capable of scoring transactions in milliseconds.
- **Imbalanced Class Handling**: Implemented **SMOTE** (Synthetic Minority Over-sampling Technique) to handle datasets where fraudulent transactions are extremely rare.
- **Contextual Data Integration**: Incorporates non-transactional data (device fingerprints, email reputation) to enhance accuracy.
- **Privacy-First**: Designed for secure handling of sensitive financial data.

## 📊 Technical Performance
- **Low Latency**: Average prediction time is **~30ms**, ensuring no disruption to the checkout process.
- **High Precision**: Optimized to minimize "False Positives," preventing the blocking of legitimate users.
- **Model**: Trained on the IEEE-CIS Fraud Detection dataset using **XGBoost**.

## 📁 Project Structure
- `main.py`: The primary entry point. A Streamlit-based interactive dashboard to simulate and visualize real-time fraud detection.
- `train_model.py`: Script used to preprocess data and train the XGBoost model.
- `explore_data.py`: Initial data analysis and exploration script.
- `fraud_model.pkl`: The serialized trained XGBoost model.
- `label_encoders.pkl`: Label encoders for categorical features.
- `feature_medians.pkl`: Pre-calculated medians for feature imputation during real-time inference.

## ⚙️ Installation & Usage

### Prerequisites
- Python 3.11 or higher
- [Optional] Virtual environment (e.g., venv or conda)

### 1. Clone the Repository
```bash
git clone https://github.com/teckann/TrustFlow.git
cd "VHack 2026"
```

### 2. Install Dependencies
```bash
pip install streamlit pandas scikit-learn xgboost imbalanced-learn matplotlib seaborn requests
```

### 3. Run the Application
Start the interactive dashboard using Streamlit:
```bash
streamlit run main.py
```

## 🖥️ Using the Dashboard
1. Once running, open the URL (typically `http://localhost:8501`) in your browser.
2. Use the **Sidebar** to enter transaction details (Amount, Card Type, etc.).
3. Adjust **Advanced Features** like Frequency Count (C8) or Session Volume (V317) to see how the risk score changes.
4. Click **"Load Fraud Example"** to quickly see how the system identifies and blocks a high-risk transaction pattern.

---
**Developed for VHack 2026 - Case Study 2: Digital Trust – Real-Time Fraud Shield for the Unbanked**
