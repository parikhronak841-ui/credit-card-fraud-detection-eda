# 🔍 Credit Card Fraud Detection – EDA & Anomaly Detection

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Domain](https://img.shields.io/badge/Domain-AML%20%2F%20Fraud-red)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Isolation%20Forest-F7931E?logo=scikit-learn)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?logo=plotly)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 📌 Project Overview

This project applies **exploratory data analysis and machine learning** to detect fraudulent credit card transactions. It mirrors real-world workflows used by **AML Analysts, Fraud Investigators, and Financial Crime Compliance** teams at banks and fintech firms.

Using a synthetic dataset of 10,000 transactions (97% legitimate, 3% fraud), the project identifies behavioral patterns that distinguish fraudulent from legitimate activity — and builds both an unsupervised anomaly detector and a supervised classification model.

---

## 🎯 Why This Matters

Financial fraud costs the global economy **$32+ billion annually**. Early detection through data-driven systems reduces losses, protects customers, and ensures regulatory compliance (FINTRAC, AML/KYC frameworks).

---

## 🗂️ Dataset

| Feature | Description |
|---|---|
| `transaction_id` | Unique transaction identifier |
| `amount` | Transaction amount ($) |
| `hour` | Hour of day (0–23) |
| `day_of_week` | Day transaction occurred |
| `merchant_category` | Grocery, ATM, Online, Travel, etc. |
| `country` | Transaction country |
| `distance_from_home_km` | Geographic distance from customer's home |
| `failed_attempts` | Failed PIN/authentication attempts |
| `is_new_merchant` | First-time merchant flag |
| `is_foreign` | Foreign transaction flag |
| `label` | **Target: 1 = Fraud, 0 = Legitimate** |

---

## 🔧 Tools & Libraries

| Tool | Purpose |
|---|---|
| `Pandas / NumPy` | Data wrangling |
| `Plotly` | Interactive visualizations |
| `Seaborn / Matplotlib` | Static charts |
| `Scikit-learn` | Isolation Forest + Logistic Regression |

---

## 📊 Analysis Modules

### 1. Fraud Pattern EDA
- Amount distribution: fraud skews toward higher transaction values
- Hour-of-day analysis: fraud spikes during late-night/early-morning hours
- Merchant category fraud rates: ATM and Online are highest risk
- Distance analysis: fraudulent transactions are made much farther from home

### 2. Red Flag Indicators
| Indicator | Elevated Fraud Risk |
|---|---|
| Transaction > $5,000 | ✅ High |
| Late night (12AM–4AM) | ✅ High |
| ATM or Online merchant | ✅ High |
| Foreign transaction | ✅ High |
| Distance > 500km from home | ✅ Very High |
| 2+ failed auth attempts | ✅ Very High |
| New merchant | ✅ Moderate |

### 3. Anomaly Detection – Isolation Forest
- Unsupervised model flags statistical outliers without needing labeled fraud data
- Useful when fraud labels are unavailable (common in real-world scenarios)

### 4. Supervised Baseline – Logistic Regression
- ROC-AUC: ~0.90
- Demonstrates ML can outperform rule-based systems

---

## 💡 AML/Fraud Prevention Recommendations

1. **Multi-Factor Risk Scoring** — Flag transactions with 2+ red flag indicators
2. **Foreign Transaction Step-Up Auth** — OTP/biometric for all cross-border transactions
3. **Night-Time Velocity Limits** — Stricter limits between 12AM–4AM
4. **ML Scoring Pipeline** — Deploy LR/RF model as a real-time secondary layer
5. **SAR Auto-Triggers** — Auto-generate Suspicious Activity Report alerts for anomalies
6. **FINTRAC Compliance** — Align flagging thresholds with Canadian AML reporting requirements

---

## 📁 Project Structure

```
project3_fraud_detection/
│
├── fraud_detection_eda.py          # Main analysis & model script
├── fraud_amount_distribution.html  # Interactive chart
├── fraud_by_hour.html              # Interactive chart
├── fraud_by_category.html          # Interactive chart
├── fraud_distance.html             # Interactive chart
├── fraud_by_country.html           # Interactive chart
└── README.md                       # This file
```

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/fraud-detection-eda.git
cd fraud-detection-eda

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn plotly scikit-learn

# 3. Run the analysis
python fraud_detection_eda.py
```

---

## 🏛️ Regulatory Context

This project demonstrates understanding of:
- **FINTRAC** (Financial Transactions and Reports Analysis Centre of Canada)
- **AML/KYC** – Anti-Money Laundering / Know Your Customer frameworks
- **SAR** – Suspicious Activity Report workflows
- **Velocity checks**, **geo-anomaly detection**, and **behavioural profiling**

> Highly relevant for roles in: AML Analyst, Fraud Investigator, KYC Analyst, Financial Crime Compliance, Risk Operations

---

## 👤 Author

**Ronak Parikh**
- 📧 parikhronak841@gmail.com
- 💼 [LinkedIn](https://www.linkedin.com/in/yourprofile)
- 🎓 Business Administration – Finance | Conestoga College
- 📜 IFIC Mutual Fund License | ACAMS (In Progress) | IBM Data Analyst Certificate (In Progress)

---

## 📝 License

MIT License
