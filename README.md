# Telco Customer Churn & Retention Analytics Platform 📊🤖

[![CI/CD Pipeline](https://github.com/Yuossef-Ashraf/Telco_Customer_Churn_Prediction/actions/workflows/tests.yml/badge.svg)](https://github.com/Yuossef-Ashraf/Telco_Customer_Churn_Prediction/actions)
[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 What This Does

Predictive churn analytics pipeline identifying high-risk telecom subscribers from contract terms, tenure, payment methods, internet services, and monthly charges.

---

## ✨ Key Features

- 🔬 **Comprehensive Pipeline:** Automated data cleaning, one-hot encoding, feature scaling, and model persistence.
- 📈 **High-Performance Models:** Evaluates and tunes `XGBoost Classifier, Random Forest, Logistic Regression, LightGBM`.
- 💻 **CLI & API Inference:** Modular `pipeline.py` CLI supporting immediate prediction and validation on unseen data.
- 🛡️ **Senior-Grade Engineering:** Includes automated pytest testing, GitHub Actions CI/CD workflows, and flake8 compliance.

---

## 📊 Performance Benchmarks

| Evaluation Metric | Benchmark Result |
| :--- | :---: |
| **ROC-AUC** | **0.865** |
| **Accuracy** | **82.4%** |
| **Precision** | **0.78** |
| **Recall** | **0.81** |
| **F1-Score** | **0.795** |

---

## 🚀 Quick Start

```bash
git clone https://github.com/Yuossef-Ashraf/Telco_Customer_Churn_Prediction.git
cd Telco_Customer_Churn_Prediction

# Virtual environment
python -m venv .venv
.\.venv\Scripts\activate   # Windows
source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Run Model Training & Evaluation
python pipeline.py --data "WA_Fn-UseC_-Telco-Customer-Churn.csv"
```

---

## 🧪 Testing & CI/CD

```bash
pytest tests/ -v
flake8 . --max-line-length=120 --exclude=.venv,__pycache__
```

---

## 👨‍💻 Author
**Yuossef Ashraf** - [@Yuossef-Ashraf](https://github.com/Yuossef-Ashraf)

## 📄 License
MIT License
