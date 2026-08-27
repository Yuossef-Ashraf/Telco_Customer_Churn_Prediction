# Telco Customer Churn & Retention Analytics Platform - Architecture & Pipeline Design

```mermaid
graph TD
    DataInput[Raw CSV Dataset: WA_Fn-UseC_-Telco-Customer-Churn.csv] --> Preproc[Data Cleaning & Column Transformer]
    Preproc -->|Numeric| Scaler[StandardScaler Normalization]
    Preproc -->|Categorical| Encoder[One-Hot Categorical Encoding]
    Scaler --> Split[Train/Test Stratified Split 80/20]
    Encoder --> Split
    Split --> Train[Model Training: XGBoost Classifier]
    Train --> Eval[Evaluation & Benchmarks]
    Eval --> Inference[Production Inference & CLI]
```

## Comparative Models Evaluated
- **XGBoost Classifier**
- **Random Forest**
- **Logistic Regression**
- **LightGBM**
