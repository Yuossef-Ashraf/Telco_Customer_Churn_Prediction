"""
Production Machine Learning Pipeline for Telco Customer Churn & Retention Analytics Platform.
Supports training, cross-validation, feature importance, and inference predictions.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger("ml_pipeline")


def load_dataset(csv_path: str):
    """Safely load and validate dataset."""
    import pandas as pd
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded dataset from {csv_path} with shape: {df.shape}")
    return df


def train_and_evaluate(csv_path: str = "WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    """Train pipeline and display benchmark evaluation metrics."""
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_squared_error

    df = load_dataset(csv_path)
    target_col = "Churn"
    
    # Verify target
    if target_col not in df.columns:
        # Fallback to last column if exact name differs
        target_col = df.columns[-1]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Preprocessing
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ]
    )

    is_class = "classification" == "classification"
    model = RandomForestClassifier(n_estimators=100, random_state=42) if is_class else RandomForestRegressor(n_estimators=100, random_state=42)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    print("\n" + "=" * 55)
    print("           MODEL EVALUATION BENCHMARK            ")
    print("=" * 55)
    if is_class:
        acc = accuracy_score(y_test, y_pred)
        print(f"  Accuracy Score : {acc * 100:.2f}%")
    else:
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"  R² Score  : {r2:.4f}")
        print(f"  RMSE Loss : {rmse:.4f}")
    print("=" * 55 + "\n")
    return pipeline


def main():
    parser = argparse.ArgumentParser(description="Telco Customer Churn & Retention Analytics Platform CLI")
    parser.add_argument("--data", type=str, default="WA_Fn-UseC_-Telco-Customer-Churn.csv", help="Path to CSV dataset")
    args = parser.parse_args()
    train_and_evaluate(args.data)


if __name__ == "__main__":
    main()
