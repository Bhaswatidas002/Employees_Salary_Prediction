import warnings
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")


def load_and_prepare() -> pd.DataFrame:
    print("Loading employee data...")
    df = pd.read_excel("Employees.xlsx")

    relevant_columns = [
        "Gender",
        "Years",
        "Department",
        "Monthly Salary",
        "Job Rate",
    ]
    df = df[relevant_columns].copy()

    # Basic cleaning
    df = df.dropna()
    df["Years"] = pd.to_numeric(df["Years"], errors="coerce")
    df = df[(df["Monthly Salary"].between(1000, 50000)) & (df["Years"].between(0, 50))]

    # Simplify departments (aligned with app)
    department_mapping = {
        "IT": ["IT", "Information Technology", "Software", "Development"],
        "Finance": ["Finance", "Accounting", "Financial"],
        "HR": ["HR", "Human Resources", "Personnel"],
        "Sales": ["Sales", "Marketing", "Business Development"],
        "Operations": ["Operations", "Production", "Manufacturing", "Logistics"],
    }

    def map_department(dept: str) -> str:
        text = dept.lower()
        for key, keywords in department_mapping.items():
            if any(k.lower() in text for k in keywords):
                return key
        return "Other"

    df["Department"] = df["Department"].apply(map_department)
    df = df[df["Department"].isin(["IT", "HR", "Sales", "Operations"])]

    df = df.reset_index(drop=True)
    print(f"Final dataset shape: {df.shape}")
    return df


def build_models() -> Dict[str, Pipeline]:
    # Feature columns and target
    numeric_features: List[str] = ["Years", "Job Rate"]
    categorical_features: List[str] = ["Gender", "Department"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    models: Dict[str, Pipeline] = {
        "LinearRegression": Pipeline([
            ("prep", preprocessor),
            ("model", LinearRegression()),
        ]),
        "Ridge": Pipeline([
            ("prep", preprocessor),
            ("model", Ridge(alpha=1.0, random_state=None)),
        ]),
        "Lasso": Pipeline([
            ("prep", preprocessor),
            ("model", Lasso(alpha=0.001, max_iter=5000, random_state=None)),
        ]),
        "KNN": Pipeline([
            ("prep", preprocessor),
            ("model", KNeighborsRegressor(n_neighbors=5)),
        ]),
        "DecisionTree": Pipeline([
            ("prep", preprocessor),
            ("model", DecisionTreeRegressor(random_state=42)),
        ]),
        "RandomForest": Pipeline([
            ("prep", preprocessor),
            ("model", RandomForestRegressor(n_estimators=200, random_state=42)),
        ]),
        "GradientBoosting": Pipeline([
            ("prep", preprocessor),
            ("model", GradientBoostingRegressor(random_state=42)),
        ]),
    }
    return models


def evaluate():
    df = load_and_prepare()
    X = df[["Gender", "Years", "Department", "Job Rate"]]
    y = df["Monthly Salary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = build_models()

    rows = []
    preds_table = pd.DataFrame({"Actual": y_test.reset_index(drop=True)})

    for name, pipe in models.items():
        print(f"Training {name}...")
        pipe.fit(X_train, y_train)

        # Holdout evaluation
        y_pred = pipe.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Cross-validated R2
        cv_r2 = cross_val_score(pipe, X, y, cv=5, scoring="r2").mean()

        rows.append({
            "model": name,
            "r2": r2,
            "cv_r2": cv_r2,
            "mse": mse,
            "mae": mae,
        })

        # Save a few predictions per model
        preds_table[name] = y_pred

    results = pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)
    results.to_csv("model_benchmark_results.csv", index=False)

    # Keep first 25 rows for a compact sample
    preds_table.head(25).to_csv("model_predictions_sample.csv", index=False)

    print("\n=== Benchmark Results (sorted by R2) ===")
    print(results.round(4))
    print("\nSaved:")
    print("- model_benchmark_results.csv")
    print("- model_predictions_sample.csv (first 25 rows)")


if __name__ == "__main__":
    evaluate()
