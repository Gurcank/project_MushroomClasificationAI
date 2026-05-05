from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


@dataclass
class TrainingResult:
    best_model_name: str
    best_pipeline: Pipeline
    leaderboard: pd.DataFrame
    best_metrics: Dict[str, float]


def get_candidate_models(random_state: int = 42) -> Dict[str, Any]:
    models: Dict[str, Any] = {
        "logistic_regression": LogisticRegression(max_iter=2000),
        "decision_tree": DecisionTreeClassifier(random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
    }

    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
        )
    except Exception:
        # Keep training flow robust if xgboost is unavailable.
        pass

    return models


def build_pipeline(model: Any, categorical_features: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            )
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )


def evaluate_pipeline(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    y_pred = pipeline.predict(X_test)

    if hasattr(pipeline, "predict_proba"):
        y_score = pipeline.predict_proba(X_test)[:, 1]
    else:
        y_score = y_pred

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "false_negative": float(fn),
        "false_positive": float(fp),
        "true_positive": float(tp),
        "true_negative": float(tn),
    }


def train_and_select_best(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42,
) -> TrainingResult:
    models = get_candidate_models(random_state=random_state)
    results = []

    for model_name, model in models.items():
        pipeline = build_pipeline(model, categorical_features=list(X_train.columns))
        pipeline.fit(X_train, y_train)
        metrics = evaluate_pipeline(pipeline, X_test, y_test)
        results.append((model_name, pipeline, metrics))

    leaderboard = pd.DataFrame(
        [
            {"model": model_name, **metrics}
            for model_name, _pipeline, metrics in results
        ]
    )

    # Prioritize FN minimization, then higher recall, then better F1.
    leaderboard = leaderboard.sort_values(
        by=["false_negative", "recall", "f1"],
        ascending=[True, False, False],
        ignore_index=True,
    )

    best_name = str(leaderboard.iloc[0]["model"])
    best_pipeline = next(p for n, p, _m in results if n == best_name)
    best_metrics = {
        k: float(v)
        for k, v in leaderboard.iloc[0].to_dict().items()
        if k != "model"
    }

    return TrainingResult(
        best_model_name=best_name,
        best_pipeline=best_pipeline,
        leaderboard=leaderboard,
        best_metrics=best_metrics,
    )


def save_artifact(
    output_path: str | Path,
    pipeline: Pipeline,
    model_name: str,
    metrics: Dict[str, float],
    feature_options: Dict[str, list[str]],
) -> None:
    payload = {
        "model_name": model_name,
        "pipeline": pipeline,
        "metrics": metrics,
        "feature_options": feature_options,
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)


def load_artifact(path: str | Path) -> Dict[str, Any]:
    return joblib.load(path)
