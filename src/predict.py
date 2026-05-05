from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from .model import load_artifact


class MushroomPredictor:
    def __init__(self, artifact_path: str | Path):
        artifact = load_artifact(artifact_path)
        self.pipeline = artifact["pipeline"]
        self.model_name = artifact["model_name"]
        self.metrics = artifact["metrics"]
        self.feature_options = artifact["feature_options"]

    def predict_one(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        X = pd.DataFrame([sample])
        pred = int(self.pipeline.predict(X)[0])

        if hasattr(self.pipeline, "predict_proba"):
            proba = float(self.pipeline.predict_proba(X)[0][1])
        else:
            proba = float(pred)

        return {
            "prediction": pred,
            "label": "Zehirli" if pred == 1 else "Yenilebilir",
            "poisonous_probability": proba,
        }

    def predict_batch(self, samples: pd.DataFrame) -> pd.DataFrame:
        preds = self.pipeline.predict(samples)

        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba(samples)[:, 1]
        else:
            probs = preds

        out = samples.copy()
        out["prediction"] = preds
        out["poisonous_probability"] = probs
        return out

    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        classifier = self.pipeline.named_steps["classifier"]
        preprocessor = self.pipeline.named_steps["preprocessor"]

        if hasattr(classifier, "feature_importances_"):
            importances = np.asarray(classifier.feature_importances_, dtype=float)
        elif hasattr(classifier, "coef_"):
            coef = np.asarray(classifier.coef_)
            if coef.ndim == 2:
                importances = np.abs(coef[0])
            else:
                importances = np.abs(coef)
        else:
            return pd.DataFrame(columns=["feature", "importance", "importance_pct"])

        encoded_names = preprocessor.get_feature_names_out()
        original_features = sorted(self.feature_options.keys(), key=len, reverse=True)

        rows = []
        for name, score in zip(encoded_names, importances):
            cleaned = name.split("__", 1)[-1]
            matched_feature = cleaned

            for feature in original_features:
                prefix = f"{feature}_"
                if cleaned.startswith(prefix):
                    matched_feature = feature
                    break

            rows.append((matched_feature, float(score)))

        importance_df = pd.DataFrame(rows, columns=["feature", "importance"])
        importance_df = (
            importance_df.groupby("feature", as_index=False)["importance"]
            .sum()
            .sort_values("importance", ascending=False, ignore_index=True)
        )

        total = float(importance_df["importance"].sum())
        if total > 0:
            importance_df["importance_pct"] = 100.0 * importance_df["importance"] / total
        else:
            importance_df["importance_pct"] = 0.0

        return importance_df.head(top_n)
