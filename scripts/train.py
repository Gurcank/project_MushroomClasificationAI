from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from src.model import save_artifact, train_and_select_best
from src.preprocessing import load_raw_dataset, preprocess_dataframe, split_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train mushroom toxicity classifier")
    parser.add_argument(
        "--data",
        type=str,
        default="data/mushrooms.csv",
        help="Path to mushrooms.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/best_model.joblib",
        help="Path to saved model artifact",
    )
    parser.add_argument(
        "--drop-unknown-rows",
        action="store_true",
        help="Drop rows with '?' encoded values instead of keeping as 'unknown'",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to save leaderboard, metrics, and confusion matrix",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_df = load_raw_dataset(args.data)
    clean_df = preprocess_dataframe(raw_df, drop_unknown_rows=args.drop_unknown_rows)
    bundle = split_dataset(clean_df, test_size=0.2, random_state=42)

    result = train_and_select_best(
        X_train=bundle.X_train,
        y_train=bundle.y_train,
        X_test=bundle.X_test,
        y_test=bundle.y_test,
        random_state=42,
    )

    save_artifact(
        output_path=args.output,
        pipeline=result.best_pipeline,
        model_name=result.best_model_name,
        metrics=result.best_metrics,
        feature_options=bundle.feature_options,
    )

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    leaderboard_path = results_dir / "leaderboard.csv"
    metrics_path = results_dir / "metrics.json"
    cm_path = results_dir / "confusion_matrix.png"

    result.leaderboard.to_csv(leaderboard_path, index=False)
    metrics_payload = {
        "model": result.best_model_name,
        **result.best_metrics,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    tn = int(result.best_metrics["true_negative"])
    fp = int(result.best_metrics["false_positive"])
    fn = int(result.best_metrics["false_negative"])
    tp = int(result.best_metrics["true_positive"])
    cm = [[tn, fp], [fn, tp]]

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        cbar=False,
        xticklabels=["Pred: Yenilebilir", "Pred: Zehirli"],
        yticklabels=["Gercek: Yenilebilir", "Gercek: Zehirli"],
    )
    plt.title(f"Confusion Matrix ({result.best_model_name})")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()

    print("Best model:", result.best_model_name)
    print("Best metrics:")
    for k, v in result.best_metrics.items():
        print(f"  {k}: {v:.4f}")
    print("\nLeaderboard:")
    print(result.leaderboard.to_string(index=False))
    print(f"\nSaved artifact to: {Path(args.output).resolve()}")
    print(f"Saved leaderboard: {leaderboard_path.resolve()}")
    print(f"Saved metrics: {metrics_path.resolve()}")
    print(f"Saved confusion matrix: {cm_path.resolve()}")


if __name__ == "__main__":
    main()
