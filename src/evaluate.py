"""Evaluate the saved LOS prediction model on the held-out test set."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocess import prepare_dataset, resolve_repo_root
from train import (
    BEST_MODEL_FILENAME,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    resolve_output_dir,
)


def resolve_model_path(model_path: str | Path | None = None, output_dir: str | Path | None = None) -> Path:
    """Resolve the path to the saved best-model artifact."""
    if model_path is not None:
        candidate = Path(model_path).expanduser()
        return candidate if candidate.is_absolute() else resolve_repo_root() / candidate
    return resolve_output_dir(output_dir) / BEST_MODEL_FILENAME


def load_model(model_path: str | Path | None = None, output_dir: str | Path | None = None):
    """Load the saved best LOS model from disk."""
    resolved_model_path = resolve_model_path(model_path=model_path, output_dir=output_dir)
    if not resolved_model_path.exists():
        raise FileNotFoundError(
            f"Saved model not found at {resolved_model_path}. Run src/train.py first or provide --model-path."
        )
    return joblib.load(resolved_model_path)


def prepare_evaluation_split(
    data_path: str | Path | None = None,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict[str, object]]:
    """Recreate the same prepared dataset and train/test split used during training."""
    X, y, _, feature_metadata = prepare_dataset(data_path=data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test, feature_metadata


def evaluate_predictions(y_true: pd.Series, y_pred) -> dict[str, float]:
    """Compute core regression metrics."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2_score(y_true, y_pred),
    }


def get_feature_names_from_pipeline(model_pipeline) -> list[str] | None:
    """Return transformed feature names when the pipeline exposes them."""
    if not isinstance(model_pipeline, Pipeline) or "preprocessor" not in model_pipeline.named_steps:
        return None

    preprocessor = model_pipeline.named_steps["preprocessor"]
    if hasattr(preprocessor, "get_feature_names_out"):
        return list(preprocessor.get_feature_names_out())
    return None


def get_final_estimator(model_pipeline):
    """Return the fitted estimator from a pipeline, or the object itself if already an estimator."""
    if isinstance(model_pipeline, Pipeline) and "model" in model_pipeline.named_steps:
        return model_pipeline.named_steps["model"]
    return model_pipeline


def plot_error_analysis(
    y_test: pd.Series,
    predictions,
    X_test: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> None:
    """Create core error-analysis plots and save them under the models directory."""
    residuals = y_test - predictions
    abs_errors = (y_test - predictions).abs()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    axes[0].scatter(y_test, predictions, alpha=0.35, color="#4C78A8")
    diagonal_min = min(y_test.min(), predictions.min())
    diagonal_max = max(y_test.max(), predictions.max())
    axes[0].plot([diagonal_min, diagonal_max], [diagonal_min, diagonal_max], linestyle="--", color="black")
    axes[0].set_title("Actual vs predicted LOS")
    axes[0].set_xlabel("Actual time_in_hospital")
    axes[0].set_ylabel("Predicted time_in_hospital")

    axes[1].hist(residuals, bins=30, edgecolor="black", alpha=0.85, color="#F58518")
    axes[1].set_title("Residual distribution")
    axes[1].set_xlabel("Residual (actual - predicted)")
    axes[1].set_ylabel("Count")

    if "age" in X_test.columns:
        age_error = (
            pd.DataFrame({"age": X_test["age"].fillna("Missing"), "absolute_error": abs_errors})
            .groupby("age")["absolute_error"]
            .mean()
        )
        try:
            age_error = age_error.reindex(
                sorted(age_error.index, key=lambda x: int(str(x).strip("[]()").split("-")[0]))
            )
        except ValueError:
            pass
        axes[2].bar(age_error.index.astype(str), age_error.values, color="#54A24B")
        axes[2].tick_params(axis="x", rotation=45)
        axes[2].set_title("Average absolute error by age group")
        axes[2].set_xlabel("Age group")
        axes[2].set_ylabel("Average absolute error")
    else:
        axes[2].axis("off")
        axes[2].text(0.5, 0.5, "Age column not available", ha="center", va="center")

    if "gender" in X_test.columns:
        gender_error = (
            pd.DataFrame({"gender": X_test["gender"].fillna("Missing"), "absolute_error": abs_errors})
            .groupby("gender")["absolute_error"]
            .mean()
            .sort_values(ascending=False)
        )
        axes[3].bar(gender_error.index.astype(str), gender_error.values, color="#E45756")
        axes[3].set_title("Average absolute error by gender")
        axes[3].set_xlabel("Gender")
        axes[3].set_ylabel("Average absolute error")
    else:
        axes[3].axis("off")
        axes[3].text(0.5, 0.5, "Gender column not available", ha="center", va="center")

    plt.tight_layout()

    output_path = resolve_output_dir(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    figure_path = output_path / "evaluation_error_analysis.png"
    plt.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved error analysis plots to: {figure_path}")


def show_model_interpretability(model_pipeline) -> None:
    """Print feature importances or coefficients when the fitted estimator supports them."""
    estimator = get_final_estimator(model_pipeline)
    feature_names = get_feature_names_from_pipeline(model_pipeline)

    if hasattr(estimator, "feature_importances_") and feature_names:
        importance_df = (
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance": estimator.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .head(15)
            .reset_index(drop=True)
        )
        print("\nTop feature importances:")
        print(importance_df.to_string(index=False))

    if isinstance(estimator, (LinearRegression, ElasticNet)) and hasattr(estimator, "coef_") and feature_names:
        coefficient_df = (
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "coefficient": estimator.coef_,
                    "abs_coefficient": pd.Series(estimator.coef_).abs(),
                }
            )
            .sort_values("abs_coefficient", ascending=False)
            .head(15)
            .drop(columns="abs_coefficient")
            .reset_index(drop=True)
        )
        print("\nTop linear coefficients:")
        print(coefficient_df.to_string(index=False))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate the saved LOS prediction model.")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Optional override path to diabetic_data.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional override directory for model artifacts and plots.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional override path to the saved model joblib file.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Fraction of rows reserved for testing.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed for consistent train/test splitting.",
    )
    return parser.parse_args()


def main() -> None:
    """Run LOS model evaluation from the command line."""
    args = parse_args()
    model_pipeline = load_model(model_path=args.model_path, output_dir=args.output_dir)
    _, X_test, _, y_test, feature_metadata = prepare_evaluation_split(
        data_path=args.data_path,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    predictions = model_pipeline.predict(X_test)
    metrics = evaluate_predictions(y_test, predictions)

    print("Evaluation metrics on held-out test set:")
    print(f"MAE:  {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R2:   {metrics['r2']:.4f}")
    print()
    print("Feature metadata:")
    print(feature_metadata)

    plot_error_analysis(
        y_test=y_test,
        predictions=predictions,
        X_test=X_test,
        output_dir=args.output_dir,
    )
    show_model_interpretability(model_pipeline)


if __name__ == "__main__":
    main()
