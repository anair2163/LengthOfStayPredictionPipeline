"""Train and compare LOS prediction models."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline

from preprocess import prepare_dataset, resolve_repo_root, split_encounters

MODEL_OUTPUT_DIR = "models"
BEST_MODEL_FILENAME = "best_los_model.joblib"
COMPARISON_FILENAME = "model_comparison.csv"
DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_CV_FOLDS = 2
DEFAULT_TUNING_SAMPLE_SIZE = 20000


def resolve_output_dir(output_dir: str | Path | None = None) -> Path:
    """Resolve the output directory for trained artifacts."""
    if output_dir is None:
        return resolve_repo_root() / MODEL_OUTPUT_DIR

    candidate = Path(output_dir).expanduser()
    return candidate if candidate.is_absolute() else resolve_repo_root() / candidate


def build_model_registry(random_state: int = DEFAULT_RANDOM_STATE) -> dict[str, object]:
    """Return the candidate models to compare."""
    return {
        "DummyRegressor": DummyRegressor(strategy="median"),
        "LinearRegression": LinearRegression(),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=random_state, max_iter=10000),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=200,
            random_state=random_state,
            n_jobs=1,
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=random_state),
    }


def build_tuning_search_spaces(random_state: int = DEFAULT_RANDOM_STATE) -> dict[str, dict[str, list[object]]]:
    """Return small, practical hyperparameter grids for selected models."""
    return {
        "ElasticNet": {
            "model__alpha": [0.01, 0.1],
            "model__l1_ratio": [0.5, 0.8],
        },
        "RandomForestRegressor": {
            "model__n_estimators": [100],
            "model__max_depth": [None, 10],
            "model__min_samples_leaf": [1],
        },
        "GradientBoostingRegressor": {
            "model__n_estimators": [100],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2],
        },
    }


def build_training_pipeline(preprocessor, model: object) -> Pipeline:
    """Create a shared training pipeline for a single estimator."""
    return Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("model", model),
        ]
    )


def evaluate_regression_model(y_true, y_pred) -> dict[str, float]:
    """Compute regression metrics for model comparison."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2_score(y_true, y_pred),
    }


def train_and_compare_models(
    data_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[Pipeline, pd.DataFrame]:
    """Train candidate models, compare them, and save the best artifacts."""
    X, y, preprocessor, feature_metadata = prepare_dataset(data_path=data_path)
    patient_groups = feature_metadata["patient_groups"]
    X_train, X_test, y_train, y_test = split_encounters(
        X, y, patient_groups, test_size, random_state
    )

    X_cv = X_train.sample(n=min(len(X_train), DEFAULT_TUNING_SAMPLE_SIZE), random_state=random_state)
    y_cv = y_train.loc[X_cv.index]
    groups_cv = patient_groups.loc[X_cv.index]
    cv = GroupKFold(n_splits=DEFAULT_CV_FOLDS)
    scoring = "neg_mean_absolute_error"
    search_spaces = build_tuning_search_spaces(random_state)
    candidates = []
    for name, model in build_model_registry(random_state).items():
        pipeline = build_training_pipeline(preprocessor, model)
        scores = cross_val_score(pipeline, X_cv, y_cv, groups=groups_cv, cv=cv, scoring=scoring)
        candidates.append((name, pipeline, -scores.mean(), {}))

        if name in search_spaces:
            search = GridSearchCV(pipeline, search_spaces[name], scoring=scoring, cv=cv, refit=False)
            search.fit(X_cv, y_cv, groups=groups_cv)
            tuned = clone(pipeline).set_params(**search.best_params_)
            candidates.append((f"{name}_tuned", tuned, -search.best_score_, search.best_params_))

    best_name, best_pipeline, _, _ = min(candidates, key=lambda candidate: candidate[2])
    comparison_rows = []
    for name, pipeline, cv_mae, params in sorted(candidates, key=lambda candidate: candidate[2]):
        # Evaluate the selected model and the median benchmark on the untouched test patients.
        test_metrics = {}
        if name in (best_name, "DummyRegressor"):
            pipeline.fit(X_train, y_train)
            test_metrics = evaluate_regression_model(y_test, pipeline.predict(X_test))
        comparison_rows.append({
            "model": name,
            "cv_mae": cv_mae,
            "test_mae": test_metrics.get("mae"),
            "test_rmse": test_metrics.get("rmse"),
            "test_r2": test_metrics.get("r2"),
            "best_params": str(params),
            "n_train_rows": len(X_train),
            "n_test_rows": len(X_test),
            "n_features": len(feature_metadata["all_features"]),
        })

    comparison_df = pd.DataFrame(comparison_rows)

    output_path = resolve_output_dir(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Keep evaluation tied to the training split even when CLI defaults are overridden.
    best_pipeline.evaluation_split_ = {"test_size": test_size, "random_state": random_state}
    joblib.dump(best_pipeline, output_path / BEST_MODEL_FILENAME, compress=3)
    comparison_df.to_csv(output_path / COMPARISON_FILENAME, index=False)

    return best_pipeline, comparison_df


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Train and compare LOS prediction models.")
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
        help="Optional override directory for saved artifacts.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Fraction of patients reserved for testing.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed for train/test splitting and model reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    """Run model training from the command line."""
    args = parse_args()
    best_pipeline, comparison_df = train_and_compare_models(
        data_path=args.data_path,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print("Model comparison (sorted by patient-grouped cross-validation MAE):")
    print(comparison_df.to_string(index=False))
    print()
    print(f"Best model by cross-validation MAE: {comparison_df.iloc[0]['model']}")
    print(f"Saved best pipeline to: {resolve_output_dir(args.output_dir) / BEST_MODEL_FILENAME}")
    print(f"Saved comparison table to: {resolve_output_dir(args.output_dir) / COMPARISON_FILENAME}")
    print(f"Fitted pipeline: {best_pipeline}")


if __name__ == "__main__":
    main()
