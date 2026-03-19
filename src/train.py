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
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline

from preprocess import prepare_dataset, resolve_repo_root

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
        "DummyRegressor": DummyRegressor(strategy="mean"),
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


def build_cv_strategy(random_state: int = DEFAULT_RANDOM_STATE, n_splits: int = DEFAULT_CV_FOLDS) -> KFold:
    """Create a lightweight cross-validation strategy for tuning."""
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def get_tuning_sample(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = DEFAULT_RANDOM_STATE,
    max_rows: int = DEFAULT_TUNING_SAMPLE_SIZE,
) -> tuple[pd.DataFrame, pd.Series]:
    """Cap the data used for hyperparameter search to keep runtime practical."""
    if len(X_train) <= max_rows:
        return X_train, y_train

    sampled_indices = X_train.sample(n=max_rows, random_state=random_state).index
    return X_train.loc[sampled_indices], y_train.loc[sampled_indices]


def fit_model_candidate(
    model_name: str,
    estimator: object,
    preprocessor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_strategy: KFold,
    tuning_search_spaces: dict[str, dict[str, list[object]]],
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[Pipeline, dict[str, object]]:
    """Fit either an untuned baseline or a lightly tuned model candidate."""
    if model_name == "DummyRegressor":
        pipeline = Pipeline(steps=[("model", estimator)])
        fitted_pipeline = pipeline.fit(X_train, y_train)
        return fitted_pipeline, {"variant": "baseline", "cv_mae": pd.NA, "best_params": {}}

    pipeline = build_training_pipeline(preprocessor=preprocessor, model=estimator)
    fitted_pipeline = pipeline.fit(X_train, y_train)
    baseline_details: dict[str, object] = {"variant": "baseline", "cv_mae": pd.NA, "best_params": {}}

    if model_name not in tuning_search_spaces:
        return fitted_pipeline, baseline_details

    X_tune, y_tune = get_tuning_sample(
        X_train=X_train,
        y_train=y_train,
        random_state=random_state,
    )

    search = GridSearchCV(
        estimator=build_training_pipeline(preprocessor=preprocessor, model=clone(estimator)),
        param_grid=tuning_search_spaces[model_name],
        scoring="neg_mean_absolute_error",
        cv=cv_strategy,
        n_jobs=1,
        refit=True,
    )
    search.fit(X_tune, y_tune)
    tuned_pipeline = build_training_pipeline(preprocessor=preprocessor, model=clone(estimator))
    tuned_pipeline.set_params(**search.best_params_)
    tuned_pipeline.fit(X_train, y_train)
    tuned_details: dict[str, object] = {
        "variant": "tuned",
        "cv_mae": -search.best_score_,
        "best_params": search.best_params_,
    }
    return fitted_pipeline, baseline_details, tuned_pipeline, tuned_details


def train_and_compare_models(
    data_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[Pipeline, pd.DataFrame]:
    """Train candidate models, compare them, and save the best artifacts."""
    X, y, preprocessor, feature_metadata = prepare_dataset(data_path=data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    model_registry = build_model_registry(random_state=random_state)
    tuning_search_spaces = build_tuning_search_spaces(random_state=random_state)
    cv_strategy = build_cv_strategy(random_state=random_state)
    trained_pipelines: dict[str, Pipeline] = {}
    comparison_rows: list[dict[str, float | int | str]] = []

    for model_name, estimator in model_registry.items():
        fit_result = fit_model_candidate(
            model_name=model_name,
            estimator=estimator,
            preprocessor=preprocessor,
            X_train=X_train,
            y_train=y_train,
            cv_strategy=cv_strategy,
            tuning_search_spaces=tuning_search_spaces,
            random_state=random_state,
        )

        candidates: list[tuple[str, Pipeline, dict[str, object]]]
        if len(fit_result) == 2:
            fitted_pipeline, fitted_details = fit_result
            candidates = [(model_name, fitted_pipeline, fitted_details)]
        else:
            baseline_pipeline, baseline_details, tuned_pipeline, tuned_details = fit_result
            candidates = [
                (model_name, baseline_pipeline, baseline_details),
                (f"{model_name}_tuned", tuned_pipeline, tuned_details),
            ]

        for candidate_name, candidate_pipeline, candidate_details in candidates:
            predictions = candidate_pipeline.predict(X_test)
            metrics = evaluate_regression_model(y_test, predictions)

            comparison_rows.append(
                {
                    "model": candidate_name,
                    "base_model": model_name,
                    "variant": candidate_details["variant"],
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "r2": metrics["r2"],
                    "cv_mae": candidate_details["cv_mae"],
                    "best_params": str(candidate_details["best_params"]),
                    "n_train_rows": len(X_train),
                    "n_test_rows": len(X_test),
                    "n_features": len(feature_metadata["all_features"]),
                }
            )
            trained_pipelines[candidate_name] = candidate_pipeline

    comparison_df = pd.DataFrame(comparison_rows).sort_values("mae", ascending=True).reset_index(drop=True)
    best_model_name = comparison_df.iloc[0]["model"]
    best_pipeline = trained_pipelines[best_model_name]

    output_path = resolve_output_dir(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_pipeline, output_path / BEST_MODEL_FILENAME)
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
        help="Fraction of rows reserved for testing.",
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

    print("Model comparison (sorted by MAE):")
    print(comparison_df.to_string(index=False))
    print()
    print(f"Best model by MAE: {comparison_df.iloc[0]['model']}")
    print(f"Saved best pipeline to: {resolve_output_dir(args.output_dir) / BEST_MODEL_FILENAME}")
    print(f"Saved comparison table to: {resolve_output_dir(args.output_dir) / COMPARISON_FILENAME}")
    print(f"Fitted pipeline: {best_pipeline}")


if __name__ == "__main__":
    main()
