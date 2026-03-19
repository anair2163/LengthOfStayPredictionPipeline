"""Reusable preprocessing utilities for the LOS prediction pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET_COLUMN = "time_in_hospital"
RAW_DATA_DIR = Path("data") / "raw"
RAW_DATA_FILE = "diabetic_data.csv"

CANDIDATE_CATEGORICAL_FEATURES = [
    "race",
    "gender",
    "age",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "max_glu_serum",
    "A1Cresult",
    "insulin",
    "change",
    "diabetesMed",
]

CANDIDATE_NUMERIC_FEATURES = [
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
]

# Diagnosis code fields such as diag_1, diag_2, and diag_3 are excluded from the
# first production pipeline because they typically need careful cleaning and
# clinical grouping. A future enhancement can add diagnosis-category features
# once a consistent grouping strategy is defined.


def resolve_repo_root() -> Path:
    """Return the project root based on this module's location."""
    return Path(__file__).resolve().parents[1]


def resolve_data_path(data_path: str | Path | None = None) -> Path:
    """Resolve the raw data path, supporting an optional override."""
    if data_path is not None:
        candidate = Path(data_path).expanduser()
        return candidate if candidate.is_absolute() else resolve_repo_root() / candidate
    return resolve_repo_root() / RAW_DATA_DIR / RAW_DATA_FILE


def load_raw_data(data_path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw diabetic encounters dataset and normalize missing markers."""
    resolved_path = resolve_data_path(data_path)
    if not resolved_path.exists():
        raise FileNotFoundError(
            "Raw dataset not found. Expected diabetic_data.csv at "
            f"{resolved_path}. Place the file in data/raw/ or provide a custom path."
        )

    df = pd.read_csv(resolved_path)
    return df.replace("?", np.nan)


def clean_target(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> tuple[pd.DataFrame, pd.Series]:
    """Drop rows with missing or invalid targets and return cleaned frame and target."""
    if target_column not in df.columns:
        raise KeyError(f"Required target column '{target_column}' is missing from the dataset.")

    target = pd.to_numeric(df[target_column], errors="coerce")
    valid_rows = target.notna()

    cleaned_df = df.loc[valid_rows].copy()
    cleaned_target = target.loc[valid_rows].astype(int)

    if cleaned_df.empty:
        raise ValueError("No valid rows remain after removing missing or invalid target values.")

    return cleaned_df, cleaned_target


def get_available_feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return candidate feature lists filtered to columns that exist in the dataset."""
    categorical_features = [col for col in CANDIDATE_CATEGORICAL_FEATURES if col in df.columns]
    numeric_features = [col for col in CANDIDATE_NUMERIC_FEATURES if col in df.columns]

    if not categorical_features and not numeric_features:
        raise ValueError("None of the configured candidate features are present in the dataset.")

    return categorical_features, numeric_features


def build_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    """Build a preprocessing transformer for numeric and categorical features."""
    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers=transformers)


def prepare_dataset(data_path: str | Path | None = None) -> tuple[pd.DataFrame, pd.Series, ColumnTransformer, dict[str, Any]]:
    """Load, clean, and prepare features, target, transformer, and metadata."""
    raw_df = load_raw_data(data_path=data_path)
    cleaned_df, y = clean_target(raw_df, target_column=TARGET_COLUMN)
    categorical_features, numeric_features = get_available_feature_lists(cleaned_df)

    selected_features = numeric_features + categorical_features
    X = cleaned_df[selected_features].copy()
    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    feature_metadata: dict[str, Any] = {
        "target_column": TARGET_COLUMN,
        "all_features": selected_features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "n_rows": len(cleaned_df),
        "source_path": str(resolve_data_path(data_path)),
    }

    return X, y, preprocessor, feature_metadata


def main() -> None:
    """Preview the prepared dataset and configured preprocessing pipeline."""
    X, y, preprocessor, feature_metadata = prepare_dataset()
    print(f"Prepared feature matrix shape: {X.shape}")
    print(f"Prepared target shape: {y.shape}")
    print("Feature metadata:")
    print(feature_metadata)
    print("Preprocessor:")
    print(preprocessor)


if __name__ == "__main__":
    main()
