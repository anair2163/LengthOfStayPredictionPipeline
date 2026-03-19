"""Run a simple LOS prediction using the saved best model."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from preprocess import resolve_repo_root
from train import BEST_MODEL_FILENAME, resolve_output_dir


def resolve_model_path(model_path: str | Path | None = None) -> Path:
    """Resolve the best-model path relative to the project root."""
    if model_path is not None:
        candidate = Path(model_path).expanduser()
        return candidate if candidate.is_absolute() else resolve_repo_root() / candidate
    return resolve_output_dir() / BEST_MODEL_FILENAME


def load_model(model_path: str | Path | None = None):
    """Load the saved LOS model pipeline."""
    resolved_path = resolve_model_path(model_path)
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Saved model not found at {resolved_path}. Run src/train.py first or provide a custom model path."
        )
    return joblib.load(resolved_path)


def build_example_input() -> pd.DataFrame:
    """Create a single realistic example row for inference."""
    return pd.DataFrame(
        [
            {
                "race": "Caucasian",
                "gender": "Female",
                "age": "[60-70)",
                "admission_type_id": 1,
                "discharge_disposition_id": 1,
                "admission_source_id": 7,
                "max_glu_serum": "None",
                "A1Cresult": ">8",
                "insulin": "Up",
                "change": "Ch",
                "diabetesMed": "Yes",
                "num_lab_procedures": 42,
                "num_procedures": 1,
                "num_medications": 15,
                "number_outpatient": 0,
                "number_emergency": 1,
                "number_inpatient": 0,
            }
        ]
    )


def main() -> None:
    """Load the trained model and print a sample LOS prediction."""
    model = load_model()

    # Replace the values below with your own patient or encounter details.
    # Keep the same column names so the saved pipeline can preprocess them correctly.
    example_row = build_example_input()

    prediction = model.predict(example_row)[0]

    print("Example input:")
    print(example_row.to_string(index=False))
    print()
    print(f"Predicted hospital length of stay: {prediction:.2f} days")


if __name__ == "__main__":
    main()
