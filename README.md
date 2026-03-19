# Length of Stay Prediction Pipeline

This repository contains a machine learning workflow for predicting hospital length of stay (LOS) using the UCI Diabetes 130-US Hospitals dataset.

## Business Problem

Hospitals need better visibility into expected inpatient stay duration to support bed planning, discharge coordination, staffing, and operational decision-making. This project frames `time_in_hospital` as the prediction target and builds a lightweight pipeline for exploration, preprocessing, model training, evaluation, and sample inference.

## Dataset

The project uses the UCI Diabetes 130-US Hospitals dataset, which includes patient encounter, admission, discharge, and utilization fields relevant to LOS analysis.

Required raw files:

- `diabetic_data.csv`
- `IDS_mapping.csv`

Place both files in `data/raw/` before running the notebook or Python scripts. Raw data is not committed to GitHub and `data/raw/` is intentionally ignored by git.

Additional notes are in [data/README.md](/Users/ahannair/Desktop/LengthOfStayPredictionPipeline/data/README.md).

## Repo Structure

```text
LengthOfStayPredictionPipeline/
├── data/
│   ├── raw/
│   └── README.md
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/
├── README.md
├── requirements.txt
└── .gitignore
```

- `notebooks/exploration.ipynb` contains exploratory analysis focused on LOS and business context.
- `src/preprocess.py` loads raw data and builds the shared preprocessing transformer.
- `src/train.py` compares baseline and lightly tuned regression models.
- `src/evaluate.py` evaluates the saved best model and produces error analysis.
- `src/predict.py` runs a sample LOS prediction with the saved pipeline.
- `models/` stores trained artifacts and evaluation outputs after the pipeline is run.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add `diabetic_data.csv` and `IDS_mapping.csv` to `data/raw/`.

## Run the Notebook

Launch Jupyter from the repository root:

```bash
jupyter notebook
```

Then open `notebooks/exploration.ipynb` to review data quality, target behavior, grouped LOS summaries, and business-oriented charts.

## Train and Evaluate

Train the models and save the best pipeline:

```bash
python src/train.py
```

Evaluate the saved best model and generate error analysis plots:

```bash
python src/evaluate.py
```

Run a simple example prediction:

```bash
python src/predict.py
```

## Results

Model metrics are generated after running the training and evaluation scripts. The comparison table is saved to `models/model_comparison.csv`, the best fitted pipeline is saved to `models/best_los_model.joblib`, and evaluation outputs such as error analysis plots are written to `models/`.
