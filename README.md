# Admission-Time Hospital Length of Stay

Can information available at admission improve a simple length-of-stay estimate for historical hospital encounters involving diabetes?

[Model evaluation](notebooks/los_regression_model.ipynb) · [Exploratory notebook](notebooks/exploration.ipynb) · [Results CSV](models/model_comparison.csv)

## Result

The four-day training median won model selection. On the same patient-grouped training folds, it achieved **2.268 days MAE**, compared with **2.308 days** for the best-performing nonconstant candidate, gradient boosting. On 20,153 held-out encounters, the selected median benchmark had **2.254 days MAE** and **2.993 days RMSE**. None of the tested models justified replacing the median for this cohort and feature set.

## Evaluation design

- **Prediction point:** Admission. Inputs include age band, admission type and source, and counts of prior outpatient, emergency and inpatient visits.
- **Target:** Days in hospital, observed from 1 to 14 in the [UCI 1999–2008 diabetes dataset](https://archive.ics.uci.edu/dataset/296/diabetes-130-us-hospitals-for-years-1999-2008).
- **Validation:** Keep each patient's encounters together. Compare the same grouped training folds using MAE, then score the selected model on patients held out from selection.
- **Interpretation:** Post-admission treatment counts and discharge disposition are used only for retrospective exploration. Generalization to other periods, hospitals and stays beyond 14 days remains untested.

[View the held-out error analysis](models/evaluation_error_analysis.png).

## Reproduce

Place UCI's `diabetic_data.csv` in `data/raw/`. Add `IDS_mapping.csv` for the exploratory notebook. From the repository root:

```bash
python -m pip install -r requirements.txt
python src/train.py
python src/evaluate.py
python src/predict.py
```

Training writes the model comparison and fitted pipeline to `models/`. Retrain before inference when using a different scikit-learn version.
