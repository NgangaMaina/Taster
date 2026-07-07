# 🍷 Wine Quality Prediction

A complete ML project predicting wine quality (Low / Medium / High) from physicochemical
lab measurements, using the UCI Wine Quality dataset (red + white *Vinho Verde*, combined).

## Contents

```
wine_quality_prediction.ipynb   Full analysis notebook: EDA → baselines → ensemble → evaluation
app.py                          Streamlit app for interactive live predictions
requirements.txt                Python dependencies
data/
  winequality-red.csv            Raw UCI red wine data (1599 samples)
  winequality-white.csv          Raw UCI white wine data (4898 samples)
models/                          Artifacts produced by the notebook (Section 13), used by app.py
  wine_quality_ensemble.joblib    Final soft-voting ensemble (RF + GB + XGBoost)
  wine_quality_rf.joblib          Tuned Random Forest (used for SHAP explanations)
  label_encoder.joblib            Encodes Low/Medium/High <-> 0/1/2
  feature_cols.joblib             Exact feature order the models expect
  metrics.json                    Saved evaluation metrics shown in the app
  feature_stats.json              Per-feature descriptive stats
```

## Setup

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run the notebook

```bash
jupyter notebook wine_quality_prediction.ipynb
```

The notebook is self-contained: it reads the CSVs in `data/`, retrains every model from
scratch, regenerates all charts, and re-saves the `models/` artifacts. It executes end-to-end
with no errors (verified via `jupyter nbconvert --execute`).

## Run the web app

```bash
streamlit run app.py
```

Open the local URL Streamlit prints (typically `http://localhost:8501`). Use the sidebar to
enter a wine's chemistry (or click the reset button for a typical red/white starting point),
and the main panel shows:

- The predicted quality tier with a confidence score
- A bar chart of class probabilities
- A SHAP-based explanation of which properties pushed the prediction up or down
- The model's held-out test performance (accuracy, macro-F1, ROC-AUC, confusion matrix)

## Model summary

Final model: a **soft-voting ensemble** of a tuned Random Forest, Gradient Boosting, and
XGBoost, trained on an 80/20 stratified split of 5,320 de-duplicated samples.

| Metric | Value |
|---|---|
| Test accuracy | ~64% |
| Macro F1 | ~0.62 |
| Macro ROC-AUC (OvR) | ~0.81 |
| 5-fold CV accuracy | ~62% (std < 0.01) |

See Section 11 of the notebook for the full breakdown (per-class precision/recall,
confusion matrix, ROC curves) and Section 14 for a discussion of what limits accuracy
further (wine quality is a subjective sensory judgment that chemistry alone can't fully
capture) and ideas for future improvement.

## Dataset citation

P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. *Modeling wine preferences by
data mining from physicochemical properties.* Decision Support Systems, Elsevier,
47(4):547-553, 2009.
