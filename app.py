"""
Wine Quality Prediction — Streamlit App
=========================================
Interactive app that loads the trained soft-voting ensemble (Random Forest +
Gradient Boosting + XGBoost) from the companion notebook and lets a user enter
a wine's physicochemical properties to get a live quality prediction, a
confidence score, and a plain-language explanation of the result.

Run with:  streamlit run app.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

CATEGORY_COLORS = {"Low": "#c0392b", "Medium": "#f39c12", "High": "#27ae60"}
CATEGORY_ICONS = {"Low": "⚠️", "Medium": "🙂", "High": "🏆"}
CATEGORY_BLURB = {
    "Low": "This wine's profile resembles samples that expert tasting panels scored 5 or below — likely to taste average-to-poor.",
    "Medium": "This wine's profile resembles samples that expert tasting panels scored 6 — solid, standard drinking quality.",
    "High": "This wine's profile resembles samples that expert tasting panels scored 7 or above — a premium, excellent wine.",
}

FEATURE_INFO = {
    "fixed acidity": ("g/dm³", "Tartaric acid content; contributes to a wine's crispness."),
    "volatile acidity": ("g/dm³", "Acetic acid content; too much gives an unpleasant vinegar taste."),
    "citric acid": ("g/dm³", "Adds freshness and flavor in small quantities."),
    "residual sugar": ("g/dm³", "Sugar remaining after fermentation stops."),
    "chlorides": ("g/dm³", "Salt content in the wine."),
    "free sulfur dioxide": ("mg/dm³", "Prevents microbial growth and oxidation."),
    "total sulfur dioxide": ("mg/dm³", "Free + bound forms of SO2; becomes noticeable above ~50 mg/dm³."),
    "density": ("g/cm³", "Close to water's density; depends on alcohol and sugar content."),
    "pH": ("scale 0-14", "Describes how acidic the wine is (most wines sit between 2.9-3.9)."),
    "sulphates": ("g/dm³", "A wine additive that contributes to SO2 levels; acts as an antioxidant."),
    "alcohol": ("% vol.", "Alcohol content by volume — historically the strongest driver of perceived quality."),
}

DEFAULTS = {
    "red": {
        "fixed acidity": 8.32, "volatile acidity": 0.53, "citric acid": 0.27,
        "residual sugar": 2.54, "chlorides": 0.087, "free sulfur dioxide": 15.87,
        "total sulfur dioxide": 46.47, "density": 0.9967, "pH": 3.31,
        "sulphates": 0.66, "alcohol": 10.4,
    },
    "white": {
        "fixed acidity": 6.85, "volatile acidity": 0.28, "citric acid": 0.33,
        "residual sugar": 6.39, "chlorides": 0.046, "free sulfur dioxide": 35.31,
        "total sulfur dioxide": 138.36, "density": 0.9940, "pH": 3.19,
        "sulphates": 0.49, "alcohol": 10.5,
    },
}


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    ensemble = joblib.load(os.path.join(MODELS_DIR, "wine_quality_ensemble.joblib"))
    rf_for_shap = joblib.load(os.path.join(MODELS_DIR, "wine_quality_rf.joblib"))
    label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.joblib"))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_cols.joblib"))
    with open(os.path.join(MODELS_DIR, "metrics.json")) as f:
        metrics = json.load(f)
    explainer = shap.TreeExplainer(rf_for_shap)
    return ensemble, rf_for_shap, label_encoder, feature_cols, metrics, explainer


try:
    ensemble, rf_model, le, FEATURE_COLS, METRICS, explainer = load_artifacts()
    MODELS_LOADED = True
except Exception as e:
    MODELS_LOADED = False
    LOAD_ERROR = str(e)


# ---------------------------------------------------------------------------
# Sidebar — inputs
# ---------------------------------------------------------------------------
st.sidebar.title("🍷 Wine Properties")
st.sidebar.caption("Enter the physicochemical lab measurements for a wine sample.")

wine_type = st.sidebar.radio("Wine type", ["red", "white"], horizontal=True)
st.sidebar.markdown("---")

if st.sidebar.button("↺ Reset to typical values for this wine type"):
    for k, v in DEFAULTS[wine_type].items():
        st.session_state[k] = v

feature_ranges = {
    "fixed acidity": (3.8, 16.0, 0.1),
    "volatile acidity": (0.05, 1.7, 0.01),
    "citric acid": (0.0, 1.7, 0.01),
    "residual sugar": (0.5, 66.0, 0.1),
    "chlorides": (0.005, 0.7, 0.001),
    "free sulfur dioxide": (1.0, 290.0, 1.0),
    "total sulfur dioxide": (5.0, 440.0, 1.0),
    "density": (0.985, 1.005, 0.0001),
    "pH": (2.7, 4.1, 0.01),
    "sulphates": (0.2, 2.1, 0.01),
    "alcohol": (8.0, 15.0, 0.1),
}

user_inputs = {}
for feat, (lo, hi, step) in feature_ranges.items():
    unit, help_text = FEATURE_INFO[feat]
    default_val = DEFAULTS[wine_type][feat]
    key = feat
    if key not in st.session_state:
        st.session_state[key] = default_val
    user_inputs[feat] = st.sidebar.slider(
        f"{feat.title()} ({unit})", min_value=float(lo), max_value=float(hi),
        step=float(step), key=key, help=help_text,
    )

st.sidebar.markdown("---")
st.sidebar.caption(
    "Model: Soft-voting ensemble (Random Forest + Gradient Boosting + XGBoost), "
    "trained on the UCI Wine Quality dataset (red + white, 5,320 samples after "
    "de-duplication)."
)

# ---------------------------------------------------------------------------
# Main panel — header
# ---------------------------------------------------------------------------
st.title("🍷 Wine Quality Prediction")
st.markdown(
    "Predict a wine's quality tier — **Low / Medium / High** — from its lab "
    "chemistry, using a tuned ensemble model trained on the classic UCI Wine "
    "Quality dataset (Cortez et al., 2009)."
)

if not MODELS_LOADED:
    st.error(
        "Model artifacts could not be loaded. Make sure the `models/` folder "
        f"(produced by the notebook's Section 13) sits next to `app.py`.\n\n"
        f"Details: {LOAD_ERROR}"
    )
    st.stop()

col_left, col_right = st.columns([1.15, 1])

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
input_row = {**user_inputs, "wine_type": 0 if wine_type == "red" else 1}
X_input = pd.DataFrame([input_row])[FEATURE_COLS]

proba = ensemble.predict_proba(X_input)[0]
pred_idx = int(np.argmax(proba))
pred_label = le.classes_[pred_idx]
confidence = float(proba[pred_idx])

proba_df = pd.DataFrame({"Category": le.classes_, "Probability": proba}).sort_values(
    "Probability", ascending=False
)

with col_left:
    st.subheader("Prediction")
    color = CATEGORY_COLORS[pred_label]
    icon = CATEGORY_ICONS[pred_label]
    st.markdown(
        f"""
        <div style="background-color:{color}22; border-left: 6px solid {color};
                    border-radius: 8px; padding: 1.1rem 1.4rem; margin-bottom: 0.8rem;">
            <div style="font-size:1.6rem; font-weight:700; color:{color};">
                {icon} {pred_label} Quality
            </div>
            <div style="font-size:1.0rem; margin-top:0.3rem;">
                Confidence: <b>{confidence:.1%}</b>
            </div>
            <div style="font-size:0.92rem; margin-top:0.5rem; color:#333;">
                {CATEGORY_BLURB[pred_label]}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig = go.Figure(
        go.Bar(
            x=proba_df["Probability"],
            y=proba_df["Category"],
            orientation="h",
            marker_color=[CATEGORY_COLORS[c] for c in proba_df["Category"]],
            text=[f"{p:.1%}" for p in proba_df["Probability"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Class Probabilities", xaxis_title="Probability", yaxis_title="",
        xaxis_range=[0, 1], height=280, margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Explanation (SHAP)
# ---------------------------------------------------------------------------
with col_right:
    st.subheader("Why this prediction?")
    st.caption(
        "SHAP values show how much each property pushed the prediction toward "
        f"**{pred_label}** relative to the model's average output, for this "
        "specific wine sample."
    )

    shap_values = explainer.shap_values(X_input.values)
    # shap_values shape: (n_samples, n_features, n_classes) for recent shap versions
    sv = np.array(shap_values)
    if sv.ndim == 3:
        class_shap = sv[0, :, pred_idx]
    else:  # older shap API: list of per-class arrays
        class_shap = shap_values[pred_idx][0]

    shap_df = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "SHAP value": class_shap,
    })
    shap_df["Feature"] = shap_df["Feature"].replace({"wine_type": "wine type (red/white)"})
    shap_df["abs"] = shap_df["SHAP value"].abs()
    shap_df = shap_df.sort_values("abs", ascending=True).tail(8)

    colors = ["#27ae60" if v > 0 else "#c0392b" for v in shap_df["SHAP value"]]
    fig2 = go.Figure(
        go.Bar(
            x=shap_df["SHAP value"], y=shap_df["Feature"], orientation="h",
            marker_color=colors,
        )
    )
    fig2.update_layout(
        title=f"Top Feature Contributions Toward '{pred_label}'",
        xaxis_title=f"← pushes away from {pred_label}   |   pushes toward {pred_label} →",
        height=340, margin=dict(l=10, r=10, t=40, b=10),
    )
    fig2.add_vline(x=0, line_width=1, line_color="gray")
    st.plotly_chart(fig2, use_container_width=True)

    top_pos = shap_df.sort_values("SHAP value", ascending=False).iloc[-1] if (shap_df["SHAP value"] > 0).any() else None
    top_feats_pos = shap_df[shap_df["SHAP value"] > 0].sort_values("SHAP value", ascending=False)
    top_feats_neg = shap_df[shap_df["SHAP value"] < 0].sort_values("SHAP value")

    explanation_bits = []
    if len(top_feats_pos):
        f1 = top_feats_pos.iloc[0]["Feature"]
        explanation_bits.append(f"**{f1}** supported a *{pred_label}* rating")
    if len(top_feats_neg):
        f2 = top_feats_neg.iloc[0]["Feature"]
        explanation_bits.append(f"**{f2}** worked against it")
    if explanation_bits:
        st.markdown("In plain terms: " + ", while ".join(explanation_bits) + ".")

st.markdown("---")

# ---------------------------------------------------------------------------
# Model performance panel
# ---------------------------------------------------------------------------
st.subheader("📊 Model Performance (held-out test set)")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", f"{METRICS['accuracy']:.1%}")
m2.metric("Macro F1", f"{METRICS['macro_f1']:.3f}")
m3.metric("Macro ROC-AUC", f"{METRICS['macro_roc_auc']:.3f}")
m4.metric("5-fold CV accuracy", f"{METRICS['cv_mean_accuracy']:.1%} ± {METRICS['cv_std_accuracy']:.1%}")

with st.expander("See confusion matrix"):
    cm = np.array(METRICS["confusion_matrix"])
    classes = METRICS["classes"]
    fig3 = go.Figure(
        data=go.Heatmap(
            z=cm, x=classes, y=classes, colorscale="Blues", showscale=False,
            text=cm, texttemplate="%{text}",
        )
    )
    fig3.update_layout(
        title="Confusion Matrix — Ensemble (Test Set)",
        xaxis_title="Predicted", yaxis_title="Actual", height=380,
    )
    st.plotly_chart(fig3, use_container_width=True)

st.caption(
    "Dataset: UCI Machine Learning Repository — Wine Quality Data Set "
    "(Cortez, Cerdeira, Almeida, Matos & Reis, 2009). Model: soft-voting "
    "ensemble of a tuned Random Forest, Gradient Boosting, and XGBoost, "
    "trained in the companion notebook `wine_quality_prediction.ipynb`."
)
