import numpy as np
import pandas as pd

from model_lgbm import make_synthetic_labels, train_lightgbm

FEATURE_COLS = [
    "gust10", "wspd10", "wdir10", "prcp_conv", "mslp",
    "spruce_pct", "stand_age", "avg_height",
    "soil_is_peat",
    "cap_sev_score", "treefall_flag", "outage_flag", "road_flag"
]

def fuse_features(weather_df, forest_df, nlp_df):
    df = weather_df.merge(forest_df, on="cell_id", how="left").merge(nlp_df, on="cell_id", how="left")
    df["soil_is_peat"] = (df["soil_type"] == "peat").astype(int)
    return df

def train_demo_model(fused_df):
    """
    Train LightGBM on synthetic labels derived from fused features.
    Returns a trained model + reported AUC (demo credibility).
    """
    labeled = make_synthetic_labels(fused_df)
    model, auc = train_lightgbm(labeled, FEATURE_COLS, label_col="y_damage")
    return model, auc

def predict_with_model(model, fused_df):
    out = fused_df.copy()
    out["risk_prob"] = model.predict_proba(out[FEATURE_COLS])[:, 1]
    out["severity_class"] = pd.cut(
        out["risk_prob"],
        bins=[-1, 0.4, 0.7, 1.01],
        labels=["low", "medium", "high"]
    ).astype(str)
    return out

def estimate_uncertainty(pred_df):
    """
    Demo uncertainty: high when close to 0.5 + CAP text uncertainty.
    """
    prob = pred_df["risk_prob"].to_numpy()
    txt_unc = pred_df["text_uncertainty"].to_numpy()
    borderline = 1 - np.abs(prob - 0.5) * 2
    unc = np.clip(0.18 * borderline + txt_unc, 0, 1)

    out = pred_df.copy()
    out["risk_uncertainty"] = unc
    return out

def explain_row(r):
    drivers = []
    if r["gust10"] >= 22: drivers.append(f"gusts {r['gust10']:.0f} m/s")
    if r["wspd10"] >= 12: drivers.append(f"wind speed {r['wspd10']:.0f} m/s")
    if r["spruce_pct"] >= 0.6: drivers.append("spruce-dominated stands")
    if r["stand_age"] >= 85: drivers.append(f"older stands ({r['stand_age']:.0f}y)")
    if r["soil_is_peat"] == 1: drivers.append("peat soil")
    if r["cap_severity"] in ["moderate", "severe"]: drivers.append(f"CAP {r['cap_severity']}")
    if r["treefall_flag"] == 1: drivers.append("warning mentions treefall")

    if not drivers:
        drivers = ["moderate winds and average susceptibility"]

    return "Risk driven by " + ", ".join(drivers) + "."

def add_explanations(pred_df):
    out = pred_df.copy()
    out["explanation"] = out.apply(explain_row, axis=1)
    return out