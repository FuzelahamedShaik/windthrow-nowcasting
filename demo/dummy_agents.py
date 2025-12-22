import numpy as np
import pandas as pd

def fuse_features(weather_df, forest_df, nlp_df):
    df = weather_df.merge(forest_df, on="cell_id", how="left").merge(nlp_df, on="cell_id", how="left")
    # Encode soil quickly
    df["soil_is_peat"] = (df["soil_type"] == "peat").astype(int)
    return df

def predict_risk(df):
    """
    Simple interpretable risk function (demo).
    Later replace with LightGBM + SHAP.
    """
    # normalize-ish
    gust = df["gust10"].to_numpy()
    spruce = df["spruce_pct"].to_numpy()
    age = df["stand_age"].to_numpy()
    peat = df["soil_is_peat"].to_numpy()
    prcp = df["prcp_conv"].to_numpy()
    cap = df["cap_sev_score"].to_numpy()
    treefall = df["treefall_flag"].to_numpy()

    # risk logit
    score = (
        0.18 * (gust - 15) +
        1.25 * (spruce - 0.4) +
        0.015 * (age - 60) +
        0.35 * peat +
        0.06 * (prcp - 1.0) +
        0.9  * cap +
        0.35 * treefall
    )
    prob = 1 / (1 + np.exp(-score))
    df = df.copy()
    df["risk_prob"] = np.clip(prob, 0, 1)

    # Severity tiers
    df["severity_class"] = pd.cut(
        df["risk_prob"],
        bins=[-1, 0.4, 0.7, 1.01],
        labels=["low", "medium", "high"]
    ).astype(str)

    return df

def estimate_uncertainty(df):
    """
    Dummy uncertainty: higher when CAP text is uncertain + when wind is borderline.
    Replace later with calibration + spatial uncertainty.
    """
    prob = df["risk_prob"].to_numpy()
    txt_unc = df["text_uncertainty"].to_numpy()
    borderline = 1 - np.abs(prob - 0.5) * 2  # max at 0.5
    unc = np.clip(0.15 * borderline + txt_unc, 0, 1)

    out = df.copy()
    out["risk_uncertainty"] = unc
    return out

def make_explanations(df, top_k=1):
    """
    Template-style explanation agent (demo).
    Later: SHAP + LLM.
    """
    out = df.copy()

    def explain_row(r):
        parts = []
        if r["gust10"] >= 22:
            parts.append(f"gusts ~{r['gust10']:.0f} m/s")
        if r["spruce_pct"] >= 0.6:
            parts.append("spruce-dominated stands")
        if r["soil_is_peat"] == 1:
            parts.append("peat soil (weaker anchoring)")
        if r["cap_severity"] in ["moderate", "severe"]:
            parts.append(f"CAP severity: {r['cap_severity']}")
        if r["treefall_flag"] == 1:
            parts.append("warning mentions treefall risk")

        if not parts:
            parts = ["moderate winds and average stand susceptibility"]

        return "High windthrow risk driven by " + ", ".join(parts) + "."

    out["explanation"] = out.apply(explain_row, axis=1)
    return out