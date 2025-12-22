import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def _synthetic_true_risk(df: pd.DataFrame) -> np.ndarray:
    """
    Ground-truth generator (hidden rule). We train LightGBM to learn this.
    """
    gust = df["gust10"].to_numpy()
    spruce = df["spruce_pct"].to_numpy()
    age = df["stand_age"].to_numpy()
    height = df["avg_height"].to_numpy()
    peat = (df["soil_type"] == "peat").astype(int)
    prcp = df["prcp_conv"].to_numpy()
    cap = df["cap_sev_score"].to_numpy()
    treefall = df["treefall_flag"].to_numpy()

    # Underlying logit-style risk
    score = (
        0.20 * (gust - 15) +
        1.10 * (spruce - 0.45) +
        0.010 * (age - 70) +
        0.015 * (height - 18) +
        0.45  * peat +
        0.07  * (prcp - 1.0) +
        0.95  * cap +
        0.35  * treefall
    )
    prob = 1 / (1 + np.exp(-score))
    return np.clip(prob, 0, 1)

def make_synthetic_labels(df: pd.DataFrame, seed=999) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p = _synthetic_true_risk(df)

    # add noise to mimic imperfect labels
    noisy_p = np.clip(p + rng.normal(0, 0.06, size=len(p)), 0, 1)
    y = (noisy_p > 0.5).astype(int)

    out = df.copy()
    out["y_damage"] = y
    out["y_prob_true"] = p
    return out

def train_lightgbm(df: pd.DataFrame, feature_cols, label_col="y_damage", seed=13):
    X = df[feature_cols]
    y = df[label_col]

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    model = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=48,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        class_weight="balanced",
    )
    model.fit(Xtr, ytr)

    proba = model.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, proba)
    return model, auc