import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def to_grid_image(df_slice, n_side):
    # df_slice has columns x, y, value
    img = np.full((n_side, n_side), np.nan, dtype=float)
    for _, r in df_slice.iterrows():
        img[int(r["y"]), int(r["x"])] = float(r["value"])
    return img

def render_and_save_maps(grid_df, preds_df, out_dir, time_point):
    os.makedirs(out_dir, exist_ok=True)

    # pick a single valid_time
    dft = preds_df[preds_df["valid_time"] == time_point].copy()
    n_side = int(np.sqrt(len(grid_df)))

    # risk map
    risk = dft.merge(grid_df[["cell_id", "x", "y"]], on="cell_id", how="left")
    risk_img = to_grid_image(risk.assign(value=risk["risk_prob"]), n_side)

    plt.figure()
    plt.imshow(risk_img, origin="lower")
    plt.title(f"Windthrow Risk Probability\n{time_point}")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "risk_map.png"), dpi=160)
    plt.close()

    # uncertainty map
    unc_img = to_grid_image(risk.assign(value=risk["risk_uncertainty"]), n_side)
    plt.figure()
    plt.imshow(unc_img, origin="lower")
    plt.title(f"Risk Uncertainty\n{time_point}")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "uncertainty_map.png"), dpi=160)
    plt.close()