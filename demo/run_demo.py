import os
from datetime import datetime, timezone

from dummy_data import (
    make_dummy_grid, make_dummy_forest, make_dummy_cap_warning,
    make_dummy_weather, apply_dummy_nlp_to_grid
)
from dummy_agents import fuse_features, predict_risk, estimate_uncertainty, make_explanations
from render_maps import render_and_save_maps

def main():
    # 1) Grid
    grid = make_dummy_grid(n=30)

    # 2) Forest
    forest = make_dummy_forest(grid)

    # 3) CAP warning + NLP features
    cap = make_dummy_cap_warning()
    nlp = apply_dummy_nlp_to_grid(grid, cap)

    # 4) Weather (24h)
    start = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    weather = make_dummy_weather(grid, start_time=start, hours=24)

    # 5) Fuse + Predict + Uncertainty + Explanations
    fused = fuse_features(weather, forest, nlp)
    preds = predict_risk(fused)
    preds = estimate_uncertainty(preds)
    preds = make_explanations(preds)

    # 6) Save outputs
    out_dir = "results/demo"
    os.makedirs(out_dir, exist_ok=True)
    preds.to_parquet(os.path.join(out_dir, "demo_predictions.parquet"), index=False)

    # 7) Render maps for a representative time
    t0 = preds["valid_time"].min()
    render_and_save_maps(grid, preds, out_dir=out_dir, time_point=t0)

    # 8) Print a few example explanations
    sample = preds[preds["valid_time"] == t0].sort_values("risk_prob", ascending=False).head(5)
    print("\nTop-5 risky cells at", t0)
    for _, r in sample.iterrows():
        print(f"- cell {r['cell_id']}: risk={r['risk_prob']:.2f}, unc={r['risk_uncertainty']:.2f}, sev={r['severity_class']}")
        print(f"  {r['explanation']}")

    print(f"\nSaved: {out_dir}/demo_predictions.parquet")
    print(f"Saved: {out_dir}/risk_map.png")
    print(f"Saved: {out_dir}/uncertainty_map.png")

if __name__ == "__main__":
    main()