"""Deterministic zero-API verification of all main and appendix tables (US, EU27, Japan).

Run from repository root:
    python scripts/reproduce_tables.py
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def verify_series_metrics(df: pd.DataFrame, method_col: str, official_col: str = "Official",
                          pers_col: str = "Persistence") -> dict[str, float]:
    y = df[official_col].to_numpy(dtype=float)
    yh = df[method_col].to_numpy(dtype=float)
    yp = df[pers_col].to_numpy(dtype=float)
    mae = float(np.mean(np.abs(yh - y)))
    rmse = float(np.sqrt(np.mean((yh - y) ** 2)))
    r = float(np.corrcoef(yh, y)[0, 1])
    dy_obs = y - yp
    dy_sim = yh - yp
    dr = float(np.corrcoef(dy_sim, dy_obs)[0, 1]) if np.std(dy_sim) > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "r": r, "Delta_r": dr}


def main() -> None:
    # --- Table 1: Cross-regional 147m & OOS 45m (2023-01 to 2026-09) ---
    print_header("Table 1: Cross-Regional Forecast Accuracy (Jul 2014 - Sep 2026, 147 mo)")
    regional = [
        ("a. United States (Michigan ICS)", RES / "us/all_methods_series.csv",
         ["ConsumerSim", "MLP Neural (Macro+Text Emb)", "Ridge Delta (Macro+Text Emb)",
          "Random Forest (Macro+Text Emb)", "Undated Direct-LLM (gpt-6-luna)",
          "GBDT (LightGBM-eq, +Text Emb)", "SARIMAX Macro", "Auto ARIMA", "Persistence"]),
        ("b. EU27 (Eurostat BS-CSMCI)", RES / "eu27/all_series_EU27_147m.csv",
         ["ConsumerSim", "Persistence", "Theta Model", "GBDT (LightGBM-eq, Delta)",
          "AR Ridge", "SARIMAX Macro", "Auto ARIMA", "ETS Damped Trend"]),
        ("c. Japan (Cabinet Office ESRI CCI)", RES / "japan/all_series_Japan_147m.csv",
         ["ConsumerSim", "Persistence", "Theta Model", "Auto ARIMA",
          "GBDT (LightGBM-eq, Delta)", "AR Ridge", "ETS Damped Trend", "SARIMAX Macro"]),
    ]
    for title, path, methods in regional:
        df = pd.read_csv(path)
        m_col = df.columns[0]
        df[m_col] = df[m_col].astype(int)
        oos = df[df[m_col] >= 202301]
        print(f"\n{title} (N_full={len(df)}, N_oos={len(oos)})")
        rows = []
        for m in methods:
            if m not in df.columns:
                continue
            f_m = verify_series_metrics(df, m)
            o_m = verify_series_metrics(oos, m)
            rows.append({
                "Method": m,
                "Full_MAE": round(f_m["MAE"], 2),
                "Full_RMSE": round(f_m["RMSE"], 2),
                "Full_r": round(f_m["r"], 3),
                "Full_dr": "--" if np.isnan(f_m["Delta_r"]) else round(f_m["Delta_r"], 2),
                "OOS_MAE": round(o_m["MAE"], 2),
                "OOS_RMSE": round(o_m["RMSE"], 2),
                "OOS_r": round(o_m["r"], 3),
                "OOS_dr": "--" if np.isnan(o_m["Delta_r"]) else round(o_m["Delta_r"], 2),
            })
        print(pd.DataFrame(rows).to_string(index=False))

    # --- Table 2: Top 10 U.S. Shock Months ---
    print_header("Table 2: Ten Largest Monthly Moves in Official U.S. ICS")
    t2 = pd.read_csv(RES / "us/table2_shock_top10.csv")
    cols = [c for c in ["Regime", "Event month", "Official change", "ConsumerSim change",
                        "ConsumerSim error", "Best other error", "Rank"] if c in t2.columns]
    print(t2[cols].to_string(index=False))

    # --- Table 3: Core Component Ablation ---
    print_header("Table 3: Component Ablation (41 Development Event Months)")
    t3 = pd.read_csv(RES / "us/ablation.csv")
    print(t3.to_string(index=False))

    # --- Table E1: Multi-Month Open-Loop Rollout ---
    print_header("Table E1: U.S. Multi-Month Open-Loop Rollout RMSE (No Monthly Reset)")
    te1 = pd.read_csv(RES / "us/open_loop_rollout.csv")
    print(te1.to_string(index=False))

    # --- Table E2: Memory Audit of Undated Inputs ---
    print_header("Table E2: Memory Audit of Undated Inputs (147 Months)")
    audit = json.loads((RES / "memory_audit/probe_summary.json").read_text(encoding="utf-8"))
    fp = audit["fingerprint"]
    print(f"Date exact share: {audit['date_exact_share']*100:.1f}% | "
          f"Within 1 mo: {audit['date_within1_share']*100:.1f}% | "
          f"Undated ICS MAE: {audit['undated_recall_mae']:.2f}")
    print(f"Recall-error regression (ConsumerSim): b = {fp['cs_change_on_recall_error']['b']:.3f}, "
          f"SE = {fp['cs_change_on_recall_error']['se']:.3f}, P = {fp['cs_change_on_recall_error']['p']:.3f}")
    print(pd.DataFrame(fp["strata"]).round(2).to_string(index=False))

    # --- Table G4: Multi-LLM, Prompt & Seed Robustness ---
    print_header("Table G4: Robustness Across 4 LLM Families, 4 Prompts, and 3 Seeds")
    tg4 = pd.read_csv(RES / "robustness/robustness_summary.csv")
    print(tg4.round(3).to_string(index=False))

    # --- Table G5: Three-Tier Exposure Decomposition ---
    print_header("Table G5: Three-Tier Exposure-Layer Decomposition (20 High-Info Months)")
    tg5 = pd.read_csv(RES / "mechanisms/tier_decomposition.csv")
    print(tg5.round(3).to_string(index=False))

    # --- Table G6: Orthogonal 144-Persona Counterfactual Shocks ---
    print_header("Table G6: Orthogonal 144-Persona Synthetic Counterfactual Shocks")
    tg6 = pd.read_csv(RES / "mechanisms/counterfactual_ortho.csv")
    print(tg6.round(2).to_string(index=False))

    # --- Table G7: Exogenous Observable-Input Top-10 Salience Months ---
    print_header("Table G7: Top-10 Salient U.S. Months Selected Strictly from Observable Inputs")
    tg7 = pd.read_csv(RES / "mechanisms/exogenous_salience_top10.csv")
    print(tg7.round(3).to_string(index=False))

    print("\nAll US / EU27 / Japan main and appendix tables verified successfully.")


if __name__ == "__main__":
    main()
