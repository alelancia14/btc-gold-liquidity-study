"""
Step 3 — Define & validate calm vs. stress regimes (NO model yet).

STLFSI4 sits near zero in normal times and rises during financial stress. We
classify each month stress/calm via a threshold on STLFSI4. The open question
is which threshold and whether results are robust to it, so we define three and
inspect them.

VAR variables (later): M2_growth, BTC_ret, GOLD_ret.
Regime classifier (here): STLFSI4.

Outputs:
  - data/processed/monthly_with_regimes.csv   (adds stress_A/B/C)
  - output/regime_classification.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROCESSED = os.path.join(ROOT, "data", "processed")
OUTPUT = os.path.join(ROOT, "output")

MIN_STRESS_MONTHS = 30  # below this, a regime-split VAR is under-identified
COVID_START = pd.Timestamp("2020-02-01")
COVID_END = pd.Timestamp("2021-12-01")


def group_episodes(stress_bool):
    """Group consecutive True (stress) months into (start, end, n) episodes."""
    episodes = []
    in_run = False
    start = None
    prev = None
    for date, val in stress_bool.items():
        if val and not in_run:
            in_run, start = True, date
        elif not val and in_run:
            episodes.append((start, prev))
            in_run = False
        prev = date
    if in_run:
        episodes.append((start, prev))
    # attach length in months
    out = []
    for s, e in episodes:
        n = int(stress_bool.loc[s:e].sum())
        out.append((s, e, n))
    return out


def main():
    print("=" * 78)
    print("STEP 3 — REGIME DEFINITION & VALIDATION")
    print("=" * 78)

    df = pd.read_csv(
        os.path.join(PROCESSED, "monthly_stationary.csv"),
        index_col="date", parse_dates=True,
    )
    fsi = df["STLFSI4"]
    n = len(df)
    print(f"\nSample: {df.index.min().date()} -> {df.index.max().date()}  ({n} months)")

    # ---- 2. Three candidate thresholds ----
    thr_A = 0.0
    thr_B = fsi.quantile(0.75)
    thr_C = fsi.mean() + fsi.std()
    print("\nThreshold values on STLFSI4:")
    print(f"  A: > 0                  -> {thr_A:.4f}")
    print(f"  B: > 75th percentile    -> {thr_B:.4f}")
    print(f"  C: > mean + 1 SD        -> {thr_C:.4f}  "
          f"(mean={fsi.mean():.4f}, sd={fsi.std():.4f})")

    df["stress_A"] = (fsi > thr_A).astype(int)
    df["stress_B"] = (fsi > thr_B).astype(int)
    df["stress_C"] = (fsi > thr_C).astype(int)

    thresholds = {
        "stress_A": ("STLFSI4 > 0", thr_A),
        "stress_B": ("STLFSI4 > 75th pct", thr_B),
        "stress_C": ("STLFSI4 > mean + 1 SD", thr_C),
    }

    # ---- 3 & 4. Counts, splits, episodes, sample-size flag ----
    for col, (desc, thr) in thresholds.items():
        s = df[col]
        n_stress = int(s.sum())
        n_calm = n - n_stress
        print("\n" + "-" * 78)
        print(f"{col}  ({desc}, threshold={thr:.4f})")
        print(f"  stress months: {n_stress:3d} ({100*n_stress/n:5.1f}%)   "
              f"calm months: {n_calm:3d} ({100*n_calm/n:5.1f}%)")
        if n_stress < MIN_STRESS_MONTHS:
            print(f"  !! FLAG: only {n_stress} stress months (< {MIN_STRESS_MONTHS}) "
                  f"-> likely too thin for a regime-split VAR")
        else:
            print(f"  OK: >= {MIN_STRESS_MONTHS} stress months")
        episodes = group_episodes(s.astype(bool))
        print(f"  stress episodes ({len(episodes)}):")
        for st, en, ln in episodes:
            tag = ""
            # flag overlap with COVID window
            if not (en < COVID_START or st > COVID_END):
                tag = "  <-- overlaps COVID window"
            print(f"     {st.date()} -> {en.date()}  ({ln:2d} mo){tag}")

    # ---- 6. Save ----
    out_path = os.path.join(PROCESSED, "monthly_with_regimes.csv")
    df.to_csv(out_path)
    print("\n" + "=" * 78)
    print(f"Saved -> data/processed/monthly_with_regimes.csv  shape={df.shape}")

    # ---- 5. Plot ----
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    colors = {"stress_A": "tab:red", "stress_B": "tab:purple", "stress_C": "tab:green"}
    for ax, (col, (desc, thr)) in zip(axes, thresholds.items()):
        ax.plot(fsi.index, fsi.values, lw=1.0, color="tab:blue", label="STLFSI4")
        ax.axhline(thr, color=colors[col], ls="--", lw=1.4,
                   label=f"{desc} = {thr:.3f}")
        ax.axhline(0, color="k", lw=0.5, alpha=0.3)
        # shade stress months
        stress_bool = df[col].astype(bool)
        for st, en, _ in group_episodes(stress_bool):
            # extend shading half a month each side for visibility
            ax.axvspan(st, en, color=colors[col], alpha=0.15)
        ax.axvspan(COVID_START, COVID_END, color="orange", alpha=0.10,
                   label="COVID window")
        n_stress = int(df[col].sum())
        ax.set_title(f"{col}: {desc}  —  {n_stress} stress months "
                     f"({100*n_stress/n:.0f}%)")
        ax.set_ylabel("STLFSI4")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("date")
    fig.suptitle("Stress-regime classification under three STLFSI4 thresholds",
                 y=1.005, fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, "regime_classification.png"), dpi=120)
    plt.close(fig)
    print("Saved -> output/regime_classification.png")
    print("\nDONE — regimes defined & validated. No model built.")


if __name__ == "__main__":
    main()
