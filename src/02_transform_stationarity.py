"""
Step 2 — Transformation & stationarity testing (NO model yet).

Why: M2, BTC, and gold all trend upward. Correlating their *levels* would just
compare three trends -> spurious. So we convert each to a stationary "changes"
series before any modeling. STLFSI4 is the regime classifier (not a VAR
variable), kept as a level but still tested for stationarity.

Outputs:
  - data/processed/monthly_stationary.csv  (M2_growth, BTC_ret, GOLD_ret, STLFSI4)
  - output/transformed_series.png          (transformed series over time)
  - output/acf_<series>.png                (ACF per transformed series)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROCESSED = os.path.join(ROOT, "data", "processed")
OUTPUT = os.path.join(ROOT, "output")

ALPHA = 0.05


def run_adf(series):
    """ADF: H0 = unit root (non-stationary). Stationary if p < ALPHA."""
    s = series.dropna()
    stat, pval = adfuller(s, autolag="AIC")[:2]
    return stat, pval, ("yes" if pval < ALPHA else "no")


def run_kpss(series):
    """KPSS: H0 = stationary. Stationary if p > ALPHA. (level stationarity)"""
    s = series.dropna()
    with warnings.catch_warnings():
        # statsmodels warns when the test stat is outside the p-value lookup
        # table; the returned p-value is then a bound (e.g. 0.01 or 0.10).
        warnings.simplefilter("ignore")
        stat, pval = kpss(s, regression="c", nlags="auto")[:2]
    return stat, pval, ("yes" if pval > ALPHA else "no")


def main():
    # ---- 1. Load & restrict to complete-case window ----
    levels = pd.read_csv(
        os.path.join(PROCESSED, "monthly_levels.csv"),
        index_col="date", parse_dates=True,
    )
    cc = levels.dropna().copy()  # all four series present
    print("=" * 78)
    print("STEP 2 — TRANSFORMATION & STATIONARITY TESTING")
    print("=" * 78)
    print(f"\nComplete-case window: {cc.index.min().date()} -> {cc.index.max().date()}"
          f"  ({len(cc)} months, all 4 series present)")

    # ---- 2. Transform to stationary changes ----
    out = pd.DataFrame(index=cc.index)
    out["M2_growth"] = 100 * np.log(cc["M2SL"]).diff()      # monthly log-diff, %
    out["BTC_ret"] = 100 * np.log(cc["BTC"]).diff()          # monthly log-return, %
    out["GOLD_ret"] = 100 * np.log(cc["GOLD"]).diff()        # monthly log-return, %
    out["STLFSI4"] = cc["STLFSI4"]                           # level (regime classifier)

    transformed = out.dropna()  # drop first month lost to differencing
    print(f"After differencing (1 month lost): {len(transformed)} months "
          f"({transformed.index.min().date()} -> {transformed.index.max().date()})")

    # ---- 3. Stationarity testing: levels vs changes ----
    # (label, data series) — note levels use full complete-case (cc), changes
    # use the differenced series; STLFSI4 level tested once.
    tests = [
        # raw LEVELS (expect NON-stationary for the three trending series)
        ("M2SL  (level)",  cc["M2SL"]),
        ("BTC   (level)",  cc["BTC"]),
        ("GOLD  (level)",  cc["GOLD"]),
        # transformed CHANGES (expect stationary)
        ("M2_growth (Δlog%)", transformed["M2_growth"]),
        ("BTC_ret   (Δlog%)", transformed["BTC_ret"]),
        ("GOLD_ret  (Δlog%)", transformed["GOLD_ret"]),
        # regime classifier level
        ("STLFSI4 (level)", cc["STLFSI4"]),
    ]

    rows = []
    for label, s in tests:
        a_stat, a_p, a_verdict = run_adf(s)
        k_stat, k_p, k_verdict = run_kpss(s)
        rows.append({
            "series": label,
            "ADF_stat": a_stat, "ADF_p": a_p, "ADF_stationary": a_verdict,
            "KPSS_stat": k_stat, "KPSS_p": k_p, "KPSS_stationary": k_verdict,
        })
    results = pd.DataFrame(rows)

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:8.4f}")
    print("\n" + "=" * 78)
    print("STATIONARITY RESULTS  (ADF H0=unit root: stat. if p<0.05 |"
          "  KPSS H0=stationary: stat. if p>0.05)")
    print("=" * 78)
    print(results.to_string(index=False))
    print("\nNote: KPSS p-values are bounded at [0.01, 0.10] by the lookup table;"
          " a reported 0.01 means '<=0.01' and 0.10 means '>=0.10'.")

    results.to_csv(os.path.join(PROCESSED, "stationarity_results.csv"), index=False)

    # ---- 4. Save transformed complete-case dataset ----
    transformed.to_csv(os.path.join(PROCESSED, "monthly_stationary.csv"))
    print(f"\nSaved -> data/processed/monthly_stationary.csv  shape={transformed.shape}")

    # ---- 5. Plots: transformed series over time + ACF per series ----
    change_cols = ["M2_growth", "BTC_ret", "GOLD_ret", "STLFSI4"]
    titles = {
        "M2_growth": "M2 growth (monthly Δlog × 100, %)",
        "BTC_ret": "BTC return (monthly Δlog × 100, %)",
        "GOLD_ret": "Gold return (monthly Δlog × 100, %)",
        "STLFSI4": "STLFSI4 (level — regime classifier)",
    }

    fig, axes = plt.subplots(len(change_cols), 1, figsize=(11, 11), sharex=True)
    for ax, col in zip(axes, change_cols):
        ax.plot(transformed.index, transformed[col], lw=1.0)
        ax.axhline(0, color="k", lw=0.6, alpha=0.4)
        ax.set_title(titles[col])
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, "transformed_series.png"), dpi=120)
    plt.close(fig)
    print("Saved -> output/transformed_series.png")

    for col in change_cols:
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_acf(transformed[col].dropna(), ax=ax, lags=24,
                 title=f"ACF — {col}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fname = f"acf_{col}.png"
        fig.savefig(os.path.join(OUTPUT, fname), dpi=120)
        plt.close(fig)
        print(f"Saved -> output/{fname}")

    print("\nDONE — transformation & stationarity testing complete. No model built.")


if __name__ == "__main__":
    main()
