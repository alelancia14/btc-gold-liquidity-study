"""
Step 2b — Resolve the M2_growth stationarity ambiguity.

Step 2 found M2_growth non-stationary by ADF (p=0.162) but stationary by KPSS.
Hypothesis: this is NOT a true unit root but a structural break from COVID (M2
growth spiked 2020-21, then contracted 2022-23). A single break is known to
bias the standard ADF toward a false unit-root finding. The Zivot-Andrews test
allows ONE endogenous break under the alternative, so it can distinguish
"unit root" from "stationary around a break".

This step ONLY investigates. No VAR. No second-differencing.

Outputs:
  - output/m2_growth_break.png   (M2_growth with ZA break date + COVID window)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, zivot_andrews

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROCESSED = os.path.join(ROOT, "data", "processed")
OUTPUT = os.path.join(ROOT, "output")

ALPHA = 0.05
COVID_START = pd.Timestamp("2020-02-01")
COVID_END = pd.Timestamp("2021-12-01")


def run_zivot_andrews(series, regression, label):
    """
    Zivot-Andrews: H0 = unit root (no break); H1 = stationary with ONE break.
    Reject unit root (=> stationary-with-break) if stat < 5% crit value.
    Returns a dict incl. the estimated break DATE mapped from the break index.
    """
    s = series.dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # statsmodels returns: zastat, pvalue, cvdict, baselag, bpidx
        zastat, pval, cvdict, baselag, bpidx = zivot_andrews(
            s.values, regression=regression, autolag="AIC"
        )
    break_date = s.index[bpidx]
    crit5 = cvdict["5%"]
    reject = zastat < crit5  # more negative than 5% critical value
    print(f"\n--- Zivot-Andrews ({label}, regression='{regression}') ---")
    print(f"  test statistic   : {zastat:8.4f}")
    print(f"  p-value          : {pval:8.4f}")
    print(f"  critical values  : 1%={cvdict['1%']:.3f}  5%={cvdict['5%']:.3f}  "
          f"10%={cvdict['10%']:.3f}")
    print(f"  chosen lag        : {baselag}")
    print(f"  estimated break  : {break_date.date()}  (obs index {bpidx})")
    print(f"  unit root rejected (stat<5% crit, p<0.05)? : "
          f"{'YES' if reject else 'no'}")
    return {
        "regression": regression, "label": label,
        "stat": zastat, "pval": pval, "crit5": crit5,
        "break_date": break_date, "reject": reject,
    }


def main():
    print("=" * 78)
    print("STEP 2b — M2_growth STRUCTURAL-BREAK DIAGNOSTIC")
    print("=" * 78)

    df = pd.read_csv(
        os.path.join(PROCESSED, "monthly_stationary.csv"),
        index_col="date", parse_dates=True,
    )
    m2 = df["M2_growth"].dropna()
    print(f"\nM2_growth: {m2.index.min().date()} -> {m2.index.max().date()}  "
          f"({len(m2)} months)")

    # ---- 2. Zivot-Andrews: intercept break, and intercept+trend break ----
    za_c = run_zivot_andrews(m2, "c", "break in intercept")
    za_ct = run_zivot_andrews(m2, "ct", "break in intercept + trend")

    # ---- 4. Pre-COVID ADF corroboration ----
    pre = m2.loc["2014-10-01":"2020-01-01"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf_stat, adf_p = adfuller(pre, autolag="AIC")[:2]
    print("\n--- Pre-COVID ADF (2014-10 -> 2020-01) ---")
    print(f"  n obs            : {len(pre)}  (short sample -> suggestive only)")
    print(f"  ADF statistic    : {adf_stat:8.4f}")
    print(f"  ADF p-value      : {adf_p:8.4f}")
    print(f"  stationary (p<0.05)? : {'YES' if adf_p < ALPHA else 'no'}")

    # ---- 3. Corroboration plot ----
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(m2.index, m2.values, lw=1.1, color="tab:blue", label="M2_growth (%)")
    ax.axhline(0, color="k", lw=0.6, alpha=0.4)
    ax.axvspan(COVID_START, COVID_END, color="orange", alpha=0.18,
               label="COVID window (2020-02 to 2021-12)")
    ax.axvline(za_c["break_date"], color="red", ls="--", lw=1.5,
               label=f"ZA break (intercept): {za_c['break_date'].date()}")
    ax.axvline(za_ct["break_date"], color="darkgreen", ls=":", lw=1.5,
               label=f"ZA break (intercept+trend): {za_ct['break_date'].date()}")
    ax.set_title("M2_growth with Zivot-Andrews estimated break(s) vs COVID window")
    ax.set_ylabel("M2 growth (monthly Δlog × 100, %)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, "m2_growth_break.png"), dpi=120)
    plt.close(fig)
    print("\nSaved -> output/m2_growth_break.png")

    # ---- 5. Compact verdict table ----
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for r in (za_c, za_ct):
        print(f"  ZA {r['label']:28s}: stat={r['stat']:7.3f}  5%crit={r['crit5']:.3f}"
              f"  break={r['break_date'].date()}  "
              f"unit-root-rejected={'YES' if r['reject'] else 'no'}")
    print(f"  Pre-COVID ADF                   : stat={adf_stat:7.3f}  "
          f"p={adf_p:.4f}  stationary={'YES' if adf_p < ALPHA else 'no'}")
    print("\nDONE — diagnostic only. No model built, no second-differencing.")


if __name__ == "__main__":
    main()
