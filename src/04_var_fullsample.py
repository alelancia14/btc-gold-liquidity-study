"""
Step 4 — Full-sample VAR (all 140 months). NO regime split yet.

Baseline VAR before splitting into thin sub-samples: estimate one model on the
full sample to get a baseline answer and confirm it is statistically sound.

Endogenous (ordered for later Cholesky ID): M2_growth, BTC_ret, GOLD_ret.
  Ordering rationale: M2_growth is the most "upstream"/slowest-moving (money
  supply), so it is placed first. Ordering robustness is tested in a later step.
Exogenous: COVID dummy (2020-03..2022-12) to absorb the unprecedented 2020-22
  M2 swings (the 2022-01 structural break) so they don't distort coefficients.

Outputs:
  - output/irf_fullsample.png             (orthogonalized IRFs, 12 mo, w/ bands)
  - output/irf_cumulative_fullsample.png  (cumulative IRFs)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROCESSED = os.path.join(ROOT, "data", "processed")
OUTPUT = os.path.join(ROOT, "output")

ENDOG = ["M2_growth", "BTC_ret", "GOLD_ret"]
COVID_DUMMY_START = pd.Timestamp("2020-03-01")
COVID_DUMMY_END = pd.Timestamp("2022-12-01")
IRF_HORIZON = 12
ALPHA = 0.05


def main():
    print("=" * 78)
    print("STEP 4 — FULL-SAMPLE VAR (140 months, no regime split)")
    print("=" * 78)

    df = pd.read_csv(
        os.path.join(PROCESSED, "monthly_with_regimes.csv"),
        index_col="date", parse_dates=True,
    )
    endog = df[ENDOG]
    print(f"\nSample: {endog.index.min().date()} -> {endog.index.max().date()}  "
          f"({len(endog)} months)")
    print(f"Endogenous (order): {ENDOG}")

    # ---- 2. Exogenous COVID dummy ----
    covid = ((df.index >= COVID_DUMMY_START) & (df.index <= COVID_DUMMY_END)).astype(int)
    exog = pd.DataFrame({"covid": covid}, index=df.index)
    print(f"COVID dummy = 1 for {COVID_DUMMY_START.date()}..{COVID_DUMMY_END.date()} "
          f"-> {int(exog['covid'].sum())} months flagged")

    # ---- 3. Lag selection ----
    model = VAR(endog, exog=exog)
    sel = model.select_order(maxlags=6)
    print("\n" + "-" * 78)
    print("LAG SELECTION (maxlags=6)")
    print("-" * 78)
    print(sel.summary())
    picks = {"AIC": sel.aic, "BIC": sel.bic, "HQIC": sel.hqic, "FPE": sel.fpe}
    print("\nCriterion picks:", ", ".join(f"{k}={v}" for k, v in picks.items()))

    # Parsimony bias: given n=140 with 3 vars + dummy, prefer the smaller lag
    # when criteria disagree. BIC/HQIC penalize complexity more, so lean on them.
    chosen_lag = min(sel.bic, sel.hqic)
    chosen_lag = max(chosen_lag, 1)  # need at least 1 lag for dynamics
    print(f"\n>>> CHOSEN LAG = {chosen_lag}  "
          f"(parsimony bias; using min of BIC={sel.bic}, HQIC={sel.hqic})")

    # ---- 4. Fit ----
    res = model.fit(chosen_lag)
    print("\n" + "=" * 78)
    print("VAR SUMMARY")
    print("=" * 78)
    print(res.summary())

    # ---- 5. Validation ----
    print("\n" + "=" * 78)
    print("MODEL VALIDATION")
    print("=" * 78)
    stable = res.is_stable()
    roots = res.roots
    max_mod = np.max(np.abs(1.0 / roots)) if len(roots) else np.nan
    print(f"\nStability: is_stable() = {stable}")
    print(f"  All roots inside unit circle? -> "
          f"{'YES — stable' if stable else 'NO — UNSTABLE'}")
    print(f"  (max modulus of reciprocal roots = {max_mod:.4f}; must be < 1)")

    # Residual whiteness (Portmanteau / Ljung-Box style). nlags must exceed order.
    lb_lags = max(chosen_lag + 6, 10)
    wh = res.test_whiteness(nlags=lb_lags, adjusted=True)
    print(f"\nResidual autocorrelation (Portmanteau, nlags={lb_lags}, adjusted):")
    print(f"  test statistic : {wh.test_statistic:.4f}")
    print(f"  p-value        : {wh.pvalue:.4f}")
    if wh.pvalue > ALPHA:
        print("  -> p > 0.05: FAIL to reject white noise => residuals look CLEAN "
              "(no leftover autocorrelation).")
    else:
        print("  -> p < 0.05: reject white noise => residuals STILL autocorrelated "
              "(model may be under-specified).")

    # ---- 6. Granger causality (the headline read) ----
    print("\n" + "=" * 78)
    print("GRANGER CAUSALITY — does M2_growth lead BTC / gold returns?")
    print("=" * 78)
    for target in ["BTC_ret", "GOLD_ret"]:
        gc = res.test_causality(target, ["M2_growth"], kind="f")
        verdict = ("YES — M2_growth Granger-causes it" if gc.pvalue < ALPHA
                   else "no — cannot reject 'no causality'")
        print(f"\n  M2_growth -> {target}:")
        print(f"    F-stat = {gc.test_statistic:.4f}, p-value = {gc.pvalue:.4f}")
        print(f"    verdict: {verdict}")

    # ---- 7. IRFs ----
    print("\n" + "=" * 78)
    print("IMPULSE RESPONSES (orthogonalized, 1-SD M2_growth shock, 12 mo)")
    print("=" * 78)
    irf = res.irf(IRF_HORIZON)

    # (a) IRF of BTC_ret and GOLD_ret to M2_growth shock, with bands
    fig = irf.plot(impulse="M2_growth", response=None, orth=True,
                   signif=ALPHA)
    fig.suptitle("Orthogonalized IRF to 1-SD M2_growth shock (full sample, 95% bands)",
                 y=1.02, fontsize=12)
    fig.savefig(os.path.join(OUTPUT, "irf_fullsample.png"), dpi=120,
                bbox_inches="tight")
    plt.close(fig)
    print("Saved -> output/irf_fullsample.png")

    # (b) cumulative IRF
    figc = irf.plot_cum_effects(impulse="M2_growth", response=None, orth=True,
                                signif=ALPHA)
    figc.suptitle("Cumulative orthogonalized IRF to 1-SD M2_growth shock (full sample)",
                  y=1.02, fontsize=12)
    figc.savefig(os.path.join(OUTPUT, "irf_cumulative_fullsample.png"), dpi=120,
                 bbox_inches="tight")
    plt.close(figc)
    print("Saved -> output/irf_cumulative_fullsample.png")

    # Numeric readout of the IRF point estimates for the summary
    m2_idx = ENDOG.index("M2_growth")
    btc_idx = ENDOG.index("BTC_ret")
    gold_idx = ENDOG.index("GOLD_ret")
    orth = irf.orth_irfs  # shape (horizon+1, neqs, neqs): [h, response, impulse]
    print("\nOrthogonalized IRF point estimates (response to 1-SD M2_growth shock):")
    print("  month   BTC_ret(%)   GOLD_ret(%)")
    for h in range(IRF_HORIZON + 1):
        print(f"   {h:3d}    {orth[h, btc_idx, m2_idx]:9.4f}   "
              f"{orth[h, gold_idx, m2_idx]:9.4f}")
    cum = irf.cum_effects
    print("\nCumulative effect over 12 months (sum of responses):")
    print(f"  BTC_ret : {cum[IRF_HORIZON, btc_idx, m2_idx]:.4f} %")
    print(f"  GOLD_ret: {cum[IRF_HORIZON, gold_idx, m2_idx]:.4f} %")

    print("\nDONE — full-sample VAR estimated & validated. No regime split yet.")


if __name__ == "__main__":
    main()
