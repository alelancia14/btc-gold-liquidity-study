"""
Step 5 — Regime-split VAR: calm vs. stress (the core test).

Primary split: Threshold A (stress = STLFSI4 > 0), column stress_A.
Spec mirrors the full-sample VAR: endog [M2_growth, BTC_ret, GOLD_ret], lag 1,
COVID dummy as exog where it has variation.

TIME-SERIES SUBTLETY (handled explicitly):
  Splitting a monthly series into non-contiguous regime months would break the
  lag structure at the seams if we concatenated-then-lagged. Instead we build
  the lag-1 regressors on the FULL ordered (contiguous) series FIRST, so every
  month's L1 is its TRUE calendar predecessor, and only THEN subset the
  dependent rows to each regime. A stress month may legitimately be regressed on
  the preceding calm month. The only row lost is the very first sample month
  (no predecessor).

Because statsmodels VAR re-lags internally and cannot take a row mask, each
regime is estimated by OLS on the properly-lagged design (identical to a VAR
when all equations share regressors). IRFs are built from the estimated
companion matrix A and residual covariance Sigma; confidence bands come from a
PAIRS bootstrap (resample (y_t, y_{t-1}) rows), the appropriate method for a
subset-estimated VAR.

Outputs:
  - output/irf_regime_comparison.png   (calm vs stress, BTC & gold)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROCESSED = os.path.join(ROOT, "data", "processed")
OUTPUT = os.path.join(ROOT, "output")

ENDOG = ["M2_growth", "BTC_ret", "GOLD_ret"]
LAGVARS = ["L1.M2_growth", "L1.BTC_ret", "L1.GOLD_ret"]
COVID_START = pd.Timestamp("2020-03-01")
COVID_END = pd.Timestamp("2022-12-01")
H = 12
ALPHA = 0.05
N_BOOT = 1000
SEED = 42


def compute_orth_irf(Xmat, Ymat, lag_idx, horizon=H):
    """OLS -> companion A and Sigma -> orthogonalized IRFs Theta_h = A^h P.
    Returns (irf[h,resp,imp], cum[h,resp,imp], A, Sigma) or None if degenerate."""
    n, k = Xmat.shape
    if n <= k:
        return None
    beta, *_ = np.linalg.lstsq(Xmat, Ymat, rcond=None)   # (k x 3)
    resid = Ymat - Xmat @ beta
    dof = max(n - k, 1)
    Sigma = (resid.T @ resid) / dof
    A = beta[lag_idx, :].T                                # (resp i, lagvar j)
    try:
        P = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        return None
    irf = np.zeros((horizon + 1, 3, 3))
    Ah = np.eye(3)
    for h in range(horizon + 1):
        irf[h] = Ah @ P
        Ah = Ah @ A
    cum = np.cumsum(irf, axis=0)
    return irf, cum, A, Sigma


def fit_regime(df, mask, label, use_covid):
    """Fit the lag-1 regime VAR via OLS on the pre-lagged design; validate."""
    print("\n" + "=" * 78)
    print(f"REGIME: {label}")
    print("=" * 78)

    cols = ["const"] + (["covid"] if use_covid else []) + LAGVARS
    sub = df.loc[mask, cols + ENDOG].dropna()
    Xmat = sub[cols].to_numpy(float)
    Ymat = sub[ENDOG].to_numpy(float)
    n, k = Xmat.shape
    lag_idx = [cols.index(v) for v in LAGVARS]

    print(f"  effective obs (valid-lag dependent rows): {n}")
    print(f"  regressors per equation (k): {k}  -> residual dof: {n - k}")
    print(f"  COVID dummy included as exog? {'yes' if use_covid else 'NO (dropped — near-constant in this regime)'}")
    if n < 30:
        print(f"  !! CAUTION: only {n} obs — inference here is fragile; treat verdicts as suggestive.")

    out = compute_orth_irf(Xmat, Ymat, lag_idx)
    if out is None:
        print("  !! Estimation degenerate (too few obs / singular). Skipping.")
        return None
    irf, cum, A, Sigma = out

    # stability
    eig = np.abs(np.linalg.eigvals(A))
    stable = eig.max() < 1
    print(f"\n  Stability: max |eigenvalue of A| = {eig.max():.4f} -> "
          f"{'STABLE' if stable else 'UNSTABLE'}")

    # residual whiteness (per-equation Ljung-Box; approximate on a subset)
    resid = Ymat - Xmat @ np.linalg.lstsq(Xmat, Ymat, rcond=None)[0]
    lb_p = {}
    lblags = min(10, max(2, n // 5))
    for i, name in enumerate(ENDOG):
        lb = acorr_ljungbox(resid[:, i], lags=[lblags], return_df=True)
        lb_p[name] = float(lb["lb_pvalue"].iloc[0])
    clean = all(p > ALPHA for p in lb_p.values())
    print(f"  Residual whiteness (Ljung-Box, {lblags} lags, per equation; approx. on subset):")
    for name, p in lb_p.items():
        print(f"    {name:10s} p={p:.3f} {'(clean)' if p > ALPHA else '(autocorr!)'}")
    print(f"    -> {'residuals look CLEAN' if clean else 'some leftover autocorrelation'}")

    # Granger at lag 1 = test L1.M2_growth coefficient in target equation (t-test -> F=t^2)
    print("\n  Granger causality (lag 1: test L1.M2_growth coeff = 0 in target eq):")
    granger = {}
    for target in ["BTC_ret", "GOLD_ret"]:
        y = sub[target].to_numpy(float)
        ols = sm.OLS(y, Xmat).fit()
        ci = cols.index("L1.M2_growth")
        tstat = ols.tvalues[ci]
        pval = ols.pvalues[ci]
        coef = ols.params[ci]
        Fstat = tstat ** 2
        verdict = "YES (significant)" if pval < ALPHA else "no"
        granger[target] = (Fstat, pval, coef)
        print(f"    M2_growth -> {target:9s}: coef(L1.M2)={coef:+.4f}  "
              f"F={Fstat:.3f}  p={pval:.4f}  -> {verdict}")

    # pairs bootstrap for IRF bands. Discard UNSTABLE draws (max|eig(A)|>=1):
    # an explosive companion matrix makes A^h blow up and yields nonsensical
    # IRFs (standard practice, Kilian/Lutkepohl). Report the discard fraction.
    rng = np.random.default_rng(SEED)
    boot_btc = np.full((N_BOOT, H + 1), np.nan)
    boot_gold = np.full((N_BOOT, H + 1), np.nan)
    boot_btc_cum = np.full((N_BOOT, H + 1), np.nan)
    boot_gold_cum = np.full((N_BOOT, H + 1), np.nan)
    n_discarded = 0
    for b in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        ob = compute_orth_irf(Xmat[idx], Ymat[idx], lag_idx)
        if ob is None:
            n_discarded += 1
            continue
        birf, bcum, bA, _ = ob
        if np.abs(np.linalg.eigvals(bA)).max() >= 1:   # explosive draw
            n_discarded += 1
            continue
        boot_btc[b] = birf[:, 1, 0]
        boot_gold[b] = birf[:, 2, 0]
        boot_btc_cum[b] = bcum[:, 1, 0]
        boot_gold_cum[b] = bcum[:, 2, 0]
    print(f"\n  bootstrap: {N_BOOT - n_discarded}/{N_BOOT} draws stable "
          f"({100*n_discarded/N_BOOT:.1f}% discarded as explosive)")

    def bands(arr):
        lo = np.nanpercentile(arr, 2.5, axis=0)
        hi = np.nanpercentile(arr, 97.5, axis=0)
        return lo, hi

    res = {
        "label": label, "n": n, "stable": stable, "clean": clean,
        "granger": granger,
        "irf_btc": irf[:, 1, 0], "irf_gold": irf[:, 2, 0],
        "cum_btc": cum[:, 1, 0], "cum_gold": cum[:, 2, 0],
        "band_btc": bands(boot_btc), "band_gold": bands(boot_gold),
        "band_btc_cum": bands(boot_btc_cum), "band_gold_cum": bands(boot_gold_cum),
    }
    print(f"\n  IRF to 1-SD M2 shock — BTC: impact={irf[0,1,0]:+.3f}%, "
          f"peak={irf[:,1,0].max():+.3f}%, cum12={cum[H,1,0]:+.3f}%")
    print(f"  IRF to 1-SD M2 shock — GOLD: impact={irf[0,2,0]:+.3f}%, "
          f"peak={irf[:,2,0].max():+.3f}%, cum12={cum[H,2,0]:+.3f}%")
    # band-includes-zero check on cumulative 12-mo effect (significance read)
    for asset, c, bc in [("BTC", cum[H, 1, 0], res["band_btc_cum"]),
                         ("GOLD", cum[H, 2, 0], res["band_gold_cum"])]:
        lo, hi = bc[0][H], bc[1][H]
        sig = (lo > 0) or (hi < 0)
        print(f"  cum12 {asset}: {c:+.3f}% [95% CI {lo:+.3f}, {hi:+.3f}] -> "
              f"{'SIGNIFICANT (excludes 0)' if sig else 'not sig (includes 0)'}")
    return res


def build_lagged(df):
    """Build lag-1 regressors on the FULL contiguous ordered series."""
    df = df.sort_index().copy()
    # verify contiguity (monthly, no calendar gaps)
    gaps = (df.index.to_series().diff().dropna() != pd.Timedelta(days=0))  # placeholder
    for v in ENDOG:
        df[f"L1.{v}"] = df[v].shift(1)
    df["const"] = 1.0
    df["covid"] = (((df.index >= COVID_START) & (df.index <= COVID_END))
                   .astype(float))
    return df


def covid_has_variation(df, mask, min_each=5):
    sub = df.loc[mask].dropna(subset=LAGVARS)
    c = sub["covid"]
    return (int((c == 1).sum()) >= min_each) and (int((c == 0).sum()) >= min_each)


def main():
    print("=" * 78)
    print("STEP 5 — REGIME-SPLIT VAR (calm vs stress; Threshold A)")
    print("=" * 78)

    raw = pd.read_csv(
        os.path.join(PROCESSED, "monthly_with_regimes.csv"),
        index_col="date", parse_dates=True,
    )
    df = build_lagged(raw)

    # dependent rows must have a valid lag (drops only first month)
    valid = df.dropna(subset=LAGVARS).index
    n_calm_total = int((raw["stress_A"] == 0).sum())
    n_stress_total = int((raw["stress_A"] == 1).sum())
    first_month = raw.index.min()
    print(f"\nFull series: {raw.index.min().date()} -> {raw.index.max().date()} "
          f"({len(raw)} months, contiguous monthly).")
    print(f"First month {first_month.date()} has no predecessor -> dropped as a dependent obs.")
    print(f"Regime totals (all months): calm={n_calm_total}, stress={n_stress_total}")

    mask_calm = (df["stress_A"] == 0) & df.index.isin(valid)
    mask_stress = (df["stress_A"] == 1) & df.index.isin(valid)
    print(f"Effective dependent obs: calm={int(mask_calm.sum())}, "
          f"stress={int(mask_stress.sum())} (total {int(mask_calm.sum()+mask_stress.sum())})")

    # COVID dummy variation per regime
    calm_covid = covid_has_variation(df, mask_calm)
    stress_covid = covid_has_variation(df, mask_stress)
    print(f"\nCOVID-dummy variation check (need >=5 months at each of 0 and 1):")
    print(f"  calm  : {'has variation -> include' if calm_covid else 'near-constant -> DROP'}")
    print(f"  stress: {'has variation -> include' if stress_covid else 'near-constant -> DROP'}")

    res_calm = fit_regime(df, mask_calm, "CALM (stress_A==0)", calm_covid)
    res_stress = fit_regime(df, mask_stress, "STRESS (stress_A==1)", stress_covid)

    # ---- comparison table ----
    print("\n" + "=" * 78)
    print("CUMULATIVE 12-MONTH EFFECT OF 1-SD M2 SHOCK (point [95% CI])")
    print("=" * 78)
    print(f"{'asset':6s} {'regime':8s} {'cum12 %':>10s}   95% CI")
    for r in (res_calm, res_stress):
        if r is None:
            continue
        reg = "calm" if "CALM" in r["label"] else "stress"
        for asset, cum, band in [("BTC", r["cum_btc"], r["band_btc_cum"]),
                                 ("GOLD", r["cum_gold"], r["band_gold_cum"])]:
            lo, hi = band[0][H], band[1][H]
            print(f"{asset:6s} {reg:8s} {cum[H]:10.3f}   [{lo:+.3f}, {hi:+.3f}]")

    # ---- side-by-side comparison plot ----
    months = np.arange(H + 1)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    panels = [
        (0, 0, res_calm, "irf_btc", "band_btc", "BTC_ret — CALM"),
        (0, 1, res_stress, "irf_btc", "band_btc", "BTC_ret — STRESS"),
        (1, 0, res_calm, "irf_gold", "band_gold", "GOLD_ret — CALM"),
        (1, 1, res_stress, "irf_gold", "band_gold", "GOLD_ret — STRESS"),
    ]
    for r_, c_, res, key, bkey, title in panels:
        ax = axes[r_, c_]
        if res is None:
            ax.text(0.5, 0.5, "no estimate", ha="center", va="center")
            ax.set_title(title)
            continue
        y = res[key]
        lo, hi = res[bkey]
        ax.plot(months, y, color="tab:blue", lw=1.8, label="IRF")
        ax.fill_between(months, lo, hi, color="tab:blue", alpha=0.18,
                        label="95% band")
        ax.axhline(0, color="k", lw=0.8, alpha=0.6)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("months after shock")
        ax.set_ylabel("response (%)")
        ax.legend(fontsize=8)
    fig.suptitle("Response to 1-SD M2_growth shock: CALM vs STRESS (orthogonalized, 95% bands)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, "irf_regime_comparison.png"), dpi=120,
                bbox_inches="tight")
    plt.close(fig)
    print("\nSaved -> output/irf_regime_comparison.png")
    print("\nDONE — regime-split VAR estimated. See summary for the calm-vs-stress contrast.")


if __name__ == "__main__":
    main()
