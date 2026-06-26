"""
Step 6 — Power extension: M2 -> gold on gold's FULL history (no BTC).

BTC's 2014 start capped the joint sample at 140 months and excluded 2008. The
M2->gold-in-stress hint (Step 5: Granger p=0.086, cum +3.5% in stress) does not
require BTC, so we re-test it on the longest M2/GOLD/STLFSI4 window (~2000-08,
~310 months), which roughly doubles the sample and adds the 2008 GFC and 2011
EU crisis as stress episodes. Pre-committed to report whichever way it lands.

Two-variable system, order [M2_growth, GOLD_ret]. COVID dummy as exog for the
M2 break. Splits use the SAME lag-then-subset seam handling as Step 5.

Output: output/irf_gold_extended.png
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROCESSED = os.path.join(ROOT, "data", "processed")
OUTPUT = os.path.join(ROOT, "output")

ENDOG = ["M2_growth", "GOLD_ret"]
COVID_START = pd.Timestamp("2020-03-01")
COVID_END = pd.Timestamp("2022-12-01")
H = 12
ALPHA = 0.05
N_BOOT = 1000
SEED = 42
M2_IDX, GOLD_IDX = 0, 1


# ---------------------------- helpers ----------------------------
def group_episodes(stress_bool):
    eps, in_run, start, prev = [], False, None, None
    for date, val in stress_bool.items():
        if val and not in_run:
            in_run, start = True, date
        elif not val and in_run:
            eps.append((start, prev)); in_run = False
        prev = date
    if in_run:
        eps.append((start, prev))
    return [(s, e, int(stress_bool.loc[s:e].sum())) for s, e in eps]


def adf_kpss(s, name):
    s = s.dropna()
    a_stat, a_p = adfuller(s, autolag="AIC")[:2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        k_stat, k_p = kpss(s, regression="c", nlags="auto")[:2]
    print(f"  {name:11s} ADF p={a_p:.4f} ({'stat' if a_p<ALPHA else 'NON-stat'})"
          f"  | KPSS p={k_p:.4f} ({'stat' if k_p>ALPHA else 'NON-stat'})")


def make_design(df, p, use_covid):
    """Lag-p design built on the FULL ordered (contiguous) series."""
    d = df.sort_index().copy()
    cols = ["const"]
    d["const"] = 1.0
    if use_covid:
        d["covid"] = (((d.index >= COVID_START) & (d.index <= COVID_END))
                      .astype(float))
        cols.append("covid")
    lag_cols_by_lag = {}
    for i in range(1, p + 1):
        idxs = []
        for v in ENDOG:
            cname = f"L{i}.{v}"
            d[cname] = d[v].shift(i)
            cols.append(cname)
            idxs.append(cname)
        lag_cols_by_lag[i] = idxs
    return d, cols, lag_cols_by_lag


def orth_irf_p(Xmat, Ymat, lag_idx_by_lag, p, horizon=H):
    """Manual lag-p VAR -> orthogonalized IRF. Returns (irf, cum, maxeig) or None."""
    n, k = Xmat.shape
    m = Ymat.shape[1]
    if n <= k + 1:
        return None
    beta, *_ = np.linalg.lstsq(Xmat, Ymat, rcond=None)        # (k x m)
    resid = Ymat - Xmat @ beta
    Sigma = (resid.T @ resid) / max(n - k, 1)
    A = {i: beta[lag_idx_by_lag[i], :].T for i in range(1, p + 1)}  # A_i: (resp,var)
    try:
        P = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        return None
    # MA coefficients Psi_h
    Psi = [np.eye(m)]
    for h in range(1, horizon + 1):
        acc = np.zeros((m, m))
        for i in range(1, min(h, p) + 1):
            acc += A[i] @ Psi[h - i]
        Psi.append(acc)
    irf = np.array([Ph @ P for Ph in Psi])                    # (H+1, resp, imp)
    cum = np.cumsum(irf, axis=0)
    # companion eigenvalues for stability
    comp = np.zeros((m * p, m * p))
    comp[:m, :] = np.hstack([A[i] for i in range(1, p + 1)])
    if p > 1:
        comp[m:, :-m] = np.eye(m * (p - 1))
    maxeig = np.abs(np.linalg.eigvals(comp)).max()
    return irf, cum, maxeig


def fit_split(d, cols, lag_idx_by_lag, p, mask, label, use_covid):
    print("\n" + "=" * 78)
    print(f"REGIME: {label}")
    print("=" * 78)
    sub = d.loc[mask, cols + ENDOG].dropna()
    Xmat = sub[cols].to_numpy(float)
    Ymat = sub[ENDOG].to_numpy(float)
    n, k = Xmat.shape
    print(f"  effective obs: {n}  | regressors k={k}  | dof={n-k}  "
          f"| COVID exog: {'yes' if use_covid else 'NO (near-constant -> dropped)'}")
    if n < 30:
        print(f"  !! CAUTION: {n} obs — fragile inference.")
    out = orth_irf_p(Xmat, Ymat, lag_idx_by_lag, p)
    if out is None:
        print("  !! degenerate; skipping."); return None
    irf, cum, maxeig = out
    print(f"  Stability: max|eig(companion)|={maxeig:.4f} -> "
          f"{'STABLE' if maxeig<1 else 'UNSTABLE'}")

    # residual whiteness (per-equation Ljung-Box; approx on subset)
    resid = Ymat - Xmat @ np.linalg.lstsq(Xmat, Ymat, rcond=None)[0]
    lblags = min(10, max(2, n // 5))
    clean = True
    print(f"  Residual whiteness (Ljung-Box {lblags} lags, approx):")
    for i, name in enumerate(ENDOG):
        pp = float(acorr_ljungbox(resid[:, i], lags=[lblags],
                                  return_df=True)["lb_pvalue"].iloc[0])
        clean &= pp > ALPHA
        print(f"    {name:10s} p={pp:.3f} {'(clean)' if pp>ALPHA else '(autocorr!)'}")
    print(f"    -> {'CLEAN' if clean else 'some leftover autocorrelation'}")

    # Granger M2->gold: joint test of all L{i}.M2_growth coeffs = 0 in gold eq
    y = sub["GOLD_ret"].to_numpy(float)
    ols = sm.OLS(y, Xmat).fit()
    m2_lag_positions = [cols.index(f"L{i}.M2_growth") for i in range(1, p + 1)]
    R = np.zeros((p, k))
    for r, pos in enumerate(m2_lag_positions):
        R[r, pos] = 1.0
    ft = ols.f_test(R)
    gp = float(ft.pvalue); gF = float(np.ravel(ft.fvalue)[0])
    print(f"  Granger M2_growth -> GOLD_ret: F={gF:.3f}  p={gp:.4f}  -> "
          f"{'YES (sig)' if gp<ALPHA else 'no'}")

    # bootstrap bands (discard explosive)
    rng = np.random.default_rng(SEED)
    boot = np.full((N_BOOT, H + 1), np.nan)
    bootc = np.full((N_BOOT, H + 1), np.nan)
    disc = 0
    for b in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        ob = orth_irf_p(Xmat[idx], Ymat[idx], lag_idx_by_lag, p)
        if ob is None or ob[2] >= 1:
            disc += 1; continue
        boot[b] = ob[0][:, GOLD_IDX, M2_IDX]
        bootc[b] = ob[1][:, GOLD_IDX, M2_IDX]
    print(f"  bootstrap: {N_BOOT-disc}/{N_BOOT} stable ({100*disc/N_BOOT:.1f}% discarded)")
    lo = np.nanpercentile(boot, 2.5, axis=0); hi = np.nanpercentile(boot, 97.5, axis=0)
    cl = np.nanpercentile(bootc, 2.5, axis=0); ch = np.nanpercentile(bootc, 97.5, axis=0)
    gold_irf = irf[:, GOLD_IDX, M2_IDX]; gold_cum = cum[:, GOLD_IDX, M2_IDX]
    sig12 = (cl[H] > 0) or (ch[H] < 0)
    print(f"  IRF GOLD to 1-SD M2 shock: impact={gold_irf[0]:+.3f}%, "
          f"peak={gold_irf.max():+.3f}%, cum12={gold_cum[H]:+.3f}% "
          f"[95% CI {cl[H]:+.3f}, {ch[H]:+.3f}] -> "
          f"{'SIGNIFICANT' if sig12 else 'not sig (includes 0)'}")
    return dict(label=label, n=n, granger_p=gp, granger_F=gF,
                irf=gold_irf, band=(lo, hi), cum=gold_cum, cumband=(cl, ch),
                sig12=sig12)


# ---------------------------- main ----------------------------
def main():
    print("=" * 78)
    print("STEP 6 — M2 -> GOLD ON EXTENDED SAMPLE (no BTC)")
    print("=" * 78)

    levels = pd.read_csv(os.path.join(PROCESSED, "monthly_levels.csv"),
                         index_col="date", parse_dates=True)
    three = levels[["M2SL", "GOLD", "STLFSI4"]].dropna().copy()
    print(f"\nLongest common window (M2SL, GOLD, STLFSI4): "
          f"{three.index.min().date()} -> {three.index.max().date()} "
          f"({len(three)} months)")

    # transforms
    df = pd.DataFrame(index=three.index)
    df["M2_growth"] = 100 * np.log(three["M2SL"]).diff()
    df["GOLD_ret"] = 100 * np.log(three["GOLD"]).diff()
    df["STLFSI4"] = three["STLFSI4"]
    df = df.dropna()
    print(f"After differencing: {df.index.min().date()} -> {df.index.max().date()} "
          f"({len(df)} months)")

    print("\nStationarity on the LONGER window (transformed series):")
    adf_kpss(df["M2_growth"], "M2_growth")
    adf_kpss(df["GOLD_ret"], "GOLD_ret")
    print("  (M2 structural break still present in this window; kept, handled by COVID dummy.)")

    # regimes
    df["stress_A"] = (df["STLFSI4"] > 0).astype(int)
    n_stress = int(df["stress_A"].sum()); n_calm = len(df) - n_stress
    print(f"\nRegimes (stress_A = STLFSI4 > 0): stress={n_stress} ({100*n_stress/len(df):.1f}%), "
          f"calm={n_calm}")
    print(f"  vs BTC-limited sample: 36 stress months -> now {n_stress} "
          f"(+{n_stress-36})")
    eps = group_episodes(df["stress_A"].astype(bool))
    gfc = any(s <= pd.Timestamp("2009-06-01") and e >= pd.Timestamp("2008-09-01")
              for s, e, _ in eps)
    eu11 = any(s <= pd.Timestamp("2012-01-01") and e >= pd.Timestamp("2011-08-01")
               for s, e, _ in eps)
    print(f"  stress episodes ({len(eps)}):")
    for s, e, ln in eps:
        tag = ""
        if not (e < pd.Timestamp("2008-01-01") or s > pd.Timestamp("2009-12-01")):
            tag = "  <-- 2008 GFC"
        elif not (e < pd.Timestamp("2011-01-01") or s > pd.Timestamp("2012-12-01")):
            tag = "  <-- 2011 EU crisis"
        print(f"     {s.date()} -> {e.date()} ({ln:2d} mo){tag}")
    print(f"  2008 GFC captured? {'YES' if gfc else 'no'}   "
          f"2011 EU crisis captured? {'YES' if eu11 else 'no'}")

    # ---- lag selection (honor criteria) ----
    covid_full = pd.DataFrame(
        {"covid": (((df.index >= COVID_START) & (df.index <= COVID_END)).astype(float))},
        index=df.index)
    endog_df = df[ENDOG]
    sel = VAR(endog_df, exog=covid_full).select_order(maxlags=6)
    print("\n" + "-" * 78)
    print("LAG SELECTION (maxlags=6) — ample dof, honoring criteria")
    print("-" * 78)
    print(sel.summary())
    picks = {"AIC": sel.aic, "BIC": sel.bic, "HQIC": sel.hqic, "FPE": sel.fpe}
    print("Picks:", ", ".join(f"{k}={v}" for k, v in picks.items()))
    # honor criteria: use the modal pick; tie-break toward HQIC (balanced)
    from collections import Counter
    modal = Counter(picks.values()).most_common(1)[0][0]
    p = max(int(modal), 1)
    print(f">>> CHOSEN LAG = {p} (modal criterion pick)")

    # ---- (a) full extended sample: native VAR ----
    print("\n" + "=" * 78)
    print("FULL EXTENDED SAMPLE (native VAR)")
    print("=" * 78)
    res_full = VAR(endog_df, exog=covid_full).fit(p)
    print(f"  effective obs: {res_full.nobs}")
    print(f"  Stability: is_stable()={res_full.is_stable()}")
    wh = res_full.test_whiteness(nlags=max(p + 6, 10), adjusted=True)
    print(f"  Residual whiteness: p={wh.pvalue:.4f} -> "
          f"{'CLEAN' if wh.pvalue>ALPHA else 'autocorrelated'}")
    gc = res_full.test_causality("GOLD_ret", ["M2_growth"], kind="f")
    print(f"  Granger M2->GOLD: F={gc.test_statistic:.3f} p={gc.pvalue:.4f} -> "
          f"{'YES (sig)' if gc.pvalue<ALPHA else 'no'}")
    irf_full = res_full.irf(H)
    g = ENDOG.index("GOLD_ret"); mm = ENDOG.index("M2_growth")
    full_cum = irf_full.cum_effects[H, g, mm]
    print(f"  IRF GOLD to 1-SD M2 shock: impact={irf_full.orth_irfs[0,g,mm]:+.3f}%, "
          f"cum12={full_cum:+.3f}%")

    # ---- (b) splits: manual seam handling ----
    d, cols, lag_idx_names = make_design(df, p, use_covid=True)
    # convert lag column NAMES to integer positions in cols
    lag_idx_by_lag = {i: [cols.index(c) for c in names]
                      for i, names in lag_idx_names.items()}
    valid = d.dropna(subset=sum(lag_idx_names.values(), [])).index
    mask_calm = (d["stress_A"] == 0) & d.index.isin(valid)
    mask_stress = (d["stress_A"] == 1) & d.index.isin(valid)

    def covid_var(mask):
        c = d.loc[mask].dropna(subset=sum(lag_idx_names.values(), []))["covid"]
        return int((c == 1).sum()) >= 5 and int((c == 0).sum()) >= 5

    calm_covid, stress_covid = covid_var(mask_calm), covid_var(mask_stress)
    print(f"\nCOVID-dummy variation: calm={'incl' if calm_covid else 'DROP'}, "
          f"stress={'incl' if stress_covid else 'DROP'}")

    # rebuild design per regime depending on covid inclusion
    def design_for(use_covid):
        dd, cc, ln = make_design(df, p, use_covid)
        idx_by_lag = {i: [cc.index(c) for c in names] for i, names in ln.items()}
        val = dd.dropna(subset=sum(ln.values(), [])).index
        return dd, cc, idx_by_lag, val

    d_c, cols_c, idx_c, val_c = design_for(calm_covid)
    mask_calm = (d_c["stress_A"] == 0) & d_c.index.isin(val_c)
    res_calm = fit_split(d_c, cols_c, idx_c, p, mask_calm, "CALM (stress_A==0)", calm_covid)

    d_s, cols_s, idx_s, val_s = design_for(stress_covid)
    mask_stress = (d_s["stress_A"] == 1) & d_s.index.isin(val_s)
    res_stress = fit_split(d_s, cols_s, idx_s, p, mask_stress, "STRESS (stress_A==1)", stress_covid)

    # ---- THE TEST: compare + back to BTC-limited ----
    print("\n" + "=" * 78)
    print("THE TEST — M2->gold stress effect on the LONGER sample")
    print("=" * 78)
    print(f"{'':18s}{'Granger p':>12s}{'cum12 %':>12s}   95% CI")
    print(f"{'BTC-limited stress':18s}{0.086:12.3f}{3.5:12.3f}   (Step 5 reference)")
    for r in (res_calm, res_stress):
        if r is None:
            continue
        reg = r["label"].split()[0].lower()
        cl, ch = r["cumband"]
        print(f"{'extended '+reg:18s}{r['granger_p']:12.3f}{r['cum'][H]:12.3f}"
              f"   [{cl[H]:+.3f}, {ch[H]:+.3f}]")

    # ---- comparison plot ----
    months = np.arange(H + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, res, title in [(axes[0], res_calm, "GOLD_ret — CALM (extended)"),
                           (axes[1], res_stress, "GOLD_ret — STRESS (extended)")]:
        if res is None:
            ax.text(0.5, 0.5, "no estimate", ha="center"); ax.set_title(title); continue
        lo, hi = res["band"]
        ax.plot(months, res["irf"], color="tab:orange", lw=1.9, label="IRF")
        ax.fill_between(months, lo, hi, color="tab:orange", alpha=0.18, label="95% band")
        ax.axhline(0, color="k", lw=0.8, alpha=0.6)
        ax.set_title(f"{title}\nGranger p={res['granger_p']:.3f}, cum12={res['cum'][H]:+.2f}%")
        ax.set_xlabel("months after shock"); ax.set_ylabel("gold response (%)")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("M2_growth -> GOLD_ret on extended sample (2000-): calm vs stress",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, "irf_gold_extended.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved -> output/irf_gold_extended.png")
    print("\nDONE — extended-sample test complete. See summary.")


if __name__ == "__main__":
    main()
