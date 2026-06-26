"""
Step 7 — Threshold robustness: fortify the null.

Re-run the KEY stress-regime tests under four stress thresholds:
  A: STLFSI4 > 0          (primary)
  B: STLFSI4 > 75th pct   (per-sample)
  C: STLFSI4 > mean + 1SD (per-sample)
  D: STLFSI4 > 0.15       (distinct intermediate cut, absolute like A)
on BOTH systems:
  - BTC-limited 3-var (M2_growth, BTC_ret, GOLD_ret), lag 1
  - extended gold-only 2-var (M2_growth, GOLD_ret), lag 2

Same seam handling (lags on full ordered series, then subset rows) and COVID
dummy as before. NOT an attempt to find significance — we expect corroboration
and report whatever we get.

Output: data/processed/threshold_robustness.csv
"""

import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROCESSED = os.path.join(ROOT, "data", "processed")

COVID_START = pd.Timestamp("2020-03-01")
COVID_END = pd.Timestamp("2022-12-01")
H = 12
ALPHA = 0.05
N_BOOT = 1000
SEED = 42


def make_design(df, endog, p, use_covid):
    d = df.sort_index().copy()
    cols = ["const"]; d["const"] = 1.0
    if use_covid:
        d["covid"] = (((d.index >= COVID_START) & (d.index <= COVID_END)).astype(float))
        cols.append("covid")
    lag_names = {}
    for i in range(1, p + 1):
        names = []
        for v in endog:
            cn = f"L{i}.{v}"; d[cn] = d[v].shift(i); cols.append(cn); names.append(cn)
        lag_names[i] = names
    return d, cols, lag_names


def orth_irf_p(Xmat, Ymat, lag_idx_by_lag, p, horizon=H):
    n, k = Xmat.shape; m = Ymat.shape[1]
    if n <= k + 1:
        return None
    beta, *_ = np.linalg.lstsq(Xmat, Ymat, rcond=None)
    resid = Ymat - Xmat @ beta
    Sigma = (resid.T @ resid) / max(n - k, 1)
    A = {i: beta[lag_idx_by_lag[i], :].T for i in range(1, p + 1)}
    try:
        P = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        return None
    Psi = [np.eye(m)]
    for h in range(1, horizon + 1):
        acc = np.zeros((m, m))
        for i in range(1, min(h, p) + 1):
            acc += A[i] @ Psi[h - i]
        Psi.append(acc)
    irf = np.array([Ph @ P for Ph in Psi])
    cum = np.cumsum(irf, axis=0)
    comp = np.zeros((m * p, m * p))
    comp[:m, :] = np.hstack([A[i] for i in range(1, p + 1)])
    if p > 1:
        comp[m:, :-m] = np.eye(m * (p - 1))
    maxeig = np.abs(np.linalg.eigvals(comp)).max()
    return irf, cum, maxeig


def covid_varies(d, mask, lag_names):
    sub = d.loc[mask].dropna(subset=sum(lag_names.values(), []))
    if "covid" not in sub:
        return False
    c = sub["covid"]
    return int((c == 1).sum()) >= 5 and int((c == 0).sum()) >= 5


def run_stress(df, endog, p, stress_mask_full, targets, m2name="M2_growth"):
    """Fit stress-regime lag-p VAR; return Granger p per target + cum12 (+CI)."""
    # decide covid inclusion using a covid-on design
    d0, cols0, ln0 = make_design(df, endog, p, use_covid=True)
    valid0 = d0.dropna(subset=sum(ln0.values(), [])).index
    mask0 = stress_mask_full & d0.index.isin(valid0)
    use_covid = covid_varies(d0, mask0, ln0)

    d, cols, ln = make_design(df, endog, p, use_covid)
    valid = d.dropna(subset=sum(ln.values(), [])).index
    mask = stress_mask_full & d.index.isin(valid)
    sub = d.loc[mask, cols + endog].dropna()
    Xmat = sub[cols].to_numpy(float); Ymat = sub[endog].to_numpy(float)
    n, k = Xmat.shape
    lag_idx = {i: [cols.index(c) for c in ln[i]] for i in ln}

    res = {"n": n, "covid": use_covid}
    out = orth_irf_p(Xmat, Ymat, lag_idx, p)
    if out is None or n < (k + 2):
        res["degenerate"] = True
        return res
    irf, cum, maxeig = out
    res["maxeig"] = maxeig
    m2idx = endog.index(m2name)

    # Granger + cum per target
    for tgt in targets:
        y = sub[tgt].to_numpy(float)
        ols = sm.OLS(y, Xmat).fit()
        positions = [cols.index(f"L{i}.{m2name}") for i in range(1, p + 1)]
        R = np.zeros((p, k))
        for r, pos in enumerate(positions):
            R[r, pos] = 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ft = ols.f_test(R)
        res[f"{tgt}_granger_p"] = float(ft.pvalue)
        tidx = endog.index(tgt)
        res[f"{tgt}_cum12"] = float(cum[H, tidx, m2idx])

    # bootstrap CI for cum12 (discard explosive)
    rng = np.random.default_rng(SEED)
    boot = {tgt: np.full((N_BOOT, ), np.nan) for tgt in targets}
    disc = 0
    for b in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        ob = orth_irf_p(Xmat[idx], Ymat[idx], lag_idx, p)
        if ob is None or ob[2] >= 1:
            disc += 1; continue
        for tgt in targets:
            boot[tgt][b] = ob[1][H, endog.index(tgt), m2idx]
    res["discard_frac"] = disc / N_BOOT
    for tgt in targets:
        lo = np.nanpercentile(boot[tgt], 2.5); hi = np.nanpercentile(boot[tgt], 97.5)
        res[f"{tgt}_ci"] = (lo, hi)
        res[f"{tgt}_sig"] = (lo > 0) or (hi < 0)
    return res


def thresholds_for(fsi):
    return {
        "A: >0": 0.0,
        "B: >75th pct": float(fsi.quantile(0.75)),
        "C: >mean+1SD": float(fsi.mean() + fsi.std()),
        "D: >0.15": 0.15,
    }


def main():
    print("=" * 90)
    print("STEP 7 — THRESHOLD ROBUSTNESS (fortifying the null)")
    print("=" * 90)

    rows = []

    # ---------- System 1: BTC-limited 3-var, lag 1 ----------
    s3 = pd.read_csv(os.path.join(PROCESSED, "monthly_stationary.csv"),
                     index_col="date", parse_dates=True)
    endog3 = ["M2_growth", "BTC_ret", "GOLD_ret"]
    thr3 = thresholds_for(s3["STLFSI4"])
    print(f"\nSystem 1 — BTC-limited 3-var, lag 1 ({len(s3)} months, "
          f"{s3.index.min().date()}->{s3.index.max().date()})")
    print(f"  thresholds: " + ", ".join(f"{k}={v:.3f}" for k, v in thr3.items()))
    for lbl, thr in thr3.items():
        mask = (s3["STLFSI4"] > thr)
        full_mask = pd.Series(mask.values, index=s3.index)
        r = run_stress(s3, endog3, 1, full_mask, ["BTC_ret", "GOLD_ret"])
        if r.get("degenerate"):
            print(f"  [{lbl}] stress n={r['n']}  -> DEGENERATE (too few obs), skipped")
            rows.append(dict(system="BTC-3var", threshold=lbl, stress_n=r["n"],
                             note="degenerate"))
            continue
        for tgt in ["BTC_ret", "GOLD_ret"]:
            rows.append(dict(
                system="BTC-3var", threshold=lbl, stress_n=r["n"],
                covid=r["covid"], target=tgt,
                granger_p=r[f"{tgt}_granger_p"], cum12=r[f"{tgt}_cum12"],
                ci_lo=r[f"{tgt}_ci"][0], ci_hi=r[f"{tgt}_ci"][1],
                sig=r[f"{tgt}_sig"], discard=r["discard_frac"]))

    # ---------- System 2: extended gold-only 2-var, lag 2 ----------
    levels = pd.read_csv(os.path.join(PROCESSED, "monthly_levels.csv"),
                         index_col="date", parse_dates=True)
    three = levels[["M2SL", "GOLD", "STLFSI4"]].dropna().copy()
    s2 = pd.DataFrame(index=three.index)
    s2["M2_growth"] = 100 * np.log(three["M2SL"]).diff()
    s2["GOLD_ret"] = 100 * np.log(three["GOLD"]).diff()
    s2["STLFSI4"] = three["STLFSI4"]
    s2 = s2.dropna()
    endog2 = ["M2_growth", "GOLD_ret"]
    thr2 = thresholds_for(s2["STLFSI4"])
    print(f"\nSystem 2 — extended gold-only 2-var, lag 2 ({len(s2)} months, "
          f"{s2.index.min().date()}->{s2.index.max().date()})")
    print(f"  thresholds: " + ", ".join(f"{k}={v:.3f}" for k, v in thr2.items()))
    for lbl, thr in thr2.items():
        mask = (s2["STLFSI4"] > thr)
        full_mask = pd.Series(mask.values, index=s2.index)
        r = run_stress(s2, endog2, 2, full_mask, ["GOLD_ret"])
        if r.get("degenerate"):
            print(f"  [{lbl}] stress n={r['n']}  -> DEGENERATE, skipped")
            rows.append(dict(system="GOLD-2var-ext", threshold=lbl, stress_n=r["n"],
                             note="degenerate"))
            continue
        rows.append(dict(
            system="GOLD-2var-ext", threshold=lbl, stress_n=r["n"],
            covid=r["covid"], target="GOLD_ret",
            granger_p=r["GOLD_ret_granger_p"], cum12=r["GOLD_ret_cum12"],
            ci_lo=r["GOLD_ret_ci"][0], ci_hi=r["GOLD_ret_ci"][1],
            sig=r["GOLD_ret_sig"], discard=r["discard_frac"]))

    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(PROCESSED, "threshold_robustness.csv"), index=False)

    # ---------- compact printed tables ----------
    pd.set_option("display.width", 200)
    print("\n" + "=" * 90)
    print("RESULTS — stress-regime Granger p and cumulative 12-mo effect, by threshold")
    print("=" * 90)
    show = tab[tab.get("note").isna()] if "note" in tab else tab
    for system in ["BTC-3var", "GOLD-2var-ext"]:
        sub = show[show["system"] == system]
        if sub.empty:
            continue
        print(f"\n{system}:")
        print(f"  {'threshold':14s} {'stress_n':>8s} {'target':10s} {'Granger_p':>10s}"
              f" {'cum12%':>9s}  {'95% CI':>20s}  sig?")
        for _, r in sub.iterrows():
            ci = f"[{r['ci_lo']:+.2f},{r['ci_hi']:+.2f}]"
            print(f"  {r['threshold']:14s} {int(r['stress_n']):8d} {r['target']:10s}"
                  f" {r['granger_p']:10.3f} {r['cum12']:9.2f}  {ci:>20s}"
                  f"  {'YES*' if r['sig'] else 'no'}")

    # ---------- verdict ----------
    sig_rows = show[show["sig"] == True]
    n_tests = len(show)
    print("\n" + "=" * 90)
    print("VERDICT")
    print("=" * 90)
    print(f"Total stress-regime tests run (system x threshold x target): {n_tests}")
    print(f"Significant at 5% (cum12 CI excludes 0): {len(sig_rows)}")
    minp = show["granger_p"].min()
    print(f"Smallest Granger p-value across ALL cells: {minp:.3f}")
    if len(sig_rows) == 0 and minp >= ALPHA:
        print("-> NULL IS STABLE across all four thresholds and both systems. "
              "No cell reaches significance.")
    else:
        print("-> At least one cell is significant — see flagged rows; interpret as a "
              "likely multiple-comparisons false positive, not a finding.")
    print("\nDONE — threshold robustness complete.")


if __name__ == "__main__":
    main()
