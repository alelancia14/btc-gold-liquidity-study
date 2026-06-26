"""
Step 8 (Part 1) — Export settled results to exhibit/data/ for the Quarto page.

NO new analysis: this reproduces the exact specs/seed/bootstrap of steps 4-7 so
the exhibit reads real numbers (point IRFs + bootstrap bands, Granger p-values,
cumulative effects, and the headline "effect dies as power grows" series).

Writes:
  exhibit/data/results.json   (everything the page plots/quotes)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROCESSED = os.path.join(ROOT, "data", "processed")
EXHIBIT_DATA = os.path.join(ROOT, "exhibit", "data")
os.makedirs(EXHIBIT_DATA, exist_ok=True)

COVID_START = pd.Timestamp("2020-03-01")
COVID_END = pd.Timestamp("2022-12-01")
H = 12
ALPHA = 0.05
N_BOOT = 1000
SEED = 42


# ----------------------------- shared helpers (mirror steps 5-7) -----------------------------
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


def fit_subset(df, endog, p, mask_series, targets, m2name="M2_growth"):
    """Estimate lag-p VAR on the masked subset (seam-safe). Return point IRFs +
    bootstrap bands (per horizon) + Granger p per target + cumulative effects."""
    # decide covid inclusion on a covid-on design
    d0, cols0, ln0 = make_design(df, endog, p, True)
    valid0 = d0.dropna(subset=sum(ln0.values(), [])).index
    mask0 = mask_series & d0.index.isin(valid0)
    use_covid = covid_varies(d0, mask0, ln0)

    d, cols, ln = make_design(df, endog, p, use_covid)
    valid = d.dropna(subset=sum(ln.values(), [])).index
    mask = mask_series & d.index.isin(valid)
    sub = d.loc[mask, cols + endog].dropna()
    Xmat = sub[cols].to_numpy(float); Ymat = sub[endog].to_numpy(float)
    n, k = Xmat.shape
    lag_idx = {i: [cols.index(c) for c in ln[i]] for i in ln}
    m2idx = endog.index(m2name)

    out = orth_irf_p(Xmat, Ymat, lag_idx, p)
    if out is None or n < k + 2:
        return {"n": int(n), "degenerate": True}
    irf, cum, maxeig = out

    result = {"n": int(n), "covid": bool(use_covid), "maxeig": float(maxeig),
              "stable": bool(maxeig < 1), "assets": {}}

    # Granger per target
    granger = {}
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
        granger[tgt] = {"F": float(np.ravel(ft.fvalue)[0]), "p": float(ft.pvalue)}
    result["granger"] = granger

    # bootstrap bands (discard explosive)
    rng = np.random.default_rng(SEED)
    boot_irf = {t: np.full((N_BOOT, H + 1), np.nan) for t in targets}
    boot_cum = {t: np.full((N_BOOT, H + 1), np.nan) for t in targets}
    disc = 0
    for b in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        ob = orth_irf_p(Xmat[idx], Ymat[idx], lag_idx, p)
        if ob is None or ob[2] >= 1:
            disc += 1; continue
        for t in targets:
            ti = endog.index(t)
            boot_irf[t][b] = ob[0][:, ti, m2idx]
            boot_cum[t][b] = ob[1][:, ti, m2idx]
    result["discard_frac"] = disc / N_BOOT

    for t in targets:
        ti = endog.index(t)
        pt = irf[:, ti, m2idx]; cpt = cum[:, ti, m2idx]
        lo = np.nanpercentile(boot_irf[t], 2.5, axis=0)
        hi = np.nanpercentile(boot_irf[t], 97.5, axis=0)
        clo = np.nanpercentile(boot_cum[t], 2.5, axis=0)
        chi = np.nanpercentile(boot_cum[t], 97.5, axis=0)
        result["assets"][t] = {
            "irf": pt.tolist(), "irf_lo": lo.tolist(), "irf_hi": hi.tolist(),
            "cum": cpt.tolist(), "cum_lo": clo.tolist(), "cum_hi": chi.tolist(),
            "cum12": float(cpt[H]), "cum12_lo": float(clo[H]), "cum12_hi": float(chi[H]),
            "granger_p": granger[t]["p"],
            "sig12": bool((clo[H] > 0) or (chi[H] < 0)),
        }
    return result


def thresholds_for(fsi):
    return {
        "A: >0": 0.0,
        "B: >75th pct": float(fsi.quantile(0.75)),
        "C: >mean+1SD": float(fsi.mean() + fsi.std()),
        "D: >0.15": 0.15,
    }


# ----------------------------- build datasets -----------------------------
def load_btc_limited():
    df = pd.read_csv(os.path.join(PROCESSED, "monthly_stationary.csv"),
                     index_col="date", parse_dates=True)
    return df  # has M2_growth, BTC_ret, GOLD_ret, STLFSI4


def load_gold_extended():
    levels = pd.read_csv(os.path.join(PROCESSED, "monthly_levels.csv"),
                         index_col="date", parse_dates=True)
    three = levels[["M2SL", "GOLD", "STLFSI4"]].dropna().copy()
    df = pd.DataFrame(index=three.index)
    df["M2_growth"] = 100 * np.log(three["M2SL"]).diff()
    df["GOLD_ret"] = 100 * np.log(three["GOLD"]).diff()
    df["STLFSI4"] = three["STLFSI4"]
    return df.dropna()


def main():
    print("=" * 70)
    print("STEP 8 (Part 1) — exporting settled results to exhibit/data/results.json")
    print("=" * 70)

    out = {"meta": {}, "months": list(range(H + 1))}

    btc = load_btc_limited()
    gold = load_gold_extended()
    out["meta"] = {
        "btc_limited": {"start": str(btc.index.min().date()),
                        "end": str(btc.index.max().date()), "n": int(len(btc)),
                        "endog": ["M2_growth", "BTC_ret", "GOLD_ret"], "lag": 1},
        "gold_extended": {"start": str(gold.index.min().date()),
                          "end": str(gold.index.max().date()), "n": int(len(gold)),
                          "endog": ["M2_growth", "GOLD_ret"], "lag": 2},
        "irf_horizon": H, "n_boot": N_BOOT, "seed": SEED,
        "covid_dummy": [str(COVID_START.date()), str(COVID_END.date())],
    }

    endog3 = ["M2_growth", "BTC_ret", "GOLD_ret"]
    endog2 = ["M2_growth", "GOLD_ret"]
    all_true_btc = pd.Series(True, index=btc.index)

    # ---- Full-sample VAR (step 4) ----
    print("\n[1] Full-sample 3-var VAR (lag 1)...")
    full = fit_subset(btc, endog3, 1, all_true_btc, ["BTC_ret", "GOLD_ret"])
    out["full_sample"] = full
    print(f"    n={full['n']}, stable={full['stable']}, "
          f"Granger M2->BTC p={full['granger']['BTC_ret']['p']:.3f}, "
          f"M2->GOLD p={full['granger']['GOLD_ret']['p']:.3f}")
    print(f"    cum12 BTC={full['assets']['BTC_ret']['cum12']:+.2f}%, "
          f"GOLD={full['assets']['GOLD_ret']['cum12']:+.2f}%")

    # ---- BTC-limited regime split (step 5), threshold A ----
    print("\n[2] BTC-limited regime split (threshold A: STLFSI4>0)...")
    calm_mask = btc["STLFSI4"] <= 0
    stress_mask = btc["STLFSI4"] > 0
    out["btc_limited"] = {
        "calm": fit_subset(btc, endog3, 1, calm_mask, ["BTC_ret", "GOLD_ret"]),
        "stress": fit_subset(btc, endog3, 1, stress_mask, ["BTC_ret", "GOLD_ret"]),
    }
    for reg in ["calm", "stress"]:
        r = out["btc_limited"][reg]
        print(f"    {reg}: n={r['n']}  BTC cum12={r['assets']['BTC_ret']['cum12']:+.2f}% "
              f"(p={r['assets']['BTC_ret']['granger_p']:.3f})  "
              f"GOLD cum12={r['assets']['GOLD_ret']['cum12']:+.2f}% "
              f"(p={r['assets']['GOLD_ret']['granger_p']:.3f})")

    # ---- Extended gold split (step 6), threshold A, lag 2 ----
    print("\n[3] Extended gold-only regime split (threshold A, lag 2)...")
    calm_mask_g = gold["STLFSI4"] <= 0
    stress_mask_g = gold["STLFSI4"] > 0
    out["gold_extended"] = {
        "calm": fit_subset(gold, endog2, 2, calm_mask_g, ["GOLD_ret"]),
        "stress": fit_subset(gold, endog2, 2, stress_mask_g, ["GOLD_ret"]),
    }
    for reg in ["calm", "stress"]:
        r = out["gold_extended"][reg]
        print(f"    {reg}: n={r['n']}  GOLD cum12={r['assets']['GOLD_ret']['cum12']:+.2f}% "
              f"(p={r['assets']['GOLD_ret']['granger_p']:.3f})")

    # ---- Headline: gold-in-stress Granger p across 4 thresholds, both samples (step 7) ----
    print("\n[4] Headline 'effect dies as power grows' series...")
    thr_btc = thresholds_for(btc["STLFSI4"])
    thr_gold = thresholds_for(gold["STLFSI4"])
    labels = list(thr_btc.keys())
    head = {"thresholds": labels,
            "threshold_values": {"btc_limited": thr_btc, "gold_extended": thr_gold},
            "gold_stress_granger_p": {"btc_limited": [], "extended": []},
            "btc_stress_granger_p": {"btc_limited": []},
            "stress_n": {"btc_limited": [], "extended": []}}
    for lbl in labels:
        # BTC-limited (3-var)
        sm_btc = btc["STLFSI4"] > thr_btc[lbl]
        rb = fit_subset(btc, endog3, 1, sm_btc, ["BTC_ret", "GOLD_ret"])
        if rb.get("degenerate"):
            head["gold_stress_granger_p"]["btc_limited"].append(None)
            head["btc_stress_granger_p"]["btc_limited"].append(None)
            head["stress_n"]["btc_limited"].append(rb["n"])
        else:
            head["gold_stress_granger_p"]["btc_limited"].append(rb["assets"]["GOLD_ret"]["granger_p"])
            head["btc_stress_granger_p"]["btc_limited"].append(rb["assets"]["BTC_ret"]["granger_p"])
            head["stress_n"]["btc_limited"].append(rb["n"])
        # Extended (2-var)
        sm_g = gold["STLFSI4"] > thr_gold[lbl]
        rg = fit_subset(gold, endog2, 2, sm_g, ["GOLD_ret"])
        if rg.get("degenerate"):
            head["gold_stress_granger_p"]["extended"].append(None)
            head["stress_n"]["extended"].append(rg["n"])
        else:
            head["gold_stress_granger_p"]["extended"].append(rg["assets"]["GOLD_ret"]["granger_p"])
            head["stress_n"]["extended"].append(rg["n"])
    out["headline"] = head
    print("    threshold | BTC-limited gold-stress p | extended gold-stress p | (stress n)")
    for i, lbl in enumerate(labels):
        bl = head["gold_stress_granger_p"]["btc_limited"][i]
        ex = head["gold_stress_granger_p"]["extended"][i]
        print(f"    {lbl:14s} {bl if bl is None else f'{bl:.3f}':>10} "
              f"{ex if ex is None else f'{ex:.3f}':>22} "
              f"  (btc n={head['stress_n']['btc_limited'][i]}, ext n={head['stress_n']['extended'][i]})")

    # ---- Cumulative-effect comparison table ----
    table = []
    for asset in ["BTC_ret", "GOLD_ret"]:
        for reg in ["calm", "stress"]:
            a = out["btc_limited"][reg]["assets"][asset]
            table.append({"system": "BTC-limited (3-var)", "asset": asset, "regime": reg,
                          "cum12": a["cum12"], "ci_lo": a["cum12_lo"], "ci_hi": a["cum12_hi"],
                          "granger_p": a["granger_p"], "sig": a["sig12"]})
    for reg in ["calm", "stress"]:
        a = out["gold_extended"][reg]["assets"]["GOLD_ret"]
        table.append({"system": "Gold extended (2-var)", "asset": "GOLD_ret", "regime": reg,
                      "cum12": a["cum12"], "ci_lo": a["cum12_lo"], "ci_hi": a["cum12_hi"],
                      "granger_p": a["granger_p"], "sig": a["sig12"]})
    out["cumulative_table"] = table

    # ---- write ----
    path = os.path.join(EXHIBIT_DATA, "results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    sz = os.path.getsize(path)
    print("\n" + "=" * 70)
    print(f"WROTE {path}  ({sz/1024:.1f} KB)")
    print("Contents:")
    print("  - meta (sample windows, specs, seed)")
    print("  - full_sample: IRF curves+bands, Granger, cum12 (BTC & gold)")
    print("  - btc_limited: calm & stress IRF curves+bands, Granger, cum12 (BTC & gold)")
    print("  - gold_extended: calm & stress IRF curves+bands, Granger, cum12 (gold)")
    print("  - headline: gold-stress Granger p across 4 thresholds x 2 samples + stress_n")
    print("  - cumulative_table: 10-row calm/stress comparison")
    print("\nDONE — Part 1 export complete.")


if __name__ == "__main__":
    main()
