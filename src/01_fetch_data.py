"""
Step 1 — Project setup & data acquisition.

Pulls four monthly series (longest history available), saves each raw pull to
data/raw/, builds one merged monthly levels dataframe in data/processed/, runs
sanity checks (NO transformations), and saves a raw time-series plot per series
to output/.

Series:
  - M2SL    : M2 money supply (FRED, monthly)
  - STLFSI4 : St. Louis Fed Financial Stress Index (FRED, weekly -> monthly avg)
  - BTC-USD : Bitcoin price (yfinance, -> month-end)
  - GC=F    : Gold futures price (yfinance, -> month-end)
"""

import os
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw")
PROCESSED = os.path.join(ROOT, "data", "processed")
OUTPUT = os.path.join(ROOT, "output")
for d in (RAW, PROCESSED, OUTPUT):
    os.makedirs(d, exist_ok=True)

# Generous start so we capture the longest history each source offers.
START = "1950-01-01"


def fetch_fred_monthly():
    """M2SL is already monthly. Return a single-column monthly series."""
    m2 = web.DataReader("M2SL", "fred", START)
    m2 = m2.rename(columns={"M2SL": "M2SL"})
    m2.index.name = "date"
    return m2


def fetch_fred_stress_monthly():
    """STLFSI4 is weekly -> resample to monthly average."""
    fsi = web.DataReader("STLFSI4", "fred", START)
    fsi.index.name = "date"
    # raw (weekly) saved separately by caller; here return monthly mean
    fsi_m = fsi.resample("MS").mean()
    fsi_m = fsi_m.rename(columns={"STLFSI4": "STLFSI4"})
    fsi_m.index.name = "date"
    return fsi, fsi_m


def fetch_yf_monthend(ticker, colname):
    """Daily prices -> month-end close. Returns (raw_daily, monthly)."""
    df = yf.download(ticker, start=START, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(c) for c in tup if c != ""]).strip("_")
                      for tup in df.columns]
    # Identify the Close column robustly
    close_col = None
    for cand in ["Close", f"Close_{ticker}", "Adj Close", f"Adj Close_{ticker}"]:
        if cand in df.columns:
            close_col = cand
            break
    if close_col is None:
        close_candidates = [c for c in df.columns if "Close" in c]
        close_col = close_candidates[0]
    monthly = df[[close_col]].resample("ME").last()
    monthly.columns = [colname]
    monthly.index.name = "date"
    return df, monthly


def main():
    print("=" * 70)
    print("STEP 1 — DATA ACQUISITION (raw pulls, no transformations)")
    print("=" * 70)

    # ---- M2 ----
    print("\n[1/4] FRED M2SL (M2 money supply, monthly)...")
    m2 = fetch_fred_monthly()
    m2.to_csv(os.path.join(RAW, "M2SL_raw.csv"))

    # ---- Financial Stress ----
    print("[2/4] FRED STLFSI4 (financial stress, weekly -> monthly avg)...")
    fsi_weekly, fsi_monthly = fetch_fred_stress_monthly()
    fsi_weekly.to_csv(os.path.join(RAW, "STLFSI4_raw_weekly.csv"))
    fsi_monthly.to_csv(os.path.join(RAW, "STLFSI4_monthly.csv"))

    # ---- BTC ----
    print("[3/4] yfinance BTC-USD (-> month-end)...")
    btc_daily, btc_monthly = fetch_yf_monthend("BTC-USD", "BTC")
    btc_daily.to_csv(os.path.join(RAW, "BTC_raw_daily.csv"))
    btc_monthly.to_csv(os.path.join(RAW, "BTC_monthly.csv"))

    # ---- Gold ----
    print("[4/4] yfinance GC=F (gold futures, -> month-end)...")
    gold_daily, gold_monthly = fetch_yf_monthend("GC=F", "GOLD")
    gold_daily.to_csv(os.path.join(RAW, "GOLD_raw_daily.csv"))
    gold_monthly.to_csv(os.path.join(RAW, "GOLD_monthly.csv"))

    # ---- Normalize all monthly indices to month-start timestamps for a clean merge ----
    # FRED monthly series are stamped at month start; yfinance month-end ("ME").
    # Align everyone to period 'M' so they merge on the same calendar month.
    def to_month_period(df):
        out = df.copy()
        out.index = pd.PeriodIndex(out.index, freq="M").to_timestamp()
        out.index.name = "date"
        return out

    series = {
        "M2SL": to_month_period(m2),
        "STLFSI4": to_month_period(fsi_monthly),
        "BTC": to_month_period(btc_monthly),
        "GOLD": to_month_period(gold_monthly),
    }

    # ---- Merge (outer join so we can see each series' full span and gaps) ----
    merged = pd.concat(series.values(), axis=1).sort_index()
    merged.columns = list(series.keys())
    merged.to_csv(os.path.join(PROCESSED, "monthly_levels.csv"))

    # ================= SANITY CHECKS (no transforms) =================
    print("\n" + "=" * 70)
    print("SANITY CHECKS — per-series date range, obs count, missing")
    print("=" * 70)
    for name, df in series.items():
        s = df.iloc[:, 0].dropna()
        print(f"\n{name}:")
        print(f"  first obs : {s.index.min().date()}")
        print(f"  last  obs : {s.index.max().date()}")
        print(f"  n obs     : {len(s)}")

    print("\n" + "-" * 70)
    btc_start = series["BTC"].dropna().index.min()
    print(f">>> BINDING DATA CONSTRAINT: BTC history begins {btc_start.date()}")
    print(f">>> Any model using all four series is limited to >= {btc_start.date()}")
    print("-" * 70)

    print("\nMERGED monthly_levels.csv overview:")
    print(f"  shape          : {merged.shape}")
    print(f"  full date span : {merged.index.min().date()} -> {merged.index.max().date()}")
    print("\n  Missing values per column (over full outer-join span):")
    print(merged.isna().sum().to_string())

    # Usable sample: rows where ALL four series are present
    complete = merged.dropna()
    print("\n  Complete-case window (all 4 series present):")
    if len(complete):
        print(f"    {complete.index.min().date()} -> {complete.index.max().date()}  "
              f"({len(complete)} months)")
    else:
        print("    NONE — series do not overlap!")

    print("\n  head of merged (around BTC start):")
    print(merged.loc[btc_start:].head(6).to_string())

    # ================= PLOTS (raw levels, no transforms) =================
    print("\nSaving raw time-series plots to output/ ...")
    plot_specs = [
        ("M2SL", "M2 Money Supply (M2SL, $B)", "M2SL_raw.png"),
        ("STLFSI4", "St. Louis Fed Financial Stress Index (monthly avg)", "STLFSI4_raw.png"),
        ("BTC", "Bitcoin Price (BTC-USD, month-end)", "BTC_raw.png"),
        ("GOLD", "Gold Futures Price (GC=F, month-end)", "GOLD_raw.png"),
    ]
    for col, title, fname in plot_specs:
        s = series[col].iloc[:, 0].dropna()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(s.index, s.values, lw=1.2)
        ax.set_title(title)
        ax.set_xlabel("date")
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT, fname), dpi=120)
        plt.close(fig)
        print(f"  saved output/{fname}")

    print("\nDONE — raw data acquired. No transformations performed.")


if __name__ == "__main__":
    main()
