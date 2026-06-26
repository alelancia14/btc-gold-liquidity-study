# Does money-supply growth move Bitcoin and gold?

An econometric study testing whether U.S. M2 money-supply growth has a measurable
lead-effect on Bitcoin and gold returns at monthly frequency — and whether any such
relationship changes in financial-stress regimes versus calm markets.

**Headline finding (a null):** at monthly frequency, M2 growth shows no statistically
reliable effect on BTC or gold returns, in calm or stress, robust across four stress
thresholds and across a sample extended back to 2000. The most promising lead
(money → gold during stress) *weakened* under a pre-committed power test — the signature
of a chance finding, not a real effect.

## The exhibit

The final deliverable is a self-contained interactive research page:
[`exhibit/index.html`](exhibit/index.html) (rendered from
[`exhibit/index.qmd`](exhibit/index.qmd) via [Quarto](https://quarto.org)).

## Method

A regime-split vector autoregression (VAR) with orthogonalized impulse response
functions:

- **Variables:** M2 growth, BTC return, gold return (monthly log-changes); the
  St. Louis Fed Financial Stress Index (`STLFSI4`) as the regime classifier.
- **Steps** (`src/01`–`src/08`): data acquisition → stationarity testing →
  M2 structural-break diagnosis (Zivot-Andrews) → regime definition →
  full-sample VAR → regime-split VAR → power extension on gold's full history →
  four-threshold robustness → export + Quarto exhibit.

## Reproducing

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# run the pipeline in order
for f in src/0*.py; do python "$f"; done

# render the exhibit (requires the Quarto CLI)
export QUARTO_PYTHON="$(pwd)/.venv/bin/python"
quarto render exhibit/index.qmd
```

All figures and numbers on the exhibit are generated from
`exhibit/data/results.json` (written by `src/08_export_for_exhibit.py`) with a fixed
random seed — no values are hand-entered.

## Data sources

- Money supply: FRED `M2SL`
- Financial stress: FRED `STLFSI4` (St. Louis Fed Financial Stress Index)
- Bitcoin: Yahoo Finance `BTC-USD`
- Gold: Yahoo Finance `GC=F` (gold futures)
