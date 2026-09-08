# Does Money-Supply Growth Move Bitcoin and Gold?

I built this project to test a claim that shows up constantly in discussions about Bitcoin and gold: **if the money supply grows, should those assets rise afterward?**

Rather than assume the relationship exists, I tested it directly using monthly data on U.S. M2 growth, Bitcoin returns, gold returns, and financial stress.

The second question was whether the relationship changes during stressed markets. In other words, maybe money-supply growth matters more when investors are actively looking for inflation hedges or safe-haven assets.

## What I did

I built a monthly time-series pipeline around a **Vector Autoregression (VAR)** framework.

The analysis includes:

1. collecting M2, Bitcoin, gold, and financial-stress data;
2. testing the series for stationarity;
3. checking for structural breaks in M2;
4. estimating a full-sample VAR;
5. splitting the sample into calmer and more stressed financial regimes;
6. comparing impulse responses across those regimes;
7. extending the gold sample further back in time as a pre-planned power check;
8. testing several alternative stress thresholds;
9. exporting all results into an interactive Quarto exhibit.

## What I found

The main result is a **null result**.

At monthly frequency, I did not find reliable evidence that M2 growth predicts Bitcoin or gold returns, either in calm markets or in stressed markets.

One result initially looked more interesting: M2 growth appeared to lead gold returns during periods of financial stress. Instead of stopping there, I extended the gold sample further back in time to give the test more statistical power. The relationship weakened substantially.

That made the conclusion more convincing to me, not less. The extra evidence suggests the original relationship was more likely a sample-specific finding than a stable effect.

I kept that null result rather than trying to force the project into a stronger headline.

## Start here: where to see the outputs

The main deliverable is the interactive research exhibit:

- **[Rendered exhibit](exhibit/index.html)** — the final visual presentation of the project.
- **[Quarto source](exhibit/index.qmd)** — source file used to generate the exhibit.
- **[Generated results data](exhibit/data/results.json)** — the numerical results consumed by the exhibit.

The analysis itself is organized in `src/`, where the numbered scripts move from data collection through estimation, robustness checks, and export.

> Note: GitHub will display `exhibit/index.html` as a file rather than as a fully rendered website. For the best viewing experience, the file should be served through GitHub Pages or opened locally in a browser.

## Method

The core model is a regime-split VAR using:

- **M2 growth** — monthly log change in U.S. M2 money supply;
- **Bitcoin return** — monthly BTC return;
- **Gold return** — monthly gold return;
- **STLFSI4** — St. Louis Fed Financial Stress Index, used to define calm and stressed regimes.

I use impulse response functions to examine how BTC and gold react after an M2 innovation, then compare those responses across different stress definitions.

## Why the null result matters

This project ended up being useful because it forced me to separate an interesting story from a result that actually survives additional testing.

The idea that monetary expansion should mechanically translate into higher Bitcoin or gold prices is intuitive, but the monthly data here do not provide strong support for it. The result also highlights how easy it is to overinterpret a relationship that appears in one particular sample.

## Reproducing the project

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

for f in src/0*.py; do python "$f"; done

export QUARTO_PYTHON="$(pwd)/.venv/bin/python"
quarto render exhibit/index.qmd
```

All figures and numbers displayed in the exhibit are generated programmatically and exported to `exhibit/data/results.json`; they are not hand-entered.

## Data sources

- **M2 money supply:** FRED `M2SL`
- **Financial stress:** FRED `STLFSI4`
- **Bitcoin:** Yahoo Finance `BTC-USD`
- **Gold:** Yahoo Finance `GC=F`

## Tools

Python | pandas | NumPy | statsmodels | FRED | Yahoo Finance | Quarto

## Author

**Alessandro Lancia**  
MS Economics (Data Science), Northeastern University
