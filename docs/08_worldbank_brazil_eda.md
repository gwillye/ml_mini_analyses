# 08 — EDA on Open Gov Data: Brazil Indicators (World Bank API)

**Headline result: Brazil GDP/capita grew from US$235 (1960) to US$10,310 (2024) — a 6.1% nominal CAGR.**

## Problem
This is an **exploratory data analysis** on open public-sector data, on-brand for
government / public-policy work. We pull three socioeconomic indicators for **Brazil** straight
from the **World Bank API** (no key, no manual download):

- **GDP per capita (current US$)** — `NY.GDP.PCAP.CD`
- **Life expectancy at birth (years)** — `SP.DYN.LE00.IN`
- **Individuals using the Internet (% of population)** — `IT.NET.USER.ZS`

The question is descriptive, not predictive: **how have these three dimensions of Brazilian
development moved over the last six decades, and at what pace?** EDA like this is the
groundwork before any modelling — and is itself the deliverable for a policy/briefing audience.

## Approach
- **Source: live World Bank REST API** (`/v2/country/BRA/indicator/<code>?format=json`). Each
  indicator is fetched, `None`/missing years dropped, and sorted chronologically — a small,
  honest ETL on a trusted official source.
- **Method: trend visualization + CAGR.** For each series we plot the full time path and compute
  the **Compound Annual Growth Rate** between first and last available year:
  `CAGR = (last/first)^(1/years) − 1`. CAGR is the right single-number summary because it
  normalizes very different series (dollars, years, percent) onto one comparable "% per year"
  scale.
- **No train/test split** — this is descriptive EDA, not a predictive model, so the validation
  is *data integrity* (drop nulls, sort, guard the divide-by-zero) rather than a hold-out.

## Results & analysis
Running `py 08_worldbank_brazil_eda.py` (also saved to
[`outputs/08_brazil_results.txt`](../outputs/08_brazil_results.txt)):

```
GDP per capita (US$):  235.3 (1960) -> 10310.5 (2024), CAGR 6.1%
Life expectancy (yrs):  53.2 (1960) ->    76.0 (2024), CAGR 0.6%
Internet users (%):       0.0 (1990) ->   84.5 (2024), CAGR nan%
```

Three side-by-side trend panels are in
[`outputs/08_brazil_indicators.png`](../outputs/08_brazil_indicators.png).

**Interpretation.** The three indicators tell a coherent development story at three very
different *tempos*:
- **GDP per capita — 6.1% CAGR (nominal).** A ~44× rise in current dollars over 64 years. The
  honest caveat: this is **nominal US$**, so a large part of that 6.1% is global and Brazilian
  **inflation plus exchange-rate swings**, not pure real growth — the plot's volatility (sharp
  dips around currency crises and the 2015 recession) shows it is *not* a smooth climb. This is
  the kind of nuance an EDA must flag before anyone quotes "6% growth."
- **Life expectancy — 0.6% CAGR.** A small *percentage* but a profound *human* change:
  **+22.8 years** (53 → 76). It demonstrates why CAGR alone misleads — a tiny annual rate on a
  bounded quantity (you can't grow lifespan like GDP) represents one of the most important
  social gains in the dataset. Read the absolute delta alongside the rate.
- **Internet users — 0% → 84.5%.** CAGR is reported as **`nan`** *by design*: the series starts
  at **exactly 0%** in 1990, and CAGR is undefined when the base is zero (`x/0`). Rather than
  fabricate a number, the code's `if vals[0] > 0` guard returns `nan` — the **correct** answer
  for an adoption-from-zero curve. The story here is the S-curve shape (near-zero through the
  1990s, steep climb in the 2000s, saturating in the 2020s), not a growth rate.

**What this EDA teaches:** **never summarize with a single statistic blindly.** The same metric
(CAGR) is informative for GDP, misleading for life expectancy, and undefined for Internet
adoption — and reporting `nan` honestly is better than inventing a placeholder.

**Limitations.** (1) Nominal, not inflation-adjusted or PPP — for real welfare comparisons use
constant-US$ or PPP indicators. (2) First/last-year CAGR ignores the volatile path in between
(crisis dips are invisible in the endpoints). (3) Three indicators are illustrative, not a full
development picture (inequality, education, emissions would round it out).

**Takeaway:** Brazil's last six decades read as **strong-but-volatile nominal income growth,
steady life-expectancy gains, and explosive internet adoption** — and the analysis's real
lesson is *choosing the right summary per series* instead of one blanket metric.

## How to run
```bash
pip install -r requirements.txt
py 08_worldbank_brazil_eda.py   # live World Bank API — no key required
```
Swap the API calls for an IBGE / gov.br / Kaggle CSV to analyze other open data with the same
plot-and-CAGR template (see top-level README).
