# 08 - EDA on Open Gov Data: Brazil Indicators (World Bank API)

Headline result: Brazil GDP/capita grew from US$235 (1960) to US$10,310 (2024), a 6.1% nominal CAGR.

## Problem

This is an exploratory data analysis on open public-sector data, which fits government and public-policy work well. We pull three socioeconomic indicators for Brazil straight from the World Bank API (no key, no manual download):

- GDP per capita (current US$), code `NY.GDP.PCAP.CD`
- Life expectancy at birth (years), code `SP.DYN.LE00.IN`
- Individuals using the Internet (% of population), code `IT.NET.USER.ZS`

The question is descriptive, not predictive. How have these three dimensions of Brazilian development moved over the last six decades, and at what pace? EDA like this is the groundwork before any modelling, and it is itself the deliverable for a policy or briefing audience.

## Approach

The source is the live World Bank REST API (`/v2/country/BRA/indicator/<code>?format=json`). Each indicator is fetched, missing or `None` years are dropped, and the series is sorted chronologically. It is a small, honest ETL on a trusted official source.

The method is trend visualization plus CAGR. For each series we plot the full time path and compute the Compound Annual Growth Rate between the first and last available year: `CAGR = (last/first)^(1/years) - 1`. CAGR is the right single-number summary because it normalizes very different series (dollars, years, percent) onto one comparable "% per year" scale.

There is no train/test split here because this is descriptive EDA, not a predictive model. The validation is data integrity (drop nulls, sort, guard the divide-by-zero) rather than a hold-out.

## Results and analysis

Running `py 08_worldbank_brazil_eda.py` (also saved to [`outputs/08_brazil_results.txt`](../outputs/08_brazil_results.txt)):

```
GDP per capita (US$):  235.3 (1960) -> 10310.5 (2024), CAGR 6.1%
Life expectancy (yrs):  53.2 (1960) ->    76.0 (2024), CAGR 0.6%
Internet users (%):       0.0 (1990) ->   84.5 (2024), CAGR nan%
```

Three side-by-side trend panels are in [`outputs/08_brazil_indicators.png`](../outputs/08_brazil_indicators.png).

The three indicators tell a coherent development story at three very different tempos.

GDP per capita comes in at a 6.1% CAGR (nominal), about a 44x rise in current dollars over 64 years. The honest caveat is that this is nominal US$, so a large part of that 6.1% is global and Brazilian inflation plus exchange-rate swings, not pure real growth. The plot's volatility (sharp dips around currency crises and the 2015 recession) shows it is not a smooth climb. This is exactly the kind of nuance an EDA has to flag before anyone quotes "6% growth."

Life expectancy shows a 0.6% CAGR. That is a small percentage but a profound human change: +22.8 years, from 53 to 76. It demonstrates why CAGR alone misleads. A tiny annual rate on a bounded quantity (you cannot grow lifespan the way you grow GDP) actually represents one of the most important social gains in the dataset. Read the absolute delta alongside the rate.

Internet users went from 0% to 84.5%. The CAGR is reported as `nan` by design: the series starts at exactly 0% in 1990, and CAGR is undefined when the base is zero (you would divide by zero). Rather than fabricate a number, the code's `if vals[0] > 0` guard returns `nan`, which is the correct answer for an adoption-from-zero curve. The real story here is the S-curve shape (near-zero through the 1990s, a steep climb in the 2000s, saturating in the 2020s), not a growth rate.

What this EDA teaches: never summarize with a single statistic blindly. The same metric (CAGR) is informative for GDP, misleading for life expectancy, and undefined for Internet adoption. Reporting `nan` honestly is better than inventing a placeholder.

A few limitations. First, these are nominal figures, not inflation-adjusted or PPP, so for real welfare comparisons you would use constant-US$ or PPP indicators. Second, first/last-year CAGR ignores the volatile path in between (the crisis dips are invisible in the endpoints). Third, three indicators are illustrative, not a full development picture (inequality, education, and emissions would round it out).

Takeaway: Brazil's last six decades read as strong-but-volatile nominal income growth, steady life-expectancy gains, and explosive internet adoption. The real lesson of the analysis is choosing the right summary per series instead of one blanket metric.

## How to run

```bash
pip install -r requirements.txt
py 08_worldbank_brazil_eda.py   # live World Bank API, no key required
```

Swap the API calls for an IBGE, gov.br, or Kaggle CSV to analyze other open data with the same plot-and-CAGR template. See the top-level README.
