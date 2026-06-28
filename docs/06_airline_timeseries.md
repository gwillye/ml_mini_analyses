# 06 — Time-Series Forecasting: Airline Passengers (Fourier-feature regression)

**Headline result: forecast MAPE = 11.6% on a held-out 20% tail (last ~29 months).**

## Problem
The **AirPassengers** series is the classic Box-Jenkins benchmark: **144 monthly totals** of
international airline passengers, Jan 1949 – Dec 1960. It has two textbook properties at once:
a strong **upward trend** and **seasonality that grows with the level** (the yearly peaks get
taller over time — *multiplicative* seasonality).

The question: **can a simple, transparent feature-engineering model forecast the future of a
trending, seasonal series**, without reaching for a full ARIMA/SARIMA or a neural model? This
is the "do you actually need the heavy machinery?" question that comes up constantly in
practical forecasting.

## Approach
- **Log transform first (the key move).** Seasonal swings here scale with the trend
  (multiplicative). Taking `log(passengers)` converts that into *additive* seasonality of
  roughly constant amplitude — which a linear model can fit cleanly. Predictions are
  exponentiated back to passenger counts.
- **Fourier seasonal features.** Instead of 11 month-dummies, we encode the annual cycle with
  two harmonics — sin/cos at period 12 *and* period 6 — plus a linear **time trend**. Four
  Fourier terms capture the smooth seasonal shape (including the secondary mid-year bump) with
  far fewer parameters than dummies, which matters on only 144 points.
- **Model: plain Linear Regression** on `[trend, sin12, cos12, sin6, cos6]`. Deliberately the
  simplest thing that could work, to test the hypothesis that good *features* beat a complex
  model.
- **Validation: a true forecast hold-out.** We train on the first 80% (chronological) and
  forecast the final 20% — **no shuffling**, because shuffling a time series leaks the future
  into training and is the most common forecasting mistake. Metric: **MAPE** (mean absolute
  percentage error), the scale-free, business-readable accuracy of a forecast.

## Results & analysis
Running `py 06_airline_timeseries.py`:

```
n=144 MAPE=11.6%
```

The forecast plot (train / actual / dashed forecast, with the split marked) is in
[`outputs/06_airline_forecast.png`](../outputs/06_airline_forecast.png): the dashed forecast
**tracks the seasonal peaks-and-troughs of the held-out tail** and follows the rising trend,
sitting slightly *below* the true values at the very end.

**Interpretation.** **MAPE = 11.6%** means forecasts on completely unseen future months are
on average within ~12% of actual — solid for a 5-feature linear model with zero
autoregressive terms. The *shape* is captured almost perfectly: the model nails *when* the
peaks and dips occur (the log + Fourier machinery is doing its job on the seasonality).

The residual ~12% error is structured and informative, not random: the forecast drifts
**slightly low at the horizon**. That is the expected signature of a *linear* trend
extrapolated onto a series whose growth is in truth gently **accelerating** — a straight trend
line in log-space cannot bend upward fast enough, so it lags the most recent, highest points.
This is exactly the kind of bias you want to *see and name* rather than hide.

**What worked:** the log transform is load-bearing. Without it, a single linear model cannot
represent seasonality that grows with the level, and MAPE would balloon. Most of the win here
is preprocessing, not modelling.

**Limitations.** (1) No autoregressive/error-correction component — a SARIMA or exponential
smoothing model would likely shave the end-of-series lag by adapting to recent residuals.
(2) The forecast is deterministic; there is no prediction interval, which a real planning use
case would need. (3) 144 points and a fixed trend mean long-horizon forecasts will increasingly
under-shoot.

**Takeaway:** **`log` + Fourier features + linear regression** is a remarkably strong, fully
interpretable forecasting baseline (~12% MAPE) — reach for SARIMA only once you've measured how
far this gets you.

## How to run
```bash
pip install -r requirements.txt
py 06_airline_timeseries.py   # downloads the CSV over HTTPS — no key required
```
Point the `URL`/loader at any Kaggle time-series CSV with `date,value` columns to reuse the
pipeline (see top-level README).
