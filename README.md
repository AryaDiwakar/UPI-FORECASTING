# UPI Intelligence Platform v3.0

## Research-Grade Time Series Forecasting System

A comprehensive, research-level machine learning system for forecasting UPI (Unified Payments Interface) digital payment transactions. This system implements advanced statistical analysis, multiple forecasting models, and provides interpretable insights.

---

## Research Objective

### Primary Goal
Develop a robust, statistically-validated forecasting system for UPI transaction volumes that:
- Provides accurate short-term predictions with reliable multi-step forecasting
- Offers interpretable model insights
- Enables rigorous model comparison
- Supports business decision-making

### Research Questions
1. **Which model architecture performs best for in-sample testing?** - Compare classical (ARIMA), ML (XGBoost), and deep learning (LSTM/Attention) approaches
2. **Which model produces reliable forecasts?** - ML models struggle with iterative forecasting; statistical models excel
3. **What features drive predictions?** - Quantify the importance of lag features, seasonality, and growth metrics
4. **Is the data stationary?** - Use ADF/KPSS tests to determine appropriate modeling strategies

---

## The Theory: Understanding Time Series Forecasting

### Why Time Series is Different from Standard ML

Unlike traditional ML where observations are independent, time series data has:
- **Temporal dependence**: Today's value depends on past values
- **Trend**: Long-term increase/decrease pattern
- **Seasonality**: Repeating patterns at fixed intervals (monthly, quarterly)
- **Autocorrelation**: Correlation between lagged values

This violates the i.i.d. assumption of standard ML, requiring specialized approaches.

### The Challenge: Model Training vs Forecasting

There's a critical distinction in time series:

| Aspect | Model Training/Test | Multi-Step Forecasting |
|--------|-------------------|----------------------|
| **Input** | Historical features (lags, rolling stats) | Only past values |
| **Output** | Single prediction | Multiple future steps |
| **Challenge** | Feature engineering | Recursive prediction |

**ML models (Ridge, XGBoost, RF)** are trained with full feature sets (62+ features including decomposition, Fourier terms, etc.). But for forecasting, these features don't exist yet—they must be computed from the predictions themselves, causing error accumulation.

**Statistical models (ARIMA, SARIMA)** are designed for this recursive forecasting—they naturally predict one step ahead using the model's internal state.

---

## Models: Theory & Implementation

### 1. ARIMA - AutoRegressive Integrated Moving Average

**Theory:**
ARIMA combines three components:
- **AR (p)**: Future value = weighted sum of past p values
- **I (d)**: Differencing to make data stationary
- **MA (q)**: Weighted sum of past q forecast errors

```
Y(t) = c + φ₁Y(t-1) + φ₂Y(t-2) + ... + θ₁ε(t-1) + θ₂ε(t-2) + ...
```

**Why it works:**
- Captures autocorrelation structure directly
- Differencing handles non-stationarity
- Proven methodology since 1970s (Box-Jenkins)

**Our configuration:** `ARIMA(2, 1, 2)` - 2 AR terms, 1 differencing, 2 MA terms

```python
# statsmodels implementation
model = ARIMA(y_train, order=(2, 1, 2))
fitted = model.fit()
forecast = fitted.forecast(steps=12)
```

---

### 2. SARIMA - Seasonal ARIMA

**Theory:**
Extends ARIMA by adding seasonal components:
- **AR (P)**: Seasonal autoregression
- **I (D)**: Seasonal differencing
- **MA (Q)**: Seasonal moving average
- **s**: Season length (12 for monthly data)

```
SARIMA(p,d,q)(P,D,Q)[12]
```

**Why it matters for UPI:**
- UPI shows strong 12-month seasonality (festive spike in Q4)
- Non-festive months show steady growth
- SARIMA captures both trend AND seasonality

**Our configuration:** `SARIMA(1,1,1)(1,1,1,12)`

```python
# Captures both monthly trend and annual seasonality
model = SARIMAX(y_train, order=(1,1,1), seasonal_order=(1,1,1,12))
```

**Result:** Produces forecasts with realistic seasonal patterns (Oct-Dec higher than Jan-Feb)

---

### 3. Ridge Regression

**Theory:**
Linear regression with L2 regularization to prevent overfitting:

```
Y = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ + λΣβᵢ²
```

Where λ controls regularization strength.

**Why it works:**
- Fast and interpretable
- L2 penalty shrinks coefficients toward zero
- Works well when features are informative

**Limitation for forecasting:**
- Trained on 62 engineered features
- Forecasting requires recursive feature computation
- Error accumulates over multiple steps

**Our configuration:** `alpha=1.0`

---

### 4. XGBoost - Gradient Boosting

**Theory:**
Ensemble of decision trees built sequentially:
- Each tree corrects the previous errors
- Uses gradient descent to minimize loss
- Feature importance derived from split gains

```
F₁(x) = learning_rate × tree₁(x)
F₂(x) = F₁(x) + learning_rate × tree₂(x)
...
Fₙ(x) = Σ learning_rate × treeᵢ(x)
```

**Why it excels at testing:**
- Captures non-linear relationships
- Handles feature interactions
- Robust to outliers

**Limitation for forecasting:**
- Complex feature dependencies
- Small errors cascade in recursive prediction
- Feature engineering doesn't match training features

**Our configuration:** `n_estimators=100, max_depth=6, learning_rate=0.1`

---

### 5. Random Forest

**Theory:**
Ensemble of decision trees with bagging:
- Bootstrap sampling (sampling with replacement)
- Each tree trained on different subset
- Final prediction = average (regression)

```
ŷ = (1/B) Σᵦ bᵦ(x)
```

**Why it excels at testing:**
- Reduces variance through averaging
- Handles missing features gracefully
- No overfitting due to ensemble

**Limitation for forecasting:** Same as XGBoost - feature mismatch in recursive prediction

**Our configuration:** `n_estimators=100, max_depth=10`

---

### 6. LSTM - Long Short-Term Memory

**Theory:**
Recurrent neural network with memory cells:
- **Cell state**: Long-term memory
- **Forget gate**: What to discard
- **Input gate**: What to add to memory
- **Output gate**: What to output

```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)  # Forget
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)  # Input
C_t = f_t * C_{t-1} + i_t * tanh(...)  # Cell state
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)  # Output
h_t = o_t * tanh(C_t)  # Hidden state
```

**Why it works:**
- Captures long-range dependencies
- Sequential processing
- Learns complex patterns

**Our configuration:** `sequence_length=6, epochs=50, lstm_units=50, dropout=0.2`

---

### 7. Ensemble Model

**Theory:**
Weighted combination of multiple models:

```
ŷ_ensemble = Σᵢ wᵢ · ŷᵢ
```

**Weighting strategy:** Inverse RMSE weighting
```
wᵢ = 1/RMSEᵢ / Σⱼ 1/RMSEⱼ
```

Better models get higher weight in the ensemble.

---

## Why ML Models Struggle with Forecasting

### The Feature Engineering Gap

During **training**, models receive rich features:
```python
# Training features (62 total)
['lag_1', 'lag_3', 'lag_6', 'lag_12',           # Lag features
 'rolling_mean_3', 'rolling_std_3', ...,         # Rolling stats
 'trend_component', 'seasonal_component',         # Decomposition
 'fourier_sin_12', 'fourier_cos_12', ...',      # Fourier terms
 'is_festive_season', 'is_lockdown', ...']       # Event indicators
```

During **forecasting**, we only have:
```python
# Forecasting inputs
last_values = [312.67, 305.11, 298.44, ...]  # Just 112 values
```

The model must compute all 62 features from just these values, leading to:
1. Approximate features that don't match training distribution
2. Error propagation: small prediction errors affect next step's features
3. Cascading inaccuracies over 12 steps

### Why ARIMA/SARIMA Work for Forecasting

ARIMA/SARIMA don't need external features—they predict directly from the time series structure:

```python
# ARIMA forecasting
# At each step, uses internal AR coefficients and recent predictions
forecast[t] = f(forecast[t-1], forecast[t-2], ..., errors)
```

No feature engineering required. No error propagation from mismatched features.

---

## Actual Results

### Test Set Performance (RMSE/MAPE)

| Rank | Model | RMSE | MAPE | Notes |
|------|-------|------|------|-------|
| 1 | Ridge | 0.69 | 0.2% | Best on test set |
| 2 | LSTM | 1.69 | 0.5% | Good with sequences |
| 3 | SARIMA | 5.64 | 1.6% | Captures seasonality |
| 4 | ARIMA | 9.91 | 2.8% | Trend only |
| 5 | XGBoost | 50.39 | 15.7% | Feature mismatch |
| 6 | RandomForest | 61.33 | 20.7% | Feature mismatch |

### Forecast Performance (12-Month Horizon)

| Model | First Forecast | Trend | Realistic? |
|-------|---------------|-------|------------|
| **SARIMA** | 320.5 | 320→396 (+24%) | ✅ Yes |
| ARIMA | 319.8 | 320→354 (+11%) | ⚠️ Underestimates |
| Ridge | 362.7 | 363→616 (+70%) | ❌ Explodes |
| XGBoost | 229.8 | Flat (229) | ❌ Ignores trend |
| RandomForest | 127.0 | Flat (127) | ❌ Ignores trend |
| LSTM | 63.2 | 63→82 (+30%) | ❌ Collapses |

### Key Insight

> **Test RMSE ≠ Forecast Accuracy**
> 
> Ridge has the lowest test RMSE (0.69) but produces unrealistic forecasts (370→616).
> SARIMA has higher test RMSE (5.64) but produces realistic forecasts (320→396).
> 
> **For forecasting, we use SARIMA.**

---

## Methodology

### Data Pipeline
```
Raw Data (NPCI) → Scraping → Validation → Storage (SQLite/CSV)
```

**Steps:**
- Web scraping with retries and synthetic data fallback
- Missing value imputation (linear interpolation)
- Outlier detection (Z-score, IQR methods)
- Dataset versioning for reproducibility

### Feature Engineering (62 Features)

| Category | Features |
|----------|----------|
| Lag Features | lag_1, lag_3, lag_6, lag_12 |
| Rolling Statistics | mean, std, min, max, median (3, 6, 12 month windows) |
| Growth Metrics | MoM growth, YoY growth, acceleration |
| Temporal | sin/cos encoding, quarter, year |
| Decomposition | Trend, seasonal, residual components |
| Fourier Terms | Annual seasonality harmonics |
| Event Indicators | Festive season (Q4), lockdown period |

### Model Selection Strategy

1. **Train all models** on 80% of data with full feature sets
2. **Evaluate** on held-out 20% using RMSE/MAPE
3. **For forecasting**:
   - Use SARIMA for multi-step prediction (reliable)
   - Report test metrics separately from forecast quality

### Validation Strategy

- **Time Series Cross-Validation** with walk-forward window
- **No data leakage** - always predict future from past
- **Residual Analysis** - check for patterns in errors

---

## Statistical Tests

### Stationarity Tests
| Test | Purpose | Interpretation |
|------|---------|----------------|
| **ADF** (Augmented Dickey-Fuller) | Tests for unit root | p < 0.05 → Stationary |
| **KPSS** | Tests stationarity (null = stationary) | p > 0.05 → Stationary |

### Normality Tests
| Test | Purpose |
|------|---------|
| **Shapiro-Wilk** | Tests for normal distribution |
| **Jarque-Bera** | Tests for skewness and kurtosis |

### Residual Diagnostics
| Test | Purpose |
|------|---------|
| **Ljung-Box** | Tests for autocorrelation |
| **Breusch-Pagan** | Tests for heteroscedasticity |
| **Runs Test** | Tests for randomness |

---

## Project Structure

```
upi-forecasting/
├── backend/
│   ├── data/
│   │   └── scraper.py          # Data fetching & storage
│   ├── preprocessing/
│   │   └── cleaner.py          # Data cleaning & imputation
│   ├── analysis/
│   │   └── eda.py              # Statistical analysis
│   ├── features/
│   │   └── engineering.py      # 62 engineered features
│   ├── models/
│   │   ├── forecasting.py      # All ML models
│   │   └── pipeline.py         # Orchestration
│   ├── evaluation/
│   │   └── metrics.py          # Metrics & diagnostics
│   └── interpretability/
│       └── shap_values.py       # SHAP explanations
├── app/
│   └── app.py                  # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Quick Start

### Installation
```bash
git clone https://github.com/AryaDiwakar/UPI-FORECASTING
cd UPI-FORECASTING
pip install -r requirements.txt
```

### Run
```bash
streamlit run app/app.py
```

### Programmatic Usage
```python
from backend.models.pipeline import run_forecast_pipeline

results = run_forecast_pipeline()

print(f"Best test model: {results['best_model']}")
print(f"Forecast (12 months): {results['forecast']['predictions']}")
```

---

## Sample Output

```
=== Pipeline Results ===
Best model (test): Ridge
Forecast model: SARIMA
Duration: 6.32 seconds

=== 12-Month Forecast ===
2024-12: 320.54
2025-01: 327.49
2025-02: 333.99
2025-03: 342.22
2025-04: 349.20
2025-05: 357.02
2025-06: 363.67
2025-07: 370.92
2025-08: 377.84
2025-09: 384.45
2025-10: 390.44
2025-11: 396.28

Last actual (2024-11): 312.67
```

---

## References

1. **Box, G. E. P., & Jenkins, G. M. (1976)**. *Time Series Analysis: Forecasting and Control*. Holden-Day.

2. **Hochreiter, S., & Schmidhuber, J. (1997)**. *Long Short-Term Memory*. Neural Computation, 9(8), 1735-1780.

3. **Vaswani, A., et al. (2017)**. *Attention Is All You Need*. NeurIPS.

4. **Chen, T., & Guestrin, C. (2016)**. *XGBoost: A Scalable Tree Boosting System*. KDD.

5. **Hyndman, R. J., & Athanasopoulos, G. (2021)**. *Forecasting: Principles and Practice*. OTexts.

---

## Key Takeaways

1. **Test RMSE ≠ Forecast Quality** - Models with low test RMSE may produce unrealistic forecasts due to feature mismatch

2. **SARIMA for Forecasting** - Statistical models designed for recursive prediction outperform ML models for multi-step forecasting

3. **Seasonality Matters** - UPI data shows strong 12-month seasonality; SARIMA captures both trend and seasonal patterns

4. **Feature Engineering is Double-Edged** - Rich features improve training but create a gap during forecasting

5. **Ensemble Helps Test Performance** - Combining models reduces variance, but the ensemble forecast may be dominated by statistical models

---

**Built with research rigor for time series forecasting**

*For questions or collaboration, please open an issue on GitHub.*
