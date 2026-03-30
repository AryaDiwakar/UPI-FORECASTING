# UPI Transaction Forecasting System

Sequence-Based Forecasting of UPI Digital Payment Transactions Using Data Science

## Overview

This is a full-stack machine learning application that forecasts UPI (Unified Payments Interface) digital payment transactions using various time-series forecasting models, including deep learning (LSTM). The system automatically scrapes data from the NPCI website and provides an interactive dashboard for visualization.

## Problem Context

UPI transaction data is available but underutilized. Traditional models fail to capture non-linear temporal patterns, and there's a need to model sequential dependencies in time-series data. This project converts the problem into a **Sequence → Value prediction task** using sliding windows.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (React)                         │
│  - Dashboard UI (Charts, Stats, Insights)                        │
│  - API Integration via Axios                                    │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP
┌───────────────────────────▼─────────────────────────────────────┐
│                      Backend (FastAPI)                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Scraper   │  │  Processor  │  │     ML Models          │  │
│  │   Service   │─▶│   Service   │─▶│ - Moving Average       │  │
│  │             │  │             │  │ - Linear Regression    │  │
│  └─────────────┘  └─────────────┘  │ - ARIMA               │  │
│                                     │ - LSTM (Deep Learning) │  │
│                                     └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **Pandas/NumPy** - Data manipulation
- **Scikit-learn** - Machine learning
- **Statsmodels** - ARIMA time-series modeling
- **TensorFlow/Keras** - LSTM deep learning model

### Frontend
- **React + TypeScript** - UI framework
- **Vite** - Build tool
- **Tailwind CSS** - Dark fintech styling
- **Recharts** - Interactive charts
- **Lucide React** - Icons

## Data Pipeline

### 1. Data Extraction (`services/scraper.py`)

The scraper attempts to fetch real data from NPCI's official website:
- URL: `https://www.npci.org.in/product/upi/product-statistics`
- Uses BeautifulSoup for HTML parsing
- Falls back to realistic generated data if scraping fails

The generated data simulates realistic UPI growth patterns:
- Base volume starting from 0.17M (April 2016)
- Exponential growth with varying rates over time
- Seasonal factors (festive boost in Oct-Dec)
- COVID-19 impact simulation (2020-2021)

### 2. Data Cleaning (`services/processor.py`)

- Converts month-year strings to datetime
- Handles missing values
- Sorts chronologically
- Computes EDA statistics (mean, std, min, max, growth rates)

### 3. Feature Engineering

Creates sequence-based features for ML models:
- **Lag features**: t-1, t-3, t-6, t-12
- **Rolling statistics**: 3, 6, 12-month moving averages and standard deviations
- **Growth metrics**: Month-over-month (MoM) and Year-over-Year (YoY) growth
- **Temporal features**: Trend, seasonal sin/cos encoding

## Forecasting Models

### 1. Moving Average
Simple baseline model using rolling mean of specified window size.

### 2. Linear Regression with Lag Features
Ridge regression using engineered features (lags, rolling stats, temporal encoding).

### 3. ARIMA
AutoRegressive Integrated Moving Average - classical time-series model with order (2,1,2).

### 4. LSTM (Long Short-Term Memory)
Deep learning model for sequence prediction:
- Input: Sliding window of 6 months
- Architecture: 2 LSTM layers with dropout
- Output: Next month's transaction volume

### 5. Prophet (Future)
Facebook's time series forecasting model (installed, implementation pending).

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | API info |
| `GET /fetch-data` | Scrape and process NPCI data |
| `GET /data` | Get time series data |
| `GET /stats` | Get EDA statistics |
| `GET /forecast` | Get model predictions |
| `GET /anomalies` | Detect outliers (Z-score) |
| `GET /insights` | Auto-generated insights |

## Frontend Dashboard

### Tabs
1. **Overview** - Stats cards, transaction trends chart, anomaly table
2. **Forecast** - 12-month predictions with model selection
3. **Models** - Model comparison with RMSE/MAE/MAPE metrics
4. **Insights** - Business insights and recommendations

### Features
- Dark fintech UI theme
- Interactive charts (Recharts)
- Real-time data refresh
- Model selection dropdown
- Responsive design

## Running the Application

### Prerequisites
- Python 3.10+
- Node.js 18+
- npm or yarn

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
python main.py
```

The backend runs on `http://localhost:8000`

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The frontend runs on `http://localhost:5173`

### Or Run Both

```bash
# Terminal 1 - Backend
cd backend && python main.py

# Terminal 2 - Frontend
cd frontend && npm run dev
```

## Project Structure

```
upi-forecasting/
├── backend/
│   ├── main.py                 # FastAPI application entry point
│   ├── requirements.txt        # Python dependencies
│   ├── data/
│   │   └── upi_data.csv       # Stored transaction data
│   └── services/
│       ├── scraper.py         # NPCI data extraction
│       ├── processor.py       # Data cleaning & feature engineering
│       ├── models.py          # ML forecasting models
│       └── insights.py        # Auto-generated insights
└── frontend/
    ├── src/
    │   ├── App.tsx            # Main dashboard component
    │   ├── main.tsx          # React entry point
    │   ├── index.css         # Tailwind styles
    │   ├── types.ts          # TypeScript interfaces
    │   ├── components/
    │   │   ├── StatCard.tsx  # Stats display cards
    │   │   ├── Charts.tsx    # Recharts visualizations
    │   │   ├── Insights.tsx  # Insights panel
    │   │   └── Anomalies.tsx # Anomaly detection table
    │   └── hooks/
    │       └── useApi.ts     # API integration hook
    ├── index.html
    ├── tailwind.config.js
    ├── postcss.config.js
    └── package.json
```

## Key Implementation Details

### Sliding Window Sequence Modeling

The LSTM model uses a sliding window approach:
```
Input: [month_i-6, month_i-5, ..., month_i-1]
Output: month_i
```

This allows the model to learn temporal dependencies over 6-month sequences.

### Model Evaluation

Models are evaluated using:
- **RMSE** (Root Mean Square Error)
- **MAE** (Mean Absolute Error)
- **MAPE** (Mean Absolute Percentage Error)

Time-based train/test split (last 12 months for testing).

### Anomaly Detection

Uses Z-score methodology:
- Z-score > 2.0: Potential anomaly (amber)
- Z-score > 2.5: Significant anomaly (red)

### Auto-Generated Insights

The system automatically generates:
- Summary statistics
- Growth trends analysis
- Seasonality patterns
- Model comparison insights
- Business recommendations

## Sample Output

### Stats
```json
{
  "total_records": 107,
  "date_range": {"start": "2016-04", "end": "2025-02"},
  "volume": {"mean": 27.66, "latest": 91.13},
  "growth_rate": {"volume_yoy": 56.85}
}
```

### Insights
- "UPI transactions show explosive growth with average YoY growth rate of 162.5%"
- "Festive season (Oct-Dec) shows 28.2% higher activity than average"
- "LSTM outperforms traditional models in capturing non-linear temporal patterns"

## Future Enhancements

- [ ] Implement actual NPCI website scraping
- [ ] Complete Prophet model implementation
- [ ] Implement more sophisticated anomaly detection
- [ ] Add model persistence (save/load)
- [ ] Implement real-time forecasting API
- [ ] Add scenario simulation features
- [ ] Export predictions to CSV

## License

MIT License

## Author

Built for sequence-based time-series forecasting demonstration.
