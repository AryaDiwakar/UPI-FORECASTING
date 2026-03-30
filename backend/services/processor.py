import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
import os
import json
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)


class DataProcessor:
    def __init__(self, df: pd.DataFrame, use_cache: bool = True):
        self.raw_df = df.copy()
        self.df = df.copy()
        self.use_cache = use_cache
        self._cache = {}
        
    def clean_data(self) -> pd.DataFrame:
        df = self.df.copy()
        
        if 'month' in df.columns:
            df['month'] = df['month'].astype(str).str.strip()
            
            month_map = {
                'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
                'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
                'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
            }
            
            def parse_month(month_str):
                parts = month_str.lower().replace('-', ' ').split()
                if len(parts) >= 2:
                    month_name = parts[0][:3]
                    year = parts[1]
                    month_num = month_map.get(month_name, 1)
                    year_full = int(year) + 2000 if int(year) < 50 else int(year) + 1900
                    return pd.Timestamp(year=year_full, month=month_num, day=1)
                return pd.NaT
            
            df['date'] = df['month'].apply(parse_month)
            df = df.dropna(subset=['date'])
            df = df.sort_values('date').reset_index(drop=True)
        
        numeric_cols = ['volume_millions', 'value_crores']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        
        self.df = df
        return df
    
    def get_eda_stats(self) -> Dict[str, Any]:
        df = self.df
        
        stats = {
            'total_records': len(df),
            'date_range': {
                'start': df['date'].min().strftime('%Y-%m') if not df.empty else None,
                'end': df['date'].max().strftime('%Y-%m') if not df.empty else None
            },
            'volume': {
                'mean': round(df['volume_millions'].mean(), 2),
                'std': round(df['volume_millions'].std(), 2),
                'min': round(df['volume_millions'].min(), 2),
                'max': round(df['volume_millions'].max(), 2),
                'latest': round(df['volume_millions'].iloc[-1], 2) if not df.empty else None,
                'median': round(df['volume_millions'].median(), 2),
                'q1': round(df['volume_millions'].quantile(0.25), 2),
                'q3': round(df['volume_millions'].quantile(0.75), 2)
            },
            'value': {
                'mean': round(df['value_crores'].mean(), 2),
                'std': round(df['value_crores'].std(), 2),
                'min': round(df['value_crores'].min(), 2),
                'max': round(df['value_crores'].max(), 2),
                'latest': round(df['value_crores'].iloc[-1], 2) if not df.empty else None
            },
            'growth_rate': {
                'volume_yoy': round(self._calculate_yoy_growth(df['volume_millions']), 2),
                'value_yoy': round(self._calculate_yoy_growth(df['value_crores']), 2),
                'volume_mom': round(self._calculate_mom_growth(df['volume_millions']), 2),
                'value_mom': round(self._calculate_mom_growth(df['value_crores']), 2),
                'cagr': round(self._calculate_cagr(df['volume_millions']), 2)
            },
            'volatility': {
                'volume_cv': round(df['volume_millions'].std() / df['volume_millions'].mean() * 100, 2),
                'volume_monthly_std': round(df['volume_millions'].diff().std(), 2)
            }
        }
        
        return stats
    
    def _calculate_yoy_growth(self, series: pd.Series) -> float:
        if len(series) >= 12:
            return ((series.iloc[-1] - series.iloc[-13]) / series.iloc[-13]) * 100
        return 0.0
    
    def _calculate_mom_growth(self, series: pd.Series) -> float:
        if len(series) >= 2:
            return ((series.iloc[-1] - series.iloc[-2]) / series.iloc[-2]) * 100
        return 0.0
    
    def _calculate_cagr(self, series: pd.Series, periods: int = 12) -> float:
        if len(series) >= periods * 2:
            start_val = series.iloc[-periods * 2]
            end_val = series.iloc[-1]
            n_years = 1
            if start_val > 0:
                return ((end_val / start_val) ** (1 / n_years) - 1) * 100
        return 0.0
    
    def create_features(self, target_col: str = 'volume_millions', 
                        lags: List[int] = [1, 3, 6, 12],
                        windows: List[int] = [3, 6, 12]) -> pd.DataFrame:
        cache_key = f"features_{target_col}_{'_'.join(map(str, lags))}_{'_'.join(map(str, windows))}"
        
        if self.use_cache and cache_key in self._cache:
            self.featured_df = self._cache[cache_key]
            return self.featured_df
        
        df = self.df.copy()
        
        for lag in lags:
            df[f'lag_{lag}'] = df[target_col].shift(lag)
        
        for window in windows:
            df[f'rolling_mean_{window}'] = df[target_col].shift(1).rolling(window=window).mean()
            df[f'rolling_std_{window}'] = df[target_col].shift(1).rolling(window=window).std()
            df[f'rolling_min_{window}'] = df[target_col].shift(1).rolling(window=window).min()
            df[f'rolling_max_{window}'] = df[target_col].shift(1).rolling(window=window).max()
        
        df['mom_growth'] = df[target_col].pct_change()
        df['mom_growth_lag1'] = df['mom_growth'].shift(1)
        df['yoy_growth'] = df[target_col].pct_change(12)
        df['yoy_growth_lag1'] = df['yoy_growth'].shift(1)
        
        df['trend'] = np.arange(len(df))
        df['trend_squared'] = df['trend'] ** 2
        
        df['month_sin'] = np.sin(2 * np.pi * df['date'].dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['date'].dt.month / 12)
        df['quarter_sin'] = np.sin(2 * np.pi * df['date'].dt.quarter / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['date'].dt.quarter / 4)
        
        df['is_festive'] = df['date'].dt.month.isin([10, 11, 12]).astype(int)
        df['is_q4'] = df['date'].dt.quarter == 4
        
        df['volume_diff'] = df[target_col].diff()
        df['volume_diff_lag1'] = df['volume_diff'].shift(1)
        
        df = df.dropna()
        
        self.featured_df = df
        self._cache[cache_key] = df
        
        return df
    
    def create_sequences(self, target_col: str = 'volume_millions', 
                         sequence_length: int = 6) -> Tuple[np.ndarray, np.ndarray]:
        df = self.featured_df.copy()
        
        values = df[target_col].values
        
        X, y = [], []
        for i in range(sequence_length, len(values)):
            X.append(values[i-sequence_length:i])
            y.append(values[i])
        
        return np.array(X), np.array(y)
    
    def get_anomalies(self, threshold: float = 2.0) -> List[Dict]:
        cache_key = f"anomalies_{threshold}"
        
        if self.use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        df = self.df.copy()
        
        df['volume_zscore'] = (df['volume_millions'] - df['volume_millions'].mean()) / df['volume_millions'].std()
        df['value_zscore'] = (df['value_crores'] - df['value_crores'].mean()) / df['value_crores'].std()
        
        df['volume_iqr'] = df['volume_millions'].quantile(0.75) - df['volume_millions'].quantile(0.25)
        df['volume_upper'] = df['volume_millions'].quantile(0.75) + 1.5 * df['volume_iqr']
        df['volume_lower'] = df['volume_millions'].quantile(0.25) - 1.5 * df['volume_iqr']
        
        df['is_outlier'] = (df['volume_millions'] > df['volume_upper']) | \
                           (df['volume_millions'] < df['volume_lower'])
        
        df['is_anomaly_zscore'] = abs(df['volume_zscore']) > threshold
        
        anomalies = df[(abs(df['volume_zscore']) > threshold) | df['is_outlier']]
        
        result = [
            {
                'date': row['date'].strftime('%Y-%m'),
                'month': row['month'],
                'volume': round(row['volume_millions'], 2),
                'value': round(row['value_crores'], 2),
                'volume_zscore': round(row['volume_zscore'], 2),
                'value_zscore': round(row['value_zscore'], 2),
                'is_outlier': bool(row['is_outlier']),
                'severity': 'high' if abs(row['volume_zscore']) > 2.5 else 'medium' if abs(row['volume_zscore']) > 2.0 else 'low'
            }
            for _, row in anomalies.iterrows()
        ]
        
        if self.use_cache:
            self._cache[cache_key] = result
        
        return result
    
    def get_time_series_data(self) -> Dict:
        return {
            'dates': self.df['date'].dt.strftime('%Y-%m').tolist(),
            'volume': self.df['volume_millions'].tolist(),
            'value': self.df['value_crores'].tolist(),
            'volume_growth': self.df['volume_millions'].pct_change().fillna(0).tolist(),
            'volume_diff': self.df['volume_millions'].diff().fillna(0).tolist()
        }
    
    def get_seasonality_pattern(self) -> Dict:
        df = self.df.copy()
        df['month'] = df['date'].dt.month
        
        monthly_avg = df.groupby('month')['volume_millions'].agg(['mean', 'std', 'count'])
        overall_avg = df['volume_millions'].mean()
        
        seasonality = {}
        for month in range(1, 13):
            if month in monthly_avg.index:
                avg = monthly_avg.loc[month, 'mean']
                std = monthly_avg.loc[month, 'std']
                seasonality[month] = {
                    'average': round(avg, 2),
                    'std': round(std, 2) if not pd.isna(std) else 0,
                    'seasonal_factor': round(avg / overall_avg, 3) if overall_avg > 0 else 1,
                    'is_peak': avg > overall_avg * 1.1
                }
        
        return {
            'seasonality': seasonality,
            'overall_average': round(overall_avg, 2),
            'peak_months': [k for k, v in seasonality.items() if v['is_peak']],
            'low_months': [k for k, v in seasonality.items() if v['seasonal_factor'] < 0.9]
        }
    
    def get_growth_trajectory(self) -> Dict:
        df = self.df.copy()
        
        df['yoy_growth'] = df['volume_millions'].pct_change(12) * 100
        df['mom_growth'] = df['volume_millions'].pct_change() * 100
        
        recent_12 = df['volume_millions'].tail(12)
        prev_12 = df['volume_millions'].tail(24).head(12)
        
        trajectory = {
            'current_volume': round(df['volume_millions'].iloc[-1], 2),
            'volume_12_months_ago': round(df['volume_millions'].iloc[-12], 2) if len(df) >= 12 else None,
            'recent_12m_avg': round(recent_12.mean(), 2),
            'prev_12m_avg': round(prev_12.mean(), 2),
            'acceleration': round((recent_12.mean() - prev_12.mean()) / prev_12.mean() * 100, 2) if len(prev_12) > 0 and prev_12.mean() > 0 else 0,
            'yoy_growth_latest': round(df['yoy_growth'].iloc[-1], 2) if len(df) >= 12 else None,
            'mom_growth_latest': round(df['mom_growth'].iloc[-1], 2),
            'trend_direction': 'accelerating' if recent_12.diff().mean() > prev_12.diff().mean() else 'decelerating',
            'predicted_reach_100m': self._predict_reach_target(df['volume_millions'].values, 100),
            'predicted_reach_200m': self._predict_reach_target(df['volume_millions'].values, 200)
        }
        
        return trajectory
    
    def _predict_reach_target(self, values: np.ndarray, target: float) -> Optional[str]:
        if values[-1] >= target:
            return "Already reached"
        
        if len(values) < 6:
            return "Insufficient data"
        
        recent_trend = np.polyfit(range(6), values[-6:], 1)
        monthly_increase = recent_trend[0]
        
        if monthly_increase <= 0:
            return "Trend unclear"
        
        months_needed = int((target - values[-1]) / monthly_increase)
        if months_needed > 60:
            return ">5 years"
        
        from datetime import datetime
        future_date = datetime.now() + pd.DateOffset(months=months_needed)
        return future_date.strftime('%Y-%m')
    
    def clear_cache(self):
        self._cache = {}

    def save_cache(self, filepath: str):
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'data_hash': hashlib.md5(pd.util.hash_pandas_object(self.df).values).hexdigest(),
            'featured_columns': list(self.featured_df.columns) if hasattr(self, 'featured_df') else []
        }
        
        with open(filepath, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        logger.info(f"Cache saved to {filepath}")

def process_data(df: pd.DataFrame, use_cache: bool = True) -> Tuple[pd.DataFrame, Dict, DataProcessor]:
    processor = DataProcessor(df, use_cache=use_cache)
    clean_df = processor.clean_data()
    stats = processor.get_eda_stats()
    featured_df = processor.create_features()
    
    return clean_df, stats, processor
