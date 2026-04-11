"""
Feature Engineering Module
Creates comprehensive features for time series forecasting models.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from statsmodels.tsa.seasonal import seasonal_decompose

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Create features for time series forecasting."""
    
    def __init__(self, df: pd.DataFrame, target_col: str = 'volume_millions'):
        self.df = df.copy()
        self.target_col = target_col
        self.feature_names = []
    
    def create_all_features(self, 
                           lags: List[int] = None,
                           rolling_windows: List[int] = None,
                           include_decomposition: bool = True,
                           include_fourier: bool = True) -> pd.DataFrame:
        """
        Create comprehensive feature set for time series.
        
        Args:
            lags: List of lag periods (default: [1, 3, 6, 12])
            rolling_windows: List of rolling window sizes (default: [3, 6, 12])
            include_decomposition: Include trend/seasonal/residual components
            include_fourier: Include Fourier terms for seasonality
        """
        if lags is None:
            lags = [1, 3, 6, 12]
        if rolling_windows is None:
            rolling_windows = [3, 6, 12]
        
        df = self.df.copy()
        
        df = self._add_lag_features(df, lags)
        df = self._add_rolling_features(df, rolling_windows)
        df = self._add_ema_features(df)
        df = self._add_growth_features(df)
        df = self._add_temporal_features(df)
        
        if include_decomposition:
            df = self._add_decomposition_features(df)
        
        if include_fourier:
            df = self._add_fourier_features(df)
        
        df = self._add_event_features(df)
        df = self._add_momentum_features(df)
        
        df = df.dropna().reset_index(drop=True)
        self.feature_names = [c for c in df.columns 
                            if c not in ['date', 'month', self.target_col, 'value_crores', 'source']]
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
        """Add lag features."""
        for lag in lags:
            df[f'lag_{lag}'] = df[self.target_col].shift(lag)
            self.feature_names.append(f'lag_{lag}')
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Add rolling window features."""
        for window in windows:
            df[f'rolling_mean_{window}'] = df[self.target_col].shift(1).rolling(window=window).mean()
            df[f'rolling_std_{window}'] = df[self.target_col].shift(1).rolling(window=window).std()
            df[f'rolling_min_{window}'] = df[self.target_col].shift(1).rolling(window=window).min()
            df[f'rolling_max_{window}'] = df[self.target_col].shift(1).rolling(window=window).max()
            df[f'rolling_median_{window}'] = df[self.target_col].shift(1).rolling(window=window).median()
            
            for feat in [f'rolling_mean_{window}', f'rolling_std_{window}', 
                         f'rolling_min_{window}', f'rolling_max_{window}', 
                         f'rolling_median_{window}']:
                self.feature_names.append(feat)
        
        return df
    
    def _add_ema_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add exponential moving average features."""
        for span in [3, 6, 12]:
            df[f'ema_{span}'] = df[self.target_col].shift(1).ewm(span=span, adjust=False).mean()
            self.feature_names.append(f'ema_{span}')
        return df
    
    def _add_growth_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add growth rate features."""
        df['mom_growth'] = df[self.target_col].pct_change()
        df['mom_growth_lag1'] = df['mom_growth'].shift(1)
        df['mom_growth_lag3'] = df['mom_growth'].shift(3)
        
        df['yoy_growth'] = df[self.target_col].pct_change(12)
        df['yoy_growth_lag1'] = df['yoy_growth'].shift(1)
        
        df['mom_acceleration'] = df['mom_growth'].diff()
        
        df['cumulative_growth'] = (df[self.target_col] / df[self.target_col].iloc[0]) - 1
        
        for feat in ['mom_growth', 'mom_growth_lag1', 'mom_growth_lag3',
                     'yoy_growth', 'yoy_growth_lag1', 'mom_acceleration', 'cumulative_growth']:
            self.feature_names.append(feat)
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calendar/temporal features."""
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['date'].dt.year
        df['month_of_year'] = df['date'].dt.month
        
        df['month_sin'] = np.sin(2 * np.pi * df['date'].dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['date'].dt.month / 12)
        
        df['quarter_sin'] = np.sin(2 * np.pi * df['date'].dt.quarter / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['date'].dt.quarter / 4)
        
        df['years_since_start'] = (df['date'].dt.year - df['date'].dt.year.min()) + \
                                 (df['date'].dt.month - 1) / 12
        
        df['quarter_of_year'] = df['date'].dt.month.apply(lambda x: (x - 1) // 3 + 1)
        
        for feat in ['month', 'quarter', 'year', 'month_of_year', 'month_sin', 'month_cos',
                     'quarter_sin', 'quarter_cos', 'years_since_start', 'quarter_of_year']:
            self.feature_names.append(feat)
        
        return df
    
    def _add_decomposition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add seasonal decomposition features."""
        try:
            decomposition = seasonal_decompose(
                df[self.target_col].dropna(), 
                model='multiplicative', 
                period=12
            )
            
            df['decomp_trend'] = decomposition.trend.shift(1)
            df['decomp_seasonal'] = decomposition.seasonal.shift(1)
            df['decomp_resid'] = decomposition.resid.shift(1)
            
            for feat in ['decomp_trend', 'decomp_seasonal', 'decomp_resid']:
                self.feature_names.append(feat)
                
        except Exception as e:
            logger.warning(f"Decomposition failed: {e}")
        
        return df
    
    def _add_fourier_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Fourier terms for seasonality."""
        for k in [1, 2, 3]:
            df[f'fourier_sin_{k}'] = np.sin(2 * np.pi * k * df['date'].dt.month / 12)
            df[f'fourier_cos_{k}'] = np.cos(2 * np.pi * k * df['date'].dt.month / 12)
            
            self.feature_names.append(f'fourier_sin_{k}')
            self.feature_names.append(f'fourier_cos_{k}')
        
        return df
    
    def _add_event_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add event/festival indicators."""
        df['is_festive'] = df['date'].dt.month.isin([10, 11, 12]).astype(int)
        df['is_q4'] = (df['date'].dt.quarter == 4).astype(int)
        df['is_jan'] = (df['date'].dt.month == 1).astype(int)
        df['is_lockdown'] = ((df['date'] >= '2020-03') & (df['date'] <= '2021-03')).astype(int)
        
        df['days_to_festive'] = df['date'].apply(lambda x: self._days_to_festive(x))
        
        for feat in ['is_festive', 'is_q4', 'is_jan', 'is_lockdown', 'days_to_festive']:
            self.feature_names.append(feat)
        
        return df
    
    def _days_to_festive(self, date) -> int:
        """Calculate days to festive season (October)."""
        if date.month >= 10:
            return 0
        return (10 - date.month) * 30
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        df['volume_diff'] = df[self.target_col].diff()
        df['volume_diff_lag1'] = df['volume_diff'].shift(1)
        df['volume_diff_lag3'] = df['volume_diff'].shift(3)
        
        df['volatility_3m'] = df[self.target_col].rolling(3).std() / df[self.target_col].rolling(3).mean()
        df['volatility_6m'] = df[self.target_col].rolling(6).std() / df[self.target_col].rolling(6).mean()
        df['volatility_12m'] = df[self.target_col].rolling(12).std() / df[self.target_col].rolling(12).mean()
        
        df['price_momentum_3'] = df[self.target_col] - df[self.target_col].shift(3)
        df['price_momentum_6'] = df[self.target_col] - df[self.target_col].shift(6)
        
        for feat in ['volume_diff', 'volume_diff_lag1', 'volume_diff_lag3',
                     'volatility_3m', 'volatility_6m', 'volatility_12m',
                     'price_momentum_3', 'price_momentum_6']:
            self.feature_names.append(feat)
        
        return df
    
    def get_feature_importance_df(self, importance_dict: Dict[str, float]) -> pd.DataFrame:
        """Create sorted DataFrame of feature importance."""
        importance_df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in importance_dict.items()
        ]).sort_values('importance', ascending=False)
        
        importance_df['cumulative'] = importance_df['importance'].cumsum()
        importance_df['rank'] = range(1, len(importance_df) + 1)
        
        return importance_df


def create_features(df: pd.DataFrame,
                   target_col: str = 'volume_millions',
                   **kwargs) -> Tuple[pd.DataFrame, List[str]]:
    """
    Convenience function to create all features.
    
    Returns:
        Tuple of (featured_df, feature_names)
    """
    engineer = FeatureEngineer(df, target_col)
    df_featured = engineer.create_all_features(**kwargs)
    return df_featured, engineer.feature_names


if __name__ == "__main__":
    from backend.data.scraper import fetch_and_store_data
    from backend.preprocessing.cleaner import preprocess_data
    
    df, _ = fetch_and_store_data()
    df_clean, _ = preprocess_data(df)
    
    engineer = FeatureEngineer(df_clean)
    df_featured = engineer.create_all_features()
    
    print(f"Created {len(engineer.feature_names)} features")
    print(f"Data shape: {df_featured.shape}")
    print(f"\nFeature names: {engineer.feature_names[:10]}...")
