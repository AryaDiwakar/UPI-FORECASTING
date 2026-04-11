"""
Feature engineering for time series forecasting.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Create features for time series forecasting."""
    
    def __init__(self, 
                 lags: List[int] = None,
                 rolling_windows: List[int] = None,
                 include_growth: bool = True,
                 include_temporal: bool = True,
                 include_volatility: bool = True):
        
        self.lags = lags or [1, 3, 6, 12]
        self.rolling_windows = rolling_windows or [3, 6, 12]
        self.include_growth = include_growth
        self.include_temporal = include_temporal
        self.include_volatility = include_volatility
        self.feature_names: List[str] = []
        
    def create_all_features(self, df: pd.DataFrame, 
                           target_col: str = 'volume_millions',
                           is_train: bool = True) -> pd.DataFrame:
        """Create all features for the dataset."""
        result_df = df.copy()
        
        if 'date' not in result_df.columns:
            return result_df
        
        result_df = self._add_lag_features(result_df, target_col)
        result_df = self._add_rolling_features(result_df, target_col)
        
        if self.include_growth:
            result_df = self._add_growth_features(result_df, target_col)
        
        if self.include_temporal:
            result_df = self._add_temporal_features(result_df)
        
        if self.include_volatility:
            result_df = self._add_volatility_features(result_df, target_col)
        
        result_df = self._add_interaction_features(result_df)
        
        result_df = result_df.dropna()
        
        self.feature_names = [col for col in result_df.columns 
                            if col not in ['month', 'date', 'volume_millions', 'value_crores']]
        
        return result_df
    
    def _add_lag_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Add lag features."""
        for lag in self.lags:
            df[f'lag_{lag}'] = df[target_col].shift(lag)
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Add rolling window features."""
        for window in self.rolling_windows:
            df[f'rolling_mean_{window}'] = df[target_col].shift(1).rolling(window=window).mean()
            df[f'rolling_std_{window}'] = df[target_col].shift(1).rolling(window=window).std()
            df[f'rolling_min_{window}'] = df[target_col].shift(1).rolling(window=window).min()
            df[f'rolling_max_{window}'] = df[target_col].shift(1).rolling(window=window).max()
            df[f'rolling_median_{window}'] = df[target_col].shift(1).rolling(window=window).median()
        return df
    
    def _add_growth_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Add growth rate features."""
        df['mom_growth'] = df[target_col].pct_change()
        df['mom_growth_lag1'] = df['mom_growth'].shift(1)
        df['mom_growth_lag3'] = df['mom_growth'].shift(3)
        
        df['yoy_growth'] = df[target_col].pct_change(12)
        df['yoy_growth_lag1'] = df['yoy_growth'].shift(1)
        
        df['mom_acceleration'] = df['mom_growth'].diff()
        df['mom_acceleration_lag1'] = df['mom_acceleration'].shift(1)
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal/calendar features."""
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['date'].dt.year
        df['month_of_year'] = df['date'].dt.month
        
        df['month_sin'] = np.sin(2 * np.pi * df['date'].dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['date'].dt.month / 12)
        
        df['quarter_sin'] = np.sin(2 * np.pi * df['date'].dt.quarter / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['date'].dt.quarter / 4)
        
        df['is_festive'] = df['date'].dt.month.isin([10, 11, 12]).astype(int)
        df['is_q4'] = (df['date'].dt.quarter == 4).astype(int)
        df['is_jan'] = (df['date'].dt.month == 1).astype(int)
        
        df['years_since_start'] = (df['date'].dt.year - df['date'].dt.year.min()) + \
                                  (df['date'].dt.month - 1) / 12
        
        df['quarter_of_year'] = df['date'].dt.month.apply(lambda x: (x - 1) // 3 + 1)
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Add volatility features."""
        df['volume_diff'] = df[target_col].diff()
        df['volume_diff_lag1'] = df['volume_diff'].shift(1)
        df['volume_diff_lag3'] = df['volume_diff'].shift(3)
        
        df['volatility_3m'] = df[target_col].rolling(window=3).std() / df[target_col].rolling(window=3).mean()
        df['volatility_6m'] = df[target_col].rolling(window=6).std() / df[target_col].rolling(window=6).mean()
        df['volatility_12m'] = df[target_col].rolling(window=12).std() / df[target_col].rolling(window=12).mean()
        
        df['volume_ratio_3_12'] = df['rolling_mean_3'] / df['rolling_mean_12']
        df['volume_ratio_6_12'] = df['rolling_mean_6'] / df['rolling_mean_12']
        
        return df
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features."""
        if 'mom_growth' in df.columns and 'is_festive' in df.columns:
            df['festive_growth_boost'] = df['mom_growth'] * df['is_festive']
        
        if 'rolling_mean_3' in df.columns and 'rolling_mean_12' in df.columns:
            df['trend_strength'] = (df['rolling_mean_3'] - df['rolling_mean_12']) / df['rolling_mean_12']
        
        if 'lag_1' in df.columns and 'lag_12' in df.columns:
            df['seasonal_ratio'] = df['lag_1'] / df['lag_12']
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self.feature_names
    
    def get_top_features(self, importance_dict: Dict[str, float], n: int = 10) -> List[Tuple[str, float]]:
        """Get top N most important features."""
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:n]


class AnomalyDetector:
    """Detect anomalies in time series data."""
    
    def __init__(self, z_threshold: float = 2.5, iqr_multiplier: float = 1.5):
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier
        
    def detect_zscore(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect anomalies using Z-score method."""
        mean = np.mean(values)
        std = np.std(values)
        z_scores = np.abs((values - mean) / std) if std > 0 else np.zeros_like(values)
        anomalies = z_scores > self.z_threshold
        return anomalies, z_scores
    
    def detect_iqr(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Detect anomalies using IQR method."""
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - self.iqr_multiplier * iqr
        upper_bound = q3 + self.iqr_multiplier * iqr
        
        anomalies = (values < lower_bound) | (values > upper_bound)
        return anomalies, lower_bound, upper_bound, iqr
    
    def detect_combined(self, df: pd.DataFrame, 
                       target_col: str = 'volume_millions') -> pd.DataFrame:
        """Combine multiple anomaly detection methods."""
        result_df = df.copy()
        values = result_df[target_col].values
        
        z_anomalies, z_scores = self.detect_zscore(values)
        iqr_anomalies, lower, upper, iqr = self.detect_iqr(values)
        
        result_df['z_score'] = z_scores
        result_df['z_anomaly'] = z_anomalies
        result_df['iqr_anomaly'] = iqr_anomalies
        result_df['is_anomaly'] = z_anomalies | iqr_anomalies
        
        result_df['anomaly_severity'] = 'normal'
        result_df.loc[z_scores > 2.0, 'anomaly_severity'] = 'low'
        result_df.loc[z_scores > 2.5, 'anomaly_severity'] = 'medium'
        result_df.loc[z_scores > 3.0, 'anomaly_severity'] = 'high'
        result_df.loc[z_scores > 3.5, 'anomaly_severity'] = 'critical'
        
        result_df['anomaly_score'] = z_scores
        result_df['iqr_lower'] = lower
        result_df['iqr_upper'] = upper
        
        return result_df
    
    def get_anomaly_context(self, df: pd.DataFrame, 
                           target_col: str = 'volume_millions') -> pd.DataFrame:
        """Add context to detected anomalies."""
        result_df = self.detect_combined(df, target_col)
        
        anomalies_df = result_df[result_df['is_anomaly']].copy()
        
        if len(anomalies_df) > 0:
            for idx in anomalies_df.index:
                month = result_df.loc[idx, 'date'].month if 'date' in result_df.columns else None
                volume = result_df.loc[idx, target_col]
                
                cause = self._infer_anomaly_cause(month, volume, result_df)
                result_df.loc[idx, 'possible_cause'] = cause
        
        return result_df
    
    def _infer_anomaly_cause(self, month: int, volume: float, 
                            df: pd.DataFrame) -> str:
        """Infer possible cause of anomaly."""
        festive_months = [10, 11, 12]
        post_covid_months = [4, 5, 6]
        
        mean_volume = df['volume_millions'].mean()
        
        if month in festive_months and volume > mean_volume:
            return "Festive season spike"
        elif month in post_covid_months and volume > mean_volume:
            return "Post-COVID adoption surge"
        elif volume < mean_volume * 0.5:
            return "Potential system outage or data issue"
        elif volume > df['volume_millions'].quantile(0.95):
            return "Exceptional transaction volume"
        else:
            return "Unusual pattern detected"


class DataValidator:
    """Validate and check data quality."""
    
    @staticmethod
    def validate_schema(df: pd.DataFrame) -> Dict[str, Any]:
        """Validate dataframe schema."""
        required_cols = ['month', 'volume_millions', 'value_crores']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        return {
            'valid': len(missing_cols) == 0,
            'missing_columns': missing_cols,
            'row_count': len(df),
            'date_range': {
                'start': df['date'].min().strftime('%Y-%m') if 'date' in df.columns and len(df) > 0 else None,
                'end': df['date'].max().strftime('%Y-%m') if 'date' in df.columns and len(df) > 0 else None
            } if 'date' in df.columns else None
        }
    
    @staticmethod
    def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
        """Check data quality issues."""
        issues = []
        
        if df['volume_millions'].isna().sum() > 0:
            issues.append(f"Missing volume values: {df['volume_millions'].isna().sum()}")
        
        if (df['volume_millions'] <= 0).sum() > 0:
            issues.append(f"Non-positive volume values: {(df['volume_millions'] <= 0).sum()}")
        
        if df['volume_millions'].duplicated().sum() > 0:
            issues.append(f"Duplicate volume values: {df['volume_millions'].duplicated().sum()}")
        
        return {
            'quality_score': 100 - len(issues) * 10,
            'issues': issues,
            'has_issues': len(issues) > 0
        }


def create_features(df: pd.DataFrame, 
                   exclude_anomalies: bool = False,
                   anomaly_threshold: float = 2.5) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Convenience function to create all features."""
    feature_engineer = FeatureEngineer()
    anomaly_detector = AnomalyDetector(z_threshold=anomaly_threshold)
    
    result_df = anomaly_detector.detect_combined(df)
    
    if exclude_anomalies:
        result_df = result_df[~result_df['is_anomaly']].copy()
    
    featured_df = feature_engineer.create_all_features(result_df)
    
    metadata = {
        'total_records': len(df),
        'clean_records': len(featured_df),
        'anomalies_detected': result_df['is_anomaly'].sum() if 'is_anomaly' in result_df.columns else 0,
        'anomalies_excluded': result_df['is_anomaly'].sum() if exclude_anomalies else 0,
        'feature_count': len(feature_engineer.get_feature_names()),
        'feature_names': feature_engineer.get_feature_names()
    }
    
    return featured_df, metadata
