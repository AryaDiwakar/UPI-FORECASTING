"""
Data Preprocessing Module
Handles data cleaning, missing value imputation, and validation.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DataCleaner:
    """Clean and validate time series data."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.cleaning_log = []
    
    def clean(self) -> pd.DataFrame:
        """Apply all cleaning steps."""
        self.df = self._ensure_datetime()
        self.df = self._remove_duplicates()
        self.df = self._validate_values()
        self.df = self._ensure_continuity()
        return self.df
    
    def _ensure_datetime(self) -> pd.DataFrame:
        """Ensure date column is datetime type."""
        if 'date' not in self.df.columns:
            raise ValueError("DataFrame must have 'date' column")
        
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date').reset_index(drop=True)
        return self.df
    
    def _remove_duplicates(self) -> pd.DataFrame:
        """Remove duplicate dates, keeping last value."""
        before = len(self.df)
        self.df = self.df.drop_duplicates(subset=['date'], keep='last')
        after = len(self.df)
        
        if before != after:
            self.cleaning_log.append(f"Removed {before - after} duplicate rows")
        
        return self.df
    
    def _validate_values(self) -> pd.DataFrame:
        """Validate and fix data quality issues."""
        issues = []
        
        if (self.df['volume_millions'] <= 0).any():
            neg_count = (self.df['volume_millions'] <= 0).sum()
            issues.append(f"Found {neg_count} non-positive volumes")
            self.df.loc[self.df['volume_millions'] <= 0, 'volume_millions'] = np.nan
        
        if self.df['volume_millions'].isna().any():
            na_count = self.df['volume_millions'].isna().sum()
            issues.append(f"Found {na_count} missing volumes")
        
        for issue in issues:
            self.cleaning_log.append(issue)
        
        return self.df
    
    def _ensure_continuity(self) -> pd.DataFrame:
        """Check for gaps in time series."""
        self.df = self.df.set_index('date')
        
        full_range = pd.date_range(
            start=self.df.index.min(),
            end=self.df.index.max(),
            freq='MS'
        )
        
        missing_dates = full_range.difference(self.df.index)
        
        if len(missing_dates) > 0:
            self.cleaning_log.append(f"Found {len(missing_dates)} missing months")
        
        self.df = self.df.reindex(full_range)
        self.df.index.name = 'date'
        self.df = self.df.reset_index()
        
        return self.df
    
    def get_cleaning_report(self) -> Dict[str, Any]:
        """Generate cleaning report."""
        return {
            'original_rows': len(self.df),
            'cleaning_actions': self.cleaning_log,
            'missing_after_cleaning': self.df['volume_millions'].isna().sum()
        }


class MissingValueImputer:
    """Handle missing values in time series data."""
    
    def __init__(self, df: pd.DataFrame, target_col: str = 'volume_millions'):
        self.df = df.copy()
        self.target_col = target_col
        self.method_used = None
    
    def impute(self, method: str = 'interpolate') -> Tuple[pd.DataFrame, Dict]:
        """
        Impute missing values using specified method.
        
        Methods:
            - forward: Forward fill
            - backward: Backward fill
            - interpolate: Linear interpolation
            - spline: Spline interpolation
            - mean: Fill with mean
            - median: Fill with median
        """
        df = self.df.copy()
        missing_before = df[self.target_col].isna().sum()
        
        if missing_before == 0:
            self.method_used = 'none'
            return df, {'method': 'none', 'values_imputed': 0}
        
        original = df[self.target_col].copy()
        
        if method == 'forward':
            df[self.target_col] = df[self.target_col].ffill()
        
        elif method == 'backward':
            df[self.target_col] = df[self.target_col].bfill()
        
        elif method == 'interpolate':
            df[self.target_col] = df[self.target_col].interpolate(method='linear')
        
        elif method == 'spline':
            df[self.target_col] = df[self.target_col].interpolate(method='spline', order=3)
        
        elif method == 'mean':
            df[self.target_col] = df[self.target_col].fillna(df[self.target_col].mean())
        
        elif method == 'median':
            df[self.target_col] = df[self.target_col].fillna(df[self.target_col].median())
        
        else:
            raise ValueError(f"Unknown imputation method: {method}")
        
        missing_after = df[self.target_col].isna().sum()
        
        self.method_used = method
        
        report = {
            'method': method,
            'values_imputed': missing_before - missing_after,
            'missing_before': missing_before,
            'missing_after': missing_after
        }
        
        if method == 'interpolate':
            imputed_dates = df.loc[original.isna() & df[self.target_col].notna(), 'date'].tolist()
            report['imputed_dates'] = [d.strftime('%Y-%m') for d in imputed_dates[:5]]
        
        return df, report


class OutlierDetector:
    """Detect and handle outliers in time series data."""
    
    def __init__(self, df: pd.DataFrame, target_col: str = 'volume_millions'):
        self.df = df.copy()
        self.target_col = target_col
        self.anomalies = None
    
    def detect_zscore(self, threshold: float = 2.5) -> pd.DataFrame:
        """Detect outliers using Z-score method."""
        values = self.df[self.target_col].values
        mean = np.mean(values)
        std = np.std(values)
        
        z_scores = np.abs((values - mean) / std) if std > 0 else np.zeros_like(values)
        
        anomalies = self.df.copy()
        anomalies['z_score'] = z_scores
        anomalies['is_anomaly_zscore'] = z_scores > threshold
        anomalies['severity'] = 'normal'
        anomalies.loc[z_scores > 2.0, 'severity'] = 'low'
        anomalies.loc[z_scores > 2.5, 'severity'] = 'medium'
        anomalies.loc[z_scores > 3.0, 'severity'] = 'high'
        anomalies.loc[z_scores > 3.5, 'severity'] = 'critical'
        
        self.anomalies = anomalies[anomalies['is_anomaly_zscore']]
        return anomalies
    
    def detect_iqr(self, multiplier: float = 1.5) -> pd.DataFrame:
        """Detect outliers using IQR method."""
        q1 = self.df[self.target_col].quantile(0.25)
        q3 = self.df[self.target_col].quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        anomalies = self.df.copy()
        anomalies['iqr_lower'] = lower_bound
        anomalies['iqr_upper'] = upper_bound
        anomalies['is_anomaly_iqr'] = (
            (anomalies[self.target_col] < lower_bound) | 
            (anomalies[self.target_col] > upper_bound)
        )
        
        iqr_anomalies = anomalies[anomalies['is_anomaly_iqr']]
        if self.anomalies is not None:
            self.anomalies = pd.concat([self.anomalies, iqr_anomalies]).drop_duplicates()
        else:
            self.anomalies = iqr_anomalies
        
        return anomalies
    
    def detect_all(self, z_threshold: float = 2.5, iqr_multiplier: float = 1.5) -> Dict[str, Any]:
        """Run all outlier detection methods."""
        df_zscore = self.detect_zscore(z_threshold)
        df_iqr = self.detect_iqr(iqr_multiplier)
        
        df_combined = df_zscore.copy()
        df_combined['is_anomaly_iqr'] = df_iqr['is_anomaly_iqr']
        df_combined['is_anomaly'] = df_combined['is_anomaly_zscore'] | df_combined['is_anomaly_iqr']
        
        self.anomalies = df_combined[df_combined['is_anomaly']]
        
        return {
            'zscore_anomalies': int(df_combined['is_anomaly_zscore'].sum()),
            'iqr_anomalies': int(df_combined['is_anomaly_iqr'].sum()),
            'total_anomalies': int(df_combined['is_anomaly'].sum()),
            'anomaly_rate': float(df_combined['is_anomaly'].mean() * 100),
            'zscore_threshold': z_threshold,
            'iqr_multiplier': iqr_multiplier
        }
    
    def get_anomalies(self, with_context: bool = True) -> pd.DataFrame:
        """Get detected anomalies with possible causes."""
        if self.anomalies is None:
            self.detect_all()
        
        if not with_context:
            return self.anomalies
        
        anomalies = self.anomalies.copy()
        
        if 'date' not in anomalies.columns or anomalies.empty:
            return pd.DataFrame(columns=['date', 'volume_millions', 'z_score', 'severity', 'possible_cause'])
        
        def infer_cause(row):
            month = row['date'].month if hasattr(row['date'], 'month') else 1
            festive_months = [10, 11, 12]
            
            if month in festive_months:
                return "Festive season spike"
            elif month in [4, 5, 6]:
                return "Post-COVID adoption surge"
            elif row.get('z_score', 0) > 3:
                return "Exceptional volume"
            else:
                return "Unusual pattern"
        
        result = anomalies[['date', 'volume_millions', 'z_score', 'severity']].copy()
        result['possible_cause'] = result.apply(infer_cause, axis=1)
        
        return result
    
    def exclude_from_training(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into clean and anomalous."""
        if self.anomalies is None:
            self.detect_all()
        
        anomaly_dates = set(self.anomalies['date'])
        
        clean_df = df[~df['date'].isin(anomaly_dates)].copy()
        anomalous_df = df[df['date'].isin(anomaly_dates)].copy()
        
        return clean_df, anomalous_df


class DataValidator:
    """Validate data quality and schema."""
    
    @staticmethod
    def validate(df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data validation."""
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        required_cols = ['date', 'volume_millions']
        for col in required_cols:
            if col not in df.columns:
                results['is_valid'] = False
                results['errors'].append(f"Missing required column: {col}")
        
        if not results['is_valid']:
            return results
        
        if df['volume_millions'].isna().any():
            results['warnings'].append(
                f"Contains {df['volume_millions'].isna().sum()} missing values"
            )
        
        if (df['volume_millions'] <= 0).any():
            results['warnings'].append(
                f"Contains {(df['volume_millions'] <= 0).sum()} non-positive values"
            )
        
        if not df['date'].is_monotonic_increasing:
            results['warnings'].append("Dates are not in chronological order")
        
        results['info'] = {
            'total_records': len(df),
            'date_range': {
                'start': df['date'].min().strftime('%Y-%m'),
                'end': df['date'].max().strftime('%Y-%m')
            },
            'missing_values': int(df['volume_millions'].isna().sum()),
            'volume_stats': {
                'mean': float(df['volume_millions'].mean()),
                'std': float(df['volume_millions'].std()),
                'min': float(df['volume_millions'].min()),
                'max': float(df['volume_millions'].max())
            }
        }
        
        return results


def preprocess_data(df: pd.DataFrame, 
                   impute_method: str = 'interpolate',
                   detect_outliers: bool = True,
                   z_threshold: float = 2.5) -> Tuple[pd.DataFrame, Dict]:
    """
    Complete preprocessing pipeline.
    
    Returns:
        Tuple of (preprocessed_df, preprocessing_report)
    """
    report = {'steps': []}
    
    cleaner = DataCleaner(df)
    df = cleaner.clean()
    report['steps'].append({
        'step': 'cleaning',
        'details': cleaner.get_cleaning_report()
    })
    
    imputer = MissingValueImputer(df)
    df, impute_report = imputer.impute(impute_method)
    report['steps'].append({
        'step': 'imputation',
        'details': impute_report
    })
    
    if detect_outliers:
        detector = OutlierDetector(df)
        outlier_report = detector.detect_all(z_threshold=z_threshold)
        
        # Add is_anomaly column to the dataframe
        df['is_anomaly'] = False
        if detector.anomalies is not None and len(detector.anomalies) > 0:
            anomaly_dates = detector.anomalies['date'].tolist()
            df.loc[df['date'].isin(anomaly_dates), 'is_anomaly'] = True
        
        report['steps'].append({
            'step': 'outlier_detection',
            'details': outlier_report
        })
        report['anomalies'] = detector.get_anomalies()
    
    validation = DataValidator.validate(df)
    report['steps'].append({
        'step': 'validation',
        'details': validation
    })
    
    report['preprocessed_records'] = len(df)
    
    return df, report


if __name__ == "__main__":
    from backend.data.scraper import fetch_and_store_data
    
    df, _ = fetch_and_store_data()
    
    cleaner = DataCleaner(df)
    df_clean = cleaner.clean()
    
    print("Cleaning Report:", cleaner.get_cleaning_report())
    print("\nData shape after cleaning:", df_clean.shape)
