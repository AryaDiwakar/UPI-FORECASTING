"""
Advanced Exploratory Data Analysis Module
Performs comprehensive statistical analysis of time series data.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class TrendAnalyzer:
    """Analyze trend patterns in time series data."""
    
    def __init__(self, df: pd.DataFrame, target_col: str = 'volume_millions'):
        self.df = df.copy()
        self.target_col = target_col
        self.values = df[target_col].values
        self.dates = df['date']
    
    def rolling_statistics(self, windows: list = None) -> Dict[str, Any]:
        """Calculate rolling mean and standard deviation."""
        if windows is None:
            windows = [3, 6, 12]
        
        results = {'windows': {}}
        
        for window in windows:
            rolling_mean = self.df[self.target_col].rolling(window=window).mean()
            rolling_std = self.df[self.target_col].rolling(window=window).std()
            
            results['windows'][f'{window}m'] = {
                'mean_series': rolling_mean.dropna().tolist(),
                'std_series': rolling_std.dropna().tolist(),
                'latest_mean': rolling_mean.iloc[-1],
                'latest_std': rolling_std.iloc[-1],
                'dates': self.df['date'].iloc[window-1:].dt.strftime('%Y-%m').tolist()
            }
        
        return results
    
    def trend_type(self) -> Dict[str, Any]:
        """Determine the type of trend."""
        x = np.arange(len(self.values))
        
        linear_coef = np.polyfit(x, self.values, 1)
        linear_fit = np.poly1d(linear_coef)
        linear_r2 = 1 - (np.sum((self.values - linear_fit(x))**2) / 
                        np.sum((self.values - np.mean(self.values))**2))
        
        exponential_coef = np.polyfit(x, np.log(self.values + 1), 1)
        exponential_fit = np.exp(exponential_coef[0] * x + exponential_coef[1]) - 1
        exponential_r2 = 1 - (np.sum((self.values - exponential_fit)**2) / 
                            np.sum((self.values - np.mean(self.values))**2))
        
        polynomial_coef = np.polyfit(x, self.values, 2)
        polynomial_fit = np.poly1d(polynomial_coef)
        polynomial_r2 = 1 - (np.sum((self.values - polynomial_fit(x))**2) / 
                            np.sum((self.values - np.mean(self.values))**2))
        
        if linear_r2 > 0.95:
            trend_type = "strong_linear"
        elif exponential_r2 > 0.95:
            trend_type = "strong_exponential"
        elif polynomial_r2 > 0.95:
            trend_type = "polynomial"
        elif linear_r2 > 0.80:
            trend_type = "weak_linear"
        else:
            trend_type = "non_linear"
        
        return {
            'trend_type': trend_type,
            'linear_r2': round(linear_r2, 4),
            'exponential_r2': round(exponential_r2, 4),
            'polynomial_r2': round(polynomial_r2, 4),
            'coefficients': {
                'linear_slope': round(linear_coef[0], 4),
                'exponential_rate': round(exponential_coef[0], 4)
            },
            'interpretation': self._interpret_trend(trend_type, linear_coef[0])
        }
    
    def _interpret_trend(self, trend_type: str, slope: float) -> str:
        """Generate trend interpretation."""
        monthly_growth = (slope / np.mean(self.values)) * 100
        
        if monthly_growth > 5:
            growth_rate = "explosive"
        elif monthly_growth > 2:
            growth_rate = "strong"
        elif monthly_growth > 0.5:
            growth_rate = "moderate"
        else:
            growth_rate = "slow"
        
        return f"Data shows {growth_rate} growth with {trend_type.replace('_', ' ')} pattern"


class SeasonalityAnalyzer:
    """Analyze seasonal patterns in time series."""
    
    def __init__(self, df: pd.DataFrame, target_col: str = 'volume_millions'):
        self.df = df.copy()
        self.target_col = target_col
        self.df['month'] = self.df['date'].dt.month
        self.df['quarter'] = self.df['date'].dt.quarter
        self.df['year'] = self.df['date'].dt.year
    
    def monthly_patterns(self) -> Dict[str, Any]:
        """Analyze monthly seasonal patterns."""
        monthly_stats = self.df.groupby('month')[self.target_col].agg([
            'mean', 'std', 'count', 'min', 'max'
        ])
        
        overall_mean = self.df[self.target_col].mean()
        
        seasonal_factors = {}
        for month in range(1, 13):
            if month in monthly_stats.index:
                factor = monthly_stats.loc[month, 'mean'] / overall_mean
                seasonal_factors[month] = {
                    'mean': round(monthly_stats.loc[month, 'mean'], 2),
                    'seasonal_factor': round(factor, 3),
                    'is_peak': factor > 1.1,
                    'is_low': factor < 0.9
                }
        
        peak_months = [m for m, s in seasonal_factors.items() if s['is_peak']]
        low_months = [m for m, s in seasonal_factors.items() if s['is_low']]
        
        return {
            'overall_mean': round(overall_mean, 2),
            'seasonal_factors': seasonal_factors,
            'peak_months': peak_months,
            'low_months': low_months,
            'strength': self._calculate_seasonal_strength()
        }
    
    def quarterly_patterns(self) -> Dict[str, Any]:
        """Analyze quarterly patterns."""
        quarterly_stats = self.df.groupby('quarter')[self.target_col].agg(['mean', 'std'])
        
        overall_mean = self.df[self.target_col].mean()
        
        quarterly_factors = {}
        for q in range(1, 5):
            if q in quarterly_stats.index:
                factor = quarterly_stats.loc[q, 'mean'] / overall_mean
                quarterly_factors[f'Q{q}'] = {
                    'mean': round(quarterly_stats.loc[q, 'mean'], 2),
                    'seasonal_factor': round(factor, 3),
                    'is_peak': factor > 1.1
                }
        
        return {
            'quarterly_factors': quarterly_factors,
            'strongest_quarter': max(quarterly_factors.items(), key=lambda x: x[1]['seasonal_factor'])[0]
        }
    
    def _calculate_seasonal_strength(self) -> str:
        """Calculate strength of seasonality."""
        monthly_means = self.df.groupby('month')[self.target_col].mean()
        variance_between = monthly_means.var()
        variance_total = self.df[self.target_col].var()
        
        if variance_total == 0:
            return "none"
        
        strength = variance_between / variance_total
        
        if strength > 0.6:
            return "strong"
        elif strength > 0.3:
            return "moderate"
        elif strength > 0.1:
            return "weak"
        else:
            return "negligible"
    
    def decompose(self, period: int = 12, model: str = 'multiplicative') -> Dict[str, Any]:
        """Decompose time series into trend, seasonal, and residual components."""
        decomposition = seasonal_decompose(
            self.df[self.target_col].dropna(),
            model=model,
            period=period
        )
        
        return {
            'trend': decomposition.trend.dropna().tolist(),
            'seasonal': decomposition.seasonal.tolist()[:period],
            'residual': decomposition.resid.dropna().tolist(),
            'trend_direction': 'increasing' if decomposition.trend.iloc[-12:].mean() > decomposition.trend.iloc[-24:-12].mean() else 'decreasing',
            'model': model,
            'period': period
        }


class StationarityTester:
    """Test for stationarity using multiple statistical tests."""
    
    def __init__(self, df: pd.DataFrame, target_col: str = 'volume_millions'):
        self.df = df.copy()
        self.target_col = target_col
        self.values = df[target_col].values
    
    def adf_test(self) -> Dict[str, Any]:
        """Augmented Dickey-Fuller test."""
        result = adfuller(self.values, autolag='AIC')
        
        is_stationary = result[1] < 0.05
        
        return {
            'test_name': 'Augmented Dickey-Fuller',
            'test_statistic': round(result[0], 4),
            'p_value': round(result[1], 4),
            'lags_used': result[2],
            'critical_values': {k: round(v, 4) for k, v in result[4].items()},
            'is_stationary': is_stationary,
            'conclusion': 'Stationary' if is_stationary else 'Non-stationary',
            'interpretation': self._interpret_adf(result)
        }
    
    def kpss_test(self) -> Dict[str, Any]:
        """KPSS test (null hypothesis: stationary)."""
        try:
            result = kpss(self.values, regression='c', nlags='auto')
            is_stationary = result[1] > 0.05
        except Exception as e:
            return {'error': str(e)}
        
        return {
            'test_name': 'KPSS',
            'test_statistic': round(result[0], 4),
            'p_value': round(result[1], 4),
            'lags_used': result[2],
            'critical_values': {k: round(v, 4) for k, v in result[3].items()},
            'is_stationary': is_stationary,
            'conclusion': 'Stationary' if is_stationary else 'Non-stationary',
            'interpretation': 'Cannot reject null hypothesis (stationary)' if is_stationary else 'Reject null hypothesis (non-stationary)'
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all stationarity tests."""
        adf_result = self.adf_test()
        kpss_result = self.kpss_test()
        
        stationary_count = sum([
            adf_result.get('is_stationary', False),
            kpss_result.get('is_stationary', False)
        ])
        
        return {
            'adf_test': adf_result,
            'kpss_test': kpss_result,
            'overall_conclusion': {
                'is_stationary': stationary_count >= 1,
                'confidence': 'high' if stationary_count == 2 else 'moderate' if stationary_count == 1 else 'low',
                'recommendation': self._get_differencing_recommendation(stationary_count)
            }
        }
    
    def _interpret_adf(self, result) -> str:
        """Interpret ADF test results."""
        if result[1] < 0.01:
            return f"Strong evidence against unit root (p={result[1]:.4f}). Series is stationary."
        elif result[1] < 0.05:
            return f"Moderate evidence against unit root (p={result[1]:.4f}). Series is stationary."
        else:
            return f"Weak evidence against unit root (p={result[1]:.4f}). Series may be non-stationary."
    
    def _get_differencing_recommendation(self, stationary_count: int) -> str:
        """Get recommendation for differencing."""
        if stationary_count >= 1:
            return "Series may be stationary. Consider d=0 or d=1 for ARIMA."
        else:
            return "Series is non-stationary. Recommend d=1 or d=2 differencing."


class AutocorrelationAnalyzer:
    """Analyze autocorrelation in time series."""
    
    def __init__(self, df: pd.DataFrame, target_col: str = 'volume_millions'):
        self.df = df.copy()
        self.target_col = target_col
        self.values = df[target_col].values
    
    def acf_pacf(self, nlags: int = 24) -> Dict[str, Any]:
        """Calculate ACF and PACF values."""
        acf_values = acf(self.values, nlags=nlags)
        pacf_values = pacf(self.values, nlags=nlags, method='ywm')
        
        conf_int = 1.96 / np.sqrt(len(self.values))
        
        significant_acf = [(i, v) for i, v in enumerate(acf_values[1:], 1) if abs(v) > conf_int]
        significant_pacf = [(i, v) for i, v in enumerate(pacf_values[1:], 1) if abs(v) > conf_int]
        
        return {
            'acf': {
                'values': acf_values.tolist(),
                'significant_lags': significant_acf[:10],
                'confidence_interval': conf_int,
                'decay_rate': 'slow' if abs(acf_values[12]) > 0.5 else 'fast'
            },
            'pacf': {
                'values': pacf_values.tolist(),
                'significant_lags': significant_pacf[:10],
                'confidence_interval': conf_int
            },
            'interpretation': self._interpret_acf_pacf(acf_values, pacf_values, significant_acf)
        }
    
    def _interpret_acf_pacf(self, acf_vals, pacf_vals, sig_acf) -> Dict[str, str]:
        """Interpret ACF and PACF plots."""
        interpretation = {}
        
        if abs(acf_vals[12]) > 0.5:
            interpretation['seasonality'] = "Strong annual seasonality detected (ACF at lag 12)"
        elif abs(acf_vals[6]) > 0.5:
            interpretation['seasonality'] = "Moderate semi-annual pattern detected"
        
        if all(abs(v) < 0.3 for v in acf_vals[1:13]):
            interpretation['memory'] = "Short memory - recent values have little influence on future"
        elif abs(acf_vals[1]) > 0.8:
            interpretation['memory'] = "Long memory - high persistence in the series"
        
        if pacf_vals[1] > 0.8:
            interpretation['ar_order'] = "Strong AR(1) component suggested"
        elif pacf_vals[1] > 0.5:
            interpretation['ar_order'] = "Moderate AR(1) component suggested"
        
        return interpretation


class DistributionAnalyzer:
    """Analyze probability distribution of time series."""
    
    def __init__(self, df: pd.DataFrame, target_col: str = 'volume_millions'):
        self.df = df.copy()
        self.target_col = target_col
        self.values = df[target_col].dropna().values
    
    def analyze(self) -> Dict[str, Any]:
        """Comprehensive distribution analysis."""
        skewness = stats.skew(self.values)
        kurtosis = stats.kurtosis(self.values)
        
        jarque_bera = stats.jarque_bera(self.values)
        shapiro = stats.shapiro(self.values[:5000] if len(self.values) > 5000 else self.values)
        
        percentiles = np.percentile(self.values, [5, 25, 50, 75, 95])
        
        return {
            'basic_stats': {
                'mean': round(np.mean(self.values), 2),
                'median': round(np.median(self.values), 2),
                'std': round(np.std(self.values), 2),
                'variance': round(np.var(self.values), 2),
                'min': round(np.min(self.values), 2),
                'max': round(np.max(self.values), 2),
                'range': round(np.max(self.values) - np.min(self.values), 2)
            },
            'percentiles': {
                'p5': round(percentiles[0], 2),
                'p25': round(percentiles[1], 2),
                'p50': round(percentiles[2], 2),
                'p75': round(percentiles[3], 2),
                'p95': round(percentiles[4], 2)
            },
            'shape_stats': {
                'skewness': round(skewness, 4),
                'kurtosis': round(kurtosis, 4),
                'skewness_interpretation': self._interpret_skewness(skewness),
                'kurtosis_interpretation': self._interpret_kurtosis(kurtosis)
            },
            'normality_tests': {
                'jarque_bera': {
                    'statistic': round(jarque_bera.statistic, 2),
                    'p_value': round(jarque_bera.pvalue, 4),
                    'is_normal': jarque_bera.pvalue > 0.05
                },
                'shapiro_wilk': {
                    'statistic': round(shapiro.statistic, 4),
                    'p_value': round(shapiro.pvalue, 4),
                    'is_normal': shapiro.pvalue > 0.05
                }
            },
            'coefficient_of_variation': round((np.std(self.values) / np.mean(self.values)) * 100, 2)
        }
    
    def _interpret_skewness(self, skewness: float) -> str:
        """Interpret skewness value."""
        if skewness > 1:
            return "Highly right-skewed (positive skew)"
        elif skewness > 0.5:
            return "Moderately right-skewed"
        elif skewness > -0.5:
            return "Approximately symmetric"
        elif skewness > -1:
            return "Moderately left-skewed"
        else:
            return "Highly left-skewed (negative skew)"
    
    def _interpret_kurtosis(self, kurtosis: float) -> str:
        """Interpret kurtosis value."""
        if kurtosis > 3:
            return "Heavy-tailed (leptokurtic) - more outliers than normal"
        elif kurtosis > 0:
            return "Slightly heavy-tailed"
        elif kurtosis > -3:
            return "Approximately normal tail weight"
        else:
            return "Light-tailed (platykurtic) - fewer outliers than normal"


class EDAGenerator:
    """Generate comprehensive EDA report."""
    
    def __init__(self, df: pd.DataFrame, target_col: str = 'volume_millions'):
        self.df = df.copy()
        self.target_col = target_col
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate complete EDA report."""
        logger.info("Generating EDA report...")
        
        trend_analyzer = TrendAnalyzer(self.df, self.target_col)
        seasonality_analyzer = SeasonalityAnalyzer(self.df, self.target_col)
        stationarity_tester = StationarityTester(self.df, self.target_col)
        autocorrelation_analyzer = AutocorrelationAnalyzer(self.df, self.target_col)
        distribution_analyzer = DistributionAnalyzer(self.df, self.target_col)
        
        report = {
            'summary': {
                'records': len(self.df),
                'date_range': {
                    'start': self.df['date'].min().strftime('%Y-%m'),
                    'end': self.df['date'].max().strftime('%Y-%m')
                },
                'target_column': self.target_col
            },
            'trend_analysis': {
                'rolling_statistics': trend_analyzer.rolling_statistics(),
                'trend_type': trend_analyzer.trend_type()
            },
            'seasonality_analysis': {
                'monthly_patterns': seasonality_analyzer.monthly_patterns(),
                'quarterly_patterns': seasonality_analyzer.quarterly_patterns(),
                'decomposition': seasonality_analyzer.decompose()
            },
            'stationarity_analysis': stationarity_tester.run_all_tests(),
            'autocorrelation_analysis': autocorrelation_analyzer.acf_pacf(),
            'distribution_analysis': distribution_analyzer.analyze()
        }
        
        report['auto_insights'] = self._generate_auto_insights(report)
        
        return report
    
    def _generate_auto_insights(self, report: Dict) -> list:
        """Automatically generate insights from EDA."""
        insights = []
        
        if report['trend_analysis']['trend_type']['trend_type'].startswith('strong'):
            insights.append(f"Strong trend detected: {report['trend_analysis']['trend_type']['interpretation']}")
        
        if report['seasonality_analysis']['monthly_patterns']['strength'] == 'strong':
            peak = report['seasonality_analysis']['monthly_patterns']['peak_months']
            insights.append(f"Strong seasonality with peak in month(s): {peak}")
        
        if not report['stationarity_analysis']['overall_conclusion']['is_stationary']:
            insights.append("Data is non-stationary - consider differencing for ARIMA")
        
        if report['distribution_analysis']['shape_stats']['skewness'] > 0.5:
            insights.append("Right-skewed distribution - log transformation may help")
        
        acf_analysis = report['autocorrelation_analysis']
        if acf_analysis['interpretation'].get('seasonality'):
            insights.append(acf_analysis['interpretation']['seasonality'])
        
        insights.append(f"Strong autocorrelation at lag 1 suggests ARIMA(p>0,1,q) may be appropriate")
        
        return insights


if __name__ == "__main__":
    from backend.data.scraper import fetch_and_store_data
    from backend.preprocessing.cleaner import preprocess_data
    
    df, _ = fetch_and_store_data()
    df_clean, _ = preprocess_data(df)
    
    eda = EDAGenerator(df_clean)
    report = eda.generate_report()
    
    print("=== EDA Report ===")
    print(f"Records: {report['summary']['records']}")
    print(f"Date Range: {report['summary']['date_range']}")
    print(f"\nTrend: {report['trend_analysis']['trend_type']['trend_type']}")
    print(f"Seasonality: {report['seasonality_analysis']['monthly_patterns']['strength']}")
    print(f"Stationary: {report['stationarity_analysis']['overall_conclusion']['is_stationary']}")
    print(f"\nAuto Insights:")
    for insight in report['auto_insights']:
        print(f"  - {insight}")
