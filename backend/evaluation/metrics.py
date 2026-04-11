"""
Evaluation Module
Handles model evaluation, cross-validation, and residual analysis.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    model_name: str
    metrics: Dict[str, float]
    cross_validation: Dict[str, Any]
    residual_analysis: Dict[str, Any]
    is_best: bool = False


class MetricsCalculator:
    """Calculate forecasting metrics."""
    
    @staticmethod
    def calculate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate all regression metrics."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        mask = y_true != 0
        if mask.sum() == 0:
            mape = 0.0
        else:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        std_error = np.std(np.abs(y_true - y_pred))
        
        return {
            'rmse': round(rmse, 4),
            'mae': round(mae, 4),
            'mape': round(mape, 2),
            'r2': round(r2, 4),
            'std_error': round(std_error, 4)
        }
    
    @staticmethod
    def calculate_percentage_improvement(baseline: float, improved: float) -> float:
        """Calculate percentage improvement."""
        if baseline == 0:
            return 0.0
        return round(((baseline - improved) / baseline) * 100, 2)


class TimeSeriesCrossValidator:
    """Time series cross-validation with rolling window."""
    
    def __init__(self, n_splits: int = 5, test_size: int = 12):
        self.n_splits = n_splits
        self.test_size = test_size
    
    def split(self, values: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test splits for time series."""
        n = len(values)
        splits = []
        
        for i in range(self.n_splits):
            train_end = n - self.test_size * (self.n_splits - i)
            test_end = train_end + self.test_size
            
            if train_end < self.test_size * 2 or test_end > n:
                continue
            
            train = values[:train_end]
            test = values[train_end:test_end]
            splits.append((train, test))
        
        return splits
    
    def cross_validate(self, model_class, values: np.ndarray, 
                       feature_creator=None, **model_kwargs) -> Dict[str, Any]:
        """Perform cross-validation on time series."""
        splits = self.split(values)
        
        fold_results = []
        for fold, (train, test) in enumerate(splits):
            try:
                if feature_creator:
                    X_train, y_train = feature_creator(train)
                    X_test, y_test = feature_creator(test)
                    model = model_class(**model_kwargs)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                else:
                    model = model_class(**model_kwargs)
                    model.fit(None, train)
                    y_pred = model.forecast(train, len(test))
                    y_test = test
                
                metrics = MetricsCalculator.calculate(y_test, y_pred)
                fold_results.append({
                    'fold': fold + 1,
                    'train_size': len(train),
                    'test_size': len(test),
                    'metrics': metrics
                })
                
            except Exception as e:
                logger.warning(f"Fold {fold + 1} failed: {e}")
                continue
        
        if not fold_results:
            return {'error': 'All folds failed'}
        
        avg_metrics = {
            'mean_rmse': np.mean([f['metrics']['rmse'] for f in fold_results]),
            'std_rmse': np.std([f['metrics']['rmse'] for f in fold_results]),
            'mean_mae': np.mean([f['metrics']['mae'] for f in fold_results]),
            'mean_mape': np.mean([f['metrics']['mape'] for f in fold_results]),
            'n_splits': len(fold_results)
        }
        
        return {
            'fold_results': fold_results,
            'summary': avg_metrics
        }


class ResidualAnalyzer:
    """Analyze model residuals for diagnostic purposes."""
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.residuals = y_true - y_pred
    
    def basic_stats(self) -> Dict[str, float]:
        """Calculate basic residual statistics."""
        return {
            'mean': round(np.mean(self.residuals), 4),
            'std': round(np.std(self.residuals), 4),
            'min': round(np.min(self.residuals), 4),
            'max': round(np.max(self.residuals), 4),
            'median': round(np.median(self.residuals), 4)
        }
    
    def _interpret_normality(self, p_value: float, skewness: float, kurtosis: float) -> str:
        """Interpret normality test results."""
        if p_value > 0.05 and abs(skewness) < 1 and abs(kurtosis) < 3:
            return "Residuals appear normally distributed"
        elif p_value > 0.05:
            return "Residuals pass statistical test but may not be perfectly normal"
        elif abs(skewness) > 2 or abs(kurtosis) > 7:
            return "Residuals show significant non-normality"
        else:
            return "Residuals may deviate from normality but acceptable for forecasting"
    
    def normality_tests(self) -> Dict[str, Any]:
        """Test if residuals are normally distributed."""
        if len(self.residuals) < 3:
            return {'error': 'Insufficient data for normality tests'}
        
        shapiro = stats.shapiro(self.residuals)
        jarque_bera = stats.jarque_bera(self.residuals)
        
        skewness = stats.skew(self.residuals)
        kurtosis = stats.kurtosis(self.residuals)
        
        return {
            'shapiro_wilk': {
                'statistic': round(shapiro.statistic, 4),
                'p_value': round(shapiro.pvalue, 4),
                'is_normal': shapiro.pvalue > 0.05
            },
            'jarque_bera': {
                'statistic': round(jarque_bera.statistic, 4),
                'p_value': round(jarque_bera.pvalue, 4),
                'is_normal': jarque_bera.pvalue > 0.05
            },
            'skewness': round(skewness, 4),
            'kurtosis': round(kurtosis, 4),
            'interpretation': self._interpret_normality(shapiro.pvalue, skewness, kurtosis)
        }
    
    def autocorrelation_test(self, nlags: int = 12) -> Dict[str, Any]:
        """Test for autocorrelation in residuals using Ljung-Box test."""
        if len(self.residuals) < nlags:
            return {'error': 'Insufficient data for autocorrelation test'}
        
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            
            lb_result = acorr_ljungbox(self.residuals, lags=[nlags], return_df=True)
            lb_stat = lb_result['lb_stat'].values[0]
            lb_pvalue = lb_result['lb_pvalue'].values[0]
            
            return {
                'ljung_box': {
                    'statistic': round(float(lb_stat), 4),
                    'p_value': round(float(lb_pvalue), 4),
                    'has_autocorrelation': lb_pvalue < 0.05
                },
                'interpretation': 'Residuals are independent' if lb_pvalue > 0.05 else 'Residuals show autocorrelation - model may be misspecified'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def heteroscedasticity_test(self) -> Dict[str, Any]:
        """Test for heteroscedasticity (non-constant variance)."""
        if len(self.residuals) < 10:
            return {'error': 'Insufficient data for heteroscedasticity test'}
        
        squared_residuals = self.residuals ** 2
        x = np.arange(len(squared_residuals))
        
        correlation = np.corrcoef(x, squared_residuals)[0, 1]
        
        spearman = stats.spearmanr(x, squared_residuals)
        
        return {
            'breusch_pagan_simple': {
                'correlation': round(correlation, 4),
                'is_heteroscedastic': abs(correlation) > 0.3
            },
            'spearman_test': {
                'rho': round(spearman.correlation, 4),
                'p_value': round(spearman.pvalue, 4),
                'is_heteroscedastic': spearman.pvalue < 0.05
            },
            'interpretation': 'Constant variance' if abs(correlation) < 0.3 else 'Variance changes over time'
        }
    
    def runs_test(self) -> Dict[str, Any]:
        """Run test for randomness (sign changes)."""
        if len(self.residuals) < 10:
            return {'error': 'Insufficient data for runs test'}
        
        median = np.median(self.residuals)
        signs = (self.residuals > median).astype(int)
        
        runs = 1
        for i in range(1, len(signs)):
            if signs[i] != signs[i-1]:
                runs += 1
        
        n1 = np.sum(signs == 1)
        n2 = np.sum(signs == 0)
        
        expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
        variance = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
        
        if variance > 0:
            z = (runs - expected_runs) / np.sqrt(variance)
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        else:
            z, p_value = 0, 1
        
        return {
            'runs': runs,
            'expected_runs': round(expected_runs, 2),
            'z_statistic': round(z, 4),
            'p_value': round(p_value, 4),
            'is_random': p_value > 0.05,
            'interpretation': 'Residuals appear random' if p_value > 0.05 else 'Residuals show patterns'
        }
    
    def analyze(self) -> Dict[str, Any]:
        """Complete residual analysis."""
        return {
            'basic_stats': self.basic_stats(),
            'normality_tests': self.normality_tests(),
            'autocorrelation': self.autocorrelation_test(),
            'heteroscedasticity': self.heteroscedasticity_test(),
            'runs_test': self.runs_test()
        }


class ModelComparator:
    """Compare multiple models and rank them."""
    
    def __init__(self):
        self.results: Dict[str, Dict[str, Any]] = {}
        self.best_model: Optional[str] = None
    
    def add_result(self, name: str, metrics: Dict[str, float], 
                   cross_val: Dict[str, Any] = None,
                   residual_analysis: Dict[str, Any] = None):
        """Add a model's evaluation results."""
        self.results[name] = {
            'metrics': metrics,
            'cross_validation': cross_val or {},
            'residual_analysis': residual_analysis or {}
        }
    
    def rank(self, metric: str = 'rmse') -> List[Dict[str, Any]]:
        """Rank models by specified metric (lower is better)."""
        ranked = []
        
        for name, result in self.results.items():
            model_metrics = result['metrics']
            cv_metrics = result.get('cross_validation', {}).get('summary', {})
            
            ranked.append({
                'rank': 0,
                'model': name,
                'rmse': model_metrics.get('rmse', float('inf')),
                'mae': model_metrics.get('mae', float('inf')),
                'mape': model_metrics.get('mape', float('inf')),
                'r2': model_metrics.get('r2', -float('inf')),
                'cv_rmse_mean': cv_metrics.get('mean_rmse', float('inf')),
                'cv_rmse_std': cv_metrics.get('std_rmse', 0)
            })
        
        ranked = sorted(ranked, key=lambda x: x[metric])
        
        for i, r in enumerate(ranked):
            r['rank'] = i + 1
        
        if ranked:
            ranked[0]['is_best'] = True
            self.best_model = ranked[0]['model']
        
        return ranked
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comparison summary."""
        if not self.results:
            return {'error': 'No results to compare'}
        
        ranked = self.rank()
        best = ranked[0] if ranked else None
        worst = ranked[-1] if ranked else None
        
        improvement = 0
        if best and worst and worst['rmse'] > 0:
            improvement = MetricsCalculator.calculate_percentage_improvement(
                worst['rmse'], best['rmse']
            )
        
        return {
            'best_model': best['model'] if best else None,
            'best_rmse': best['rmse'] if best else None,
            'worst_model': worst['model'] if worst else None,
            'worst_rmse': worst['rmse'] if worst else None,
            'improvement_over_worst': improvement,
            'rankings': ranked
        }
    
    def get_best_features(self) -> Dict[str, Any]:
        """Identify best model and its strengths."""
        if not self.results:
            return {}
        
        ranked = self.rank()
        best_name = ranked[0]['model'] if ranked else None
        best_result = self.results.get(best_name, {}) if best_name else {}
        
        return {
            'best_model': best_name,
            'metrics': best_result.get('metrics', {}),
            'cross_validation': best_result.get('cross_validation', {}),
            'residual_quality': best_result.get('residual_analysis', {}).get('normality_tests', {})
        }


class ForecastEvaluator:
    """Evaluate forecasts with confidence intervals."""
    
    @staticmethod
    def calculate_confidence_interval(predictions: np.ndarray,
                                     residuals: np.ndarray,
                                     confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate prediction confidence intervals."""
        std = np.std(residuals)
        z = 1.96 if confidence == 0.95 else 2.576
        
        margin = z * std
        lower = predictions - margin
        upper = predictions + margin
        
        return lower, upper
    
    @staticmethod
    def evaluate_forecast(y_actual: np.ndarray, 
                          y_predicted: np.ndarray,
                          confidence_level: float = 0.95) -> Dict[str, Any]:
        """Complete forecast evaluation."""
        metrics = MetricsCalculator.calculate(y_actual, y_predicted)
        residuals = y_actual - y_predicted
        
        residual_analyzer = ResidualAnalyzer(y_actual, y_predicted)
        residual_analysis = residual_analyzer.analyze()
        
        lower, upper = ForecastEvaluator.calculate_confidence_interval(
            y_predicted, residuals, confidence_level
        )
        
        coverage = np.mean((y_actual >= lower) & (y_actual <= upper)) * 100
        
        return {
            'metrics': metrics,
            'residual_analysis': residual_analysis,
            'confidence_intervals': {
                'lower': lower.tolist(),
                'upper': upper.tolist(),
                'coverage_pct': round(coverage, 2)
            },
            'interpretation': ForecastEvaluator._interpret_forecast(
                metrics, residual_analysis, coverage
            )
        }
    
    @staticmethod
    def _interpret_forecast(metrics: Dict, residual_analysis: Dict, coverage: float) -> str:
        """Generate interpretation of forecast quality."""
        interpretations = []
        
        if metrics['r2'] > 0.9:
            interpretations.append("Excellent fit (R² > 0.9)")
        elif metrics['r2'] > 0.8:
            interpretations.append("Good fit (R² > 0.8)")
        elif metrics['r2'] > 0.6:
            interpretations.append("Moderate fit (R² > 0.6)")
        else:
            interpretations.append("Poor fit (R² < 0.6)")
        
        if metrics['mape'] < 5:
            interpretations.append("High accuracy (MAPE < 5%)")
        elif metrics['mape'] < 10:
            interpretations.append("Good accuracy (MAPE < 10%)")
        
        if coverage > 90:
            interpretations.append(f"Reliable confidence intervals ({coverage:.0f}% coverage)")
        
        normality = residual_analysis.get('normality_tests', {})
        if normality.get('shapiro_wilk', {}).get('is_normal'):
            interpretations.append("Residuals are normally distributed")
        
        autocorr = residual_analysis.get('autocorrelation', {})
        if not autocorr.get('ljung_box', {}).get('has_autocorrelation', True):
            interpretations.append("No significant autocorrelation in residuals")
        
        return ". ".join(interpretations)


def evaluate_models(df: pd.DataFrame, 
                  feature_names: List[str],
                  target_col: str = 'volume_millions',
                  test_size: int = 12) -> Dict[str, Any]:
    """Evaluate all models on the dataset."""
    from models.forecasting import (
        ARIMAModel, RidgeRegressionModel, XGBoostModel, 
        RandomForestModel, LSTMModel, AttentionLSTMModel
    )
    
    train = df.iloc[:-test_size]
    test = df.iloc[-test_size:]
    
    X_train = train[feature_names].values
    X_test = test[feature_names].values
    y_train = train[target_col].values
    y_test = test[target_col].values
    
    results = {}
    comparator = ModelComparator()
    
    model_configs = [
        ('ARIMA', ARIMAModel(order=(2, 1, 2))),
        ('Ridge', RidgeRegressionModel(alpha=1.0)),
        ('XGBoost', XGBoostModel(n_estimators=100)),
        ('RandomForest', RandomForestModel(n_estimators=100)),
    ]
    
    for name, model in model_configs:
        try:
            if name in ['ARIMA']:
                model.fit(None, y_train)
                y_pred = model.predict(X_test)
            else:
                model.fit(X_train, y_train, feature_names=feature_names)
                y_pred = model.predict(X_test)
            
            metrics = MetricsCalculator.calculate(y_test, y_pred)
            
            residual_analyzer = ResidualAnalyzer(y_test, y_pred)
            residual_analysis = residual_analyzer.analyze()
            
            results[name] = {
                'metrics': metrics,
                'predictions': y_pred.tolist(),
                'test_actual': y_test.tolist(),
                'residual_analysis': residual_analysis,
                'feature_importance': model.get_feature_importance() if hasattr(model, 'get_feature_importance') else {}
            }
            
            comparator.add_result(name, metrics, residual_analysis=residual_analysis)
            
        except Exception as e:
            logger.error(f"Model {name} failed: {e}")
            results[name] = {'error': str(e)}
    
    comparison = comparator.get_summary()
    
    return {
        'model_results': results,
        'comparison': comparison
    }


if __name__ == "__main__":
    from backend.data.scraper import fetch_and_store_data
    from backend.preprocessing.cleaner import preprocess_data
    from backend.features.engineering import create_features
    
    df, _ = fetch_and_store_data()
    df_clean, _ = preprocess_data(df)
    df_featured, feature_names = create_features(df_clean)
    
    print("Running model evaluation...")
    results = evaluate_models(df_featured, feature_names)
    
    print("\n=== Model Comparison ===")
    for r in results['comparison']['rankings']:
        print(f"{r['rank']}. {r['model']}: RMSE={r['rmse']}, MAPE={r['mape']}%")
