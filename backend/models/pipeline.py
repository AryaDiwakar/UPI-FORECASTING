"""
Main Forecasting Pipeline
Orchestrates the entire forecasting workflow.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from datetime import datetime
import json

from backend.data.scraper import fetch_and_store_data
from backend.preprocessing.cleaner import preprocess_data
from backend.analysis.eda import EDAGenerator
from backend.features.engineering import create_features
from backend.models.forecasting import (
    ARIMAModel, SARIMAModel, RidgeRegressionModel, XGBoostModel, 
    RandomForestModel, LSTMModel, AttentionLSTMModel,
    EnsembleModel, ModelMetrics, ModelResult
)
from backend.evaluation.metrics import (
    MetricsCalculator, TimeSeriesCrossValidator, 
    ResidualAnalyzer, ModelComparator, ForecastEvaluator
)
from backend.interpretability.shap_values import generate_interpretability_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ForecastingPipeline:
    """
    Complete forecasting pipeline that orchestrates all components.
    """
    
    def __init__(self, 
                 test_size: int = 12,
                 forecast_horizon: int = 12,
                 sequence_length: int = 6):
        self.test_size = test_size
        self.forecast_horizon = forecast_horizon
        self.sequence_length = sequence_length
        
        self.df_raw = None
        self.df_clean = None
        self.df_featured = None
        self.feature_names = []
        self.eda_report = {}
        
        self.models = {}
        self.model_results = {}
        self.ensemble_result = None
        
        self.pipeline_status = {
            'data_loaded': False,
            'eda_complete': False,
            'features_created': False,
            'models_trained': False,
            'ensemble_created': False
        }
    
    def load_data(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Load and preprocess data."""
        logger.info("Step 1: Loading data...")
        
        self.df_raw, version = fetch_and_store_data(force_refresh)
        self.df_clean, preprocessing_report = preprocess_data(
            self.df_raw, 
            impute_method='interpolate',
            detect_outliers=True
        )
        
        self.pipeline_status['data_loaded'] = True
        
        return {
            'version': version,
            'records': len(self.df_clean),
            'preprocessing': preprocessing_report,
            'date_range': {
                'start': self.df_clean['date'].min().strftime('%Y-%m'),
                'end': self.df_clean['date'].max().strftime('%Y-%m')
            }
        }
    
    def run_eda(self) -> Dict[str, Any]:
        """Run exploratory data analysis."""
        logger.info("Step 2: Running EDA...")
        
        if not self.pipeline_status['data_loaded']:
            raise RuntimeError("Data not loaded. Run load_data() first.")
        
        eda = EDAGenerator(self.df_clean)
        self.eda_report = eda.generate_report()
        
        self.pipeline_status['eda_complete'] = True
        
        return self.eda_report
    
    def create_features(self) -> Dict[str, Any]:
        """Create features for modeling."""
        logger.info("Step 3: Creating features...")
        
        if not self.pipeline_status['data_loaded']:
            raise RuntimeError("Data not loaded. Run load_data() first.")
        
        self.df_featured, self.feature_names = create_features(
            self.df_clean,
            lags=[1, 3, 6, 12],
            rolling_windows=[3, 6, 12],
            include_decomposition=True,
            include_fourier=True
        )
        
        self.pipeline_status['features_created'] = True
        
        return {
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names,
            'data_shape': self.df_featured.shape
        }
    
    def train_models(self, 
                     exclude_anomalies: bool = False,
                     run_deep_learning: bool = True) -> Dict[str, Any]:
        """Train all forecasting models."""
        logger.info("Step 4: Training models...")
        
        if not self.pipeline_status['features_created']:
            raise RuntimeError("Features not created. Run create_features() first.")
        
        if exclude_anomalies and 'is_anomaly' in self.df_featured.columns:
            df_train = self.df_featured[~self.df_featured['is_anomaly']].copy()
        else:
            df_train = self.df_featured
        
        train = df_train.iloc[:-self.test_size]
        test = df_train.iloc[-self.test_size:]
        
        X_train = train[self.feature_names].values
        X_test = test[self.feature_names].values
        y_train = train['volume_millions'].values
        y_test = test['volume_millions'].values
        
        y_series = df_train['volume_millions'].values
        
        results = {}
        
        model_configs = [
            ('ARIMA', ARIMAModel(order=(2, 1, 2)), 'statistical'),
            ('SARIMA', SARIMAModel(), 'statistical'),
            ('Ridge', RidgeRegressionModel(alpha=1.0), 'classical'),
            ('XGBoost', XGBoostModel(n_estimators=100, max_depth=6), 'gradient_boosting'),
            ('RandomForest', RandomForestModel(n_estimators=100, max_depth=10), 'ensemble'),
        ]
        
        for name, model, model_type in model_configs:
            try:
                logger.info(f"  Training {name}...")
                
                if name in ['ARIMA', 'SARIMA']:
                    model.fit(None, y_train)
                    y_pred = model.predict(len(y_test))
                    y_test_actual = y_test
                else:
                    model.fit(X_train, y_train, feature_names=self.feature_names)
                    y_pred = model.predict(X_test)
                    y_test_actual = y_test
                
                metrics = MetricsCalculator.calculate(y_test_actual, y_pred)
                
                residual_analyzer = ResidualAnalyzer(y_test_actual, y_pred)
                residual_analysis = residual_analyzer.analyze()
                
                feature_importance = {}
                if hasattr(model, 'get_feature_importance'):
                    feature_importance = model.get_feature_importance()
                
                results[name] = ModelResult(
                    name=name,
                    model_type=model_type,
                    metrics=ModelMetrics(
                        rmse=metrics['rmse'],
                        mae=metrics['mae'],
                        mape=metrics['mape'],
                        r2=metrics['r2'],
                        std_error=metrics['std_error']
                    ),
                    predictions=y_pred,
                    test_actual=y_test_actual,
                    residuals=y_test_actual - y_pred,
                    feature_importance=feature_importance
                )
                
                self.models[name] = model
                
            except Exception as e:
                logger.error(f"  {name} failed: {e}")
                results[name] = {'error': str(e)}
        
        if run_deep_learning:
            try:
                logger.info("  Training LSTM...")
                lstm = LSTMModel(sequence_length=self.sequence_length, epochs=50)
                
                seq_X, seq_y = self._create_sequences(y_series, self.sequence_length)
                seq_train_end = len(seq_X) - self.test_size
                seq_X_train, seq_X_test = seq_X[:seq_train_end], seq_X[seq_train_end:]
                seq_y_train, seq_y_test = seq_y[:seq_train_end], seq_y[seq_train_end:]
                
                lstm.fit(seq_X_train, seq_y_train, seq_X_test, seq_y_test)
                lstm_pred = lstm.predict(seq_X_test)
                
                metrics = MetricsCalculator.calculate(seq_y_test, lstm_pred)
                
                results['LSTM'] = ModelResult(
                    name='LSTM',
                    model_type='deep_learning',
                    metrics=ModelMetrics(
                        rmse=metrics['rmse'],
                        mae=metrics['mae'],
                        mape=metrics['mape'],
                        r2=metrics['r2'],
                        std_error=metrics['std_error']
                    ),
                    predictions=lstm_pred,
                    test_actual=seq_y_test,
                    residuals=seq_y_test - lstm_pred
                )
                
                self.models['LSTM'] = lstm
                
            except Exception as e:
                logger.error(f"  LSTM failed: {e}")
        
        self.model_results = results
        self.pipeline_status['models_trained'] = True
        
        return {
            'models_trained': len([r for r in results if 'error' not in r]),
            'results': {k: v.to_dict() if isinstance(v, ModelResult) else v for k, v in results.items()}
        }
    
    def create_ensemble(self) -> Dict[str, Any]:
        """Create weighted ensemble of models."""
        logger.info("Step 5: Creating ensemble...")
        
        if not self.pipeline_status['models_trained']:
            raise RuntimeError("Models not trained. Run train_models() first.")
        
        successful_results = [
            r for r in self.model_results.values() 
            if isinstance(r, ModelResult)
        ]
        
        if len(successful_results) < 2:
            return {'error': 'Need at least 2 models for ensemble'}
        
        ml_model_names = ['ARIMA', 'SARIMA', 'Ridge', 'XGBoost', 'RandomForest']
        ensemble_models = [r.name for r in successful_results if r.name in ml_model_names]
        ensemble_weights = []
        ensemble_model_objs = []
        
        for r in successful_results:
            if r.name in ml_model_names:
                ensemble_weights.append(1.0 / (r.metrics.rmse + 1e-6))
                ensemble_model_objs.append(self.models[r.name])
        
        if sum(ensemble_weights) > 0:
            ensemble_weights = [w / sum(ensemble_weights) for w in ensemble_weights]
        
        if len(ensemble_model_objs) < 2:
            return {'error': 'Need at least 2 ML models for ensemble'}
        
        ensemble = EnsembleModel(ensemble_model_objs, ensemble_weights)
        
        test = self.df_featured.iloc[-self.test_size:]
        X_test = test[self.feature_names].values
        y_test = test['volume_millions'].values
        
        y_pred = ensemble.predict(X_test)
        metrics = MetricsCalculator.calculate(y_test, y_pred)
        
        self.ensemble_result = ModelResult(
            name='Ensemble',
            model_type='ensemble',
            metrics=ModelMetrics(
                rmse=metrics['rmse'],
                mae=metrics['mae'],
                mape=metrics['mape'],
                r2=metrics['r2'],
                std_error=metrics['std_error']
            ),
            predictions=y_pred,
            test_actual=y_test,
            residuals=y_test - y_pred
        )
        
        self.models['Ensemble'] = ensemble
        self.pipeline_status['ensemble_created'] = True
        
        return {
            'ensemble_metrics': metrics,
            'weights': dict(zip(ensemble_models, ensemble_weights))
        }
    
    def generate_forecast(self, model_name: str = 'best') -> Dict[str, Any]:
        """Generate forecast for specified model."""
        logger.info(f"Step 6: Generating forecast for {model_name}...")
        
        if not self.pipeline_status['models_trained']:
            raise RuntimeError("Models not trained. Run train_models() first.")
        
        if model_name == 'best':
            comparator = ModelComparator()
            for name, result in self.model_results.items():
                if isinstance(result, ModelResult):
                    comparator.add_result(name, result.metrics.to_dict())
            
            summary = comparator.get_summary()
            model_name = summary.get('best_model', 'Ensemble')
        
        model = self.models.get(model_name)
        if model is None:
            return {'error': f'Model {model_name} not found'}
        
        last_values = self.df_featured['volume_millions'].values
        
        if model_name in ['LSTM', 'AttentionLSTM']:
            forecast = model.forecast(last_values, self.forecast_horizon)
        else:
            forecast = model.forecast(last_values, self.forecast_horizon)
        
        last_date = self.df_featured['date'].iloc[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=self.forecast_horizon,
            freq='MS'
        )
        
        std_error = self.model_results.get(model_name, ModelResult('','',ModelMetrics(0,0,0,0,0),np.array([]),np.array([]))).metrics.std_error
        
        conf_factor = 1.96 * std_error / 100
        
        return {
            'model': model_name,
            'forecast_dates': [d.strftime('%Y-%m') for d in forecast_dates],
            'predictions': [round(p, 2) for p in forecast],
            'confidence_lower': [round(p * (1 - conf_factor), 2) for p in forecast],
            'confidence_upper': [round(p * (1 + conf_factor), 2) for p in forecast]
        }
    
    def get_model_comparison(self) -> Dict[str, Any]:
        """Get model comparison summary."""
        if not self.pipeline_status['models_trained']:
            raise RuntimeError("Models not trained. Run train_models() first.")
        
        comparator = ModelComparator()
        
        for name, result in self.model_results.items():
            if isinstance(result, ModelResult):
                comparator.add_result(name, result.metrics.to_dict())
        
        summary = comparator.get_summary()
        
        return summary
    
    def get_interpretability_report(self, model_name: str = None) -> Dict[str, Any]:
        """Get feature importance and explanation report."""
        if not self.pipeline_status['models_trained']:
            raise RuntimeError("Models not trained. Run train_models() first.")
        
        if model_name is None:
            result = self.get_model_comparison()
            model_name = result.get('best_model')
        
        result = self.model_results.get(model_name)
        if result is None or not isinstance(result, ModelResult):
            return {'error': f'Model {model_name} not found'}
        
        if result.feature_importance:
            return generate_interpretability_report(
                result.feature_importance,
                self.feature_names
            )
        
        return {'error': 'No feature importance available for this model'}
    
    def run_full_pipeline(self, 
                         exclude_anomalies: bool = False,
                         run_deep_learning: bool = True) -> Dict[str, Any]:
        """Run the complete forecasting pipeline."""
        logger.info("=" * 50)
        logger.info("Starting UPI Forecasting Pipeline")
        logger.info("=" * 50)
        
        start_time = datetime.now()
        
        data_info = self.load_data()
        logger.info(f"Loaded {data_info['records']} records")
        
        self.run_eda()
        logger.info("EDA complete")
        
        feature_info = self.create_features()
        logger.info(f"Created {feature_info['feature_count']} features")
        
        train_results = self.train_models(exclude_anomalies, run_deep_learning)
        logger.info(f"Trained {train_results['models_trained']} models")
        
        ensemble_info = self.create_ensemble()
        logger.info("Ensemble created")
        
        comparison = self.get_model_comparison()
        best_model = comparison.get('best_model')
        
        if best_model in ['Ridge', 'XGBoost', 'RandomForest']:
            forecast_model = 'SARIMA'
        else:
            forecast_model = best_model
        
        forecast = self.generate_forecast(forecast_model)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("=" * 50)
        logger.info(f"Pipeline complete in {duration:.2f} seconds")
        logger.info(f"Best model: {best_model}")
        logger.info("=" * 50)
        
        return {
            'data_info': data_info,
            'feature_info': feature_info,
            'eda_insights': self.eda_report.get('auto_insights', []),
            'models_trained': train_results['models_trained'],
            'model_comparison': comparison,
            'ensemble': ensemble_info,
            'best_model': best_model,
            'forecast': forecast,
            'duration_seconds': round(duration, 2)
        }
    
    def _create_sequences(self, values: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM models."""
        X, y = [], []
        for i in range(seq_length, len(values)):
            X.append(values[i - seq_length:i])
            y.append(values[i])
        return np.array(X), np.array(y)


def run_forecast_pipeline(**kwargs) -> Dict[str, Any]:
    """Convenience function to run the pipeline."""
    pipeline = ForecastingPipeline(**kwargs)
    return pipeline.run_full_pipeline()


if __name__ == "__main__":
    results = run_forecast_pipeline()
    
    print("\n=== Pipeline Results ===")
    print(f"Models trained: {results['models_trained']}")
    print(f"Best model: {results['best_model']}")
    print(f"Duration: {results['duration_seconds']}s")
    
    print("\n=== Model Rankings ===")
    for r in results['model_comparison']['rankings']:
        print(f"{r['rank']}. {r['model']}: RMSE={r['rmse']}, MAPE={r['mape']}%")
    
    print("\n=== Top EDA Insights ===")
    for insight in results['eda_insights'][:5]:
        print(f"  - {insight}")
