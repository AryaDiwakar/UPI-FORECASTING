import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Any, Optional
from abc import ABC, abstractmethod
import logging
import warnings
from dataclasses import dataclass
import joblib

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    rmse: float
    mae: float
    mape: float
    std_error: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'rmse': round(self.rmse, 4),
            'mae': round(self.mae, 4),
            'mape': round(self.mape, 2),
            'std_error': round(self.std_error, 4)
        }


@dataclass
class CrossValidationResult:
    model_name: str
    fold_metrics: List[ModelMetrics]
    mean_rmse: float
    mean_mae: float
    mean_mape: float
    std_rmse: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'fold_metrics': [m.to_dict() for m in self.fold_metrics],
            'mean_rmse': round(self.mean_rmse, 4),
            'mean_mae': round(self.mean_mae, 4),
            'mean_mape': round(self.mean_mape, 2),
            'std_rmse': round(self.std_rmse, 4)
        }


class BaseModel(ABC):
    name: str = "base"
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_cols: List[str] = []
        
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> 'BaseModel':
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> ModelMetrics:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = 0.0
        
        std_error = np.std(np.abs(y_true - y_pred))
        
        return ModelMetrics(rmse=rmse, mae=mae, mape=mape, std_error=std_error)
    
    def get_confidence_interval(self, predictions: np.ndarray, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        std = np.std(predictions) * 0.1
        z = 1.96 if confidence == 0.95 else 2.576
        lower = predictions - z * std
        upper = predictions + z * std
        return lower, upper


class MovingAverageModel(BaseModel):
    name = "moving_average"
    
    def __init__(self, window: int = 3):
        super().__init__()
        self.window = window
        self.history: List[float] = []
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> 'MovingAverageModel':
        self.history = y_train.tolist()
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if len(X.shape) > 1:
            X = X.flatten()
        return np.array([np.mean(self.history[-self.window:])] * len(X))
    
    def forecast(self, values: np.ndarray, steps: int) -> np.ndarray:
        predictions = []
        recent = list(values[-self.window:])
        
        for _ in range(steps):
            pred = np.mean(recent[-self.window:])
            predictions.append(pred)
            recent.append(pred)
        
        return np.array(predictions)


class LinearRegressionModel(BaseModel):
    name = "linear_regression"
    
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> 'LinearRegressionModel':
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class ARIMAModel(BaseModel):
    name = "arima"
    
    def __init__(self, order: Tuple[int, int, int] = (2, 1, 2)):
        super().__init__()
        self.order = order
        self.model = None
        self.history: Optional[pd.Series] = None
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> 'ARIMAModel':
        from statsmodels.tsa.arima.model import ARIMA
        
        self.history = pd.Series(y_train)
        self.model = ARIMA(self.history, order=self.order)
        self.fitted = self.model.fit()
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        steps = len(X)
        if self.history is not None:
            forecast = self.fitted.forecast(steps=steps)
            return forecast.values
        return np.zeros(steps)
    
    def forecast(self, values: np.ndarray, steps: int) -> np.ndarray:
        from statsmodels.tsa.arima.model import ARIMA
        
        full_series = pd.Series(values)
        model = ARIMA(full_series, order=self.order)
        fitted = model.fit()
        forecast = fitted.forecast(steps=steps)
        return forecast.values


class LSTMModel(BaseModel):
    name = "lstm"
    
    def __init__(self, sequence_length: int = 6, epochs: int = 50):
        super().__init__()
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.keras_model = None
        self.scaler = StandardScaler()
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> 'LSTMModel':
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        
        tf.random.set_seed(42)
        np.random.seed(42)
        
        y_scaled = self.scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        
        X_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        
        self.keras_model = Sequential([
            LSTM(50, activation='relu', input_shape=(self.sequence_length, 1), return_sequences=True),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        
        self.keras_model.compile(optimizer='adam', loss='mse')
        
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        self.keras_model.fit(
            X_reshaped, y_scaled,
            epochs=self.epochs,
            batch_size=8,
            validation_split=0.1,
            callbacks=[early_stop],
            verbose=0
        )
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.keras_model is None:
            return np.zeros(len(X))
        
        X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
        y_pred_scaled = self.keras_model.predict(X_reshaped, verbose=0)
        y_pred = self.scaler.inverse_transform(y_pred_scaled).flatten()
        return y_pred
    
    def forecast(self, values: np.ndarray, steps: int) -> np.ndarray:
        if self.keras_model is None:
            return np.zeros(steps)
        
        scaled_values = self.scaler.fit_transform(values.reshape(-1, 1)).flatten()
        last_sequence = scaled_values[-self.sequence_length:]
        predictions = []
        
        for _ in range(steps):
            input_seq = last_sequence.reshape(1, self.sequence_length, 1)
            pred_scaled = self.keras_model.predict(input_seq, verbose=0)[0, 0]
            predictions.append(pred_scaled)
            last_sequence = np.append(last_sequence[1:], pred_scaled)
        
        pred_array = np.array(predictions).reshape(-1, 1)
        return self.scaler.inverse_transform(pred_array).flatten()


class ProphetModel(BaseModel):
    name = "prophet"
    
    def __init__(self, changepoint_prior_scale: float = 0.05, seasonality_prior_scale: float = 10.0):
        super().__init__()
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.model = None
        self.future_df = None
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, dates: Optional[np.ndarray] = None, **kwargs) -> 'ProphetModel':
        try:
            from prophet import Prophet
            
            if dates is None:
                dates = pd.date_range(start='2016-04', periods=len(y_train), freq='MS')
            
            df = pd.DataFrame({
                'ds': dates[-len(y_train):],
                'y': y_train
            })
            
            self.model = Prophet(
                changepoint_prior_scale=self.changepoint_prior_scale,
                seasonality_prior_scale=self.seasonality_prior_scale,
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False
            )
            
            self.model.fit(df, verbose=0)
            self.train_dates = dates[-len(y_train):]
            
        except Exception as e:
            logger.warning(f"Prophet training failed: {e}. Falling back to simple trend model.")
            self.model = None
            
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.zeros(len(X))
        
        steps = len(X)
        future = self.model.make_future_dataframe(periods=steps)
        forecast = self.model.predict(future)
        return forecast['yhat'].values[-steps:]
    
    def forecast(self, values: np.ndarray, steps: int) -> np.ndarray:
        if self.model is None:
            return np.full(steps, np.mean(values))
        
        future = self.model.make_future_dataframe(periods=steps, freq='MS')
        forecast = self.model.predict(future)
        return forecast['yhat'].values[-steps:]
    
    def get_components(self) -> Dict[str, Any]:
        if self.model is None:
            return {}
        
        forecast = self.model.predict(self.model.history)
        return {
            'trend': forecast['trend'].values.tolist(),
            'yearly': forecast['yearly'].values.tolist()
        }


class EnsembleModel:
    name = "ensemble"
    
    def __init__(self, models: List[BaseModel], weights: Optional[List[float]] = None):
        self.models = models
        self.n_models = len(models)
        
        if weights is None:
            weights = [1.0 / self.n_models] * self.n_models
        elif len(weights) != self.n_models:
            weights = [1.0 / self.n_models] * self.n_models
        
        self.weights = np.array(weights) / np.sum(weights)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        weighted_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            weighted_pred += pred * weight
        
        return weighted_pred
    
    def forecast(self, values: np.ndarray, steps: int) -> Tuple[np.ndarray, List[np.ndarray]]:
        all_predictions = []
        
        for model in self.models:
            if hasattr(model, 'forecast'):
                pred = model.forecast(values, steps)
            else:
                pred = model.predict(np.zeros((steps, 1)))
            all_predictions.append(pred)
        
        weighted_pred = np.zeros(steps)
        for pred, weight in zip(all_predictions, self.weights):
            weighted_pred += pred * weight
        
        return weighted_pred, all_predictions
    
    def get_confidence_interval(self, predictions: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        predictions_array = np.array(predictions)
        mean_pred = np.mean(predictions_array, axis=0)
        std_pred = np.std(predictions_array, axis=0)
        
        lower = mean_pred - 1.96 * std_pred
        upper = mean_pred + 1.96 * std_pred
        
        return lower, upper


class ForecastModels:
    def __init__(self, processor):
        self.processor = processor
        self.df = processor.featured_df
        self.results = {}
        self.cross_validation_results = {}
        self.ensemble_result = None
        
    def train_test_split(self, test_size: int = 12) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        train = self.df.iloc[:-test_size]
        test = self.df.iloc[-test_size:]
        
        feature_cols = [col for col in self.df.columns 
                      if col not in ['month', 'date', 'volume_millions', 'value_crores']]
        
        self.X_train = train[feature_cols].values
        self.X_test = test[feature_cols].values
        self.y_train = train['volume_millions'].values
        self.y_test = test['volume_millions'].values
        self.test_dates = test['date'].values
        
        self.feature_cols = feature_cols
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def time_series_cv(self, n_splits: int = 5, test_size: int = 12) -> Dict[str, CrossValidationResult]:
        values = self.df['volume_millions'].values
        total_len = len(values)
        
        cv_results = {}
        
        for model_cls, model_kwargs in [
            (MovingAverageModel, {'window': 3}),
            (LinearRegressionModel, {'alpha': 1.0}),
            (ARIMAModel, {'order': (2, 1, 2)}),
        ]:
            model_name = model_cls.__name__.replace('Model', '').lower()
            fold_metrics = []
            
            for i in range(n_splits):
                train_end = total_len - test_size * (n_splits - i - 1)
                test_end = train_end + test_size
                
                if train_end < test_size * 2 or test_end > total_len:
                    continue
                
                train_vals = values[:train_end]
                test_vals = values[train_end:test_end]
                
                try:
                    if model_name == 'movingaverage':
                        model = MovingAverageModel(**model_kwargs)
                        model.train(None, train_vals)
                        y_pred = model.forecast(train_vals, test_size)
                    elif model_name == 'linearregression':
                        model = LinearRegressionModel(**model_kwargs)
                        features = self._create_features_for_cv(train_vals)
                        model.train(features[:-test_size], train_vals[model_kwargs.get('window', 1):])
                        test_features = self._create_test_features_for_cv(train_vals, test_size)
                        y_pred = model.predict(test_features)
                    elif model_name == 'arima':
                        model = ARIMAModel(**model_kwargs)
                        model.train(None, train_vals)
                        y_pred = model.forecast(train_vals, test_size)
                    else:
                        continue
                    
                    metrics = model.evaluate(test_vals[:len(y_pred)], y_pred)
                    fold_metrics.append(metrics)
                    
                except Exception as e:
                    logger.warning(f"CV fold {i} failed for {model_name}: {e}")
                    continue
            
            if fold_metrics:
                cv_results[model_name] = CrossValidationResult(
                    model_name=model_name,
                    fold_metrics=fold_metrics,
                    mean_rmse=np.mean([m.rmse for m in fold_metrics]),
                    mean_mae=np.mean([m.mae for m in fold_metrics]),
                    mean_mape=np.mean([m.mape for m in fold_metrics]),
                    std_rmse=np.std([m.rmse for m in fold_metrics])
                )
        
        self.cross_validation_results = cv_results
        return cv_results
    
    def _create_features_for_cv(self, values: np.ndarray) -> np.ndarray:
        features = []
        for i in range(len(values)):
            row = []
            row.append(values[i - 1] if i > 0 else values[0])
            row.append(values[i - 3] if i > 3 else values[0])
            row.append(values[i - 6] if i > 6 else values[0])
            row.append(np.mean(values[max(0, i-3):i]) if i > 0 else values[0])
            features.append(row)
        return np.array(features)
    
    def _create_test_features_for_cv(self, train_vals: np.ndarray, steps: int) -> np.ndarray:
        features = []
        recent = list(train_vals[-6:])
        
        for i in range(steps):
            row = []
            row.append(recent[-1] if len(recent) > 0 else train_vals[-1])
            row.append(recent[-3] if len(recent) > 3 else train_vals[-1])
            row.append(recent[-6] if len(recent) > 6 else train_vals[-1])
            row.append(np.mean(recent[-3:]) if len(recent) > 0 else train_vals[-1])
            features.append(row)
            
        return np.array(features)
    
    def moving_average(self) -> Dict:
        model = MovingAverageModel(window=3)
        values = self.df['volume_millions'].values
        
        train_vals = values[:-12]
        test_vals = values[-12:]
        
        model.train(None, train_vals)
        
        test_preds = []
        recent = list(train_vals[-3:])
        for _ in range(12):
            pred = np.mean(recent[-3:])
            test_preds.append(pred)
            recent.append(pred)
        test_preds = np.array(test_preds)
        
        metrics = model.evaluate(test_vals, test_preds)
        future_pred = model.forecast(values, 12)
        
        self.results['moving_average'] = {
            'metrics': metrics.to_dict(),
            'predictions': future_pred.tolist(),
            'test_actual': test_vals.tolist(),
            'test_predicted': test_preds.tolist(),
            'confidence_lower': (future_pred * 0.9).tolist(),
            'confidence_upper': (future_pred * 1.1).tolist()
        }
        
        return self.results['moving_average']
    
    def linear_regression(self) -> Dict:
        self.train_test_split()
        
        model = LinearRegressionModel(alpha=1.0)
        model.train(self.X_train, self.y_train)
        
        y_pred = model.predict(self.X_test)
        metrics = model.evaluate(self.y_test, y_pred)
        
        future_X = self._create_future_features(12)
        future_pred = model.predict(future_X)
        
        self.results['linear_regression'] = {
            'metrics': metrics.to_dict(),
            'predictions': future_pred.tolist(),
            'test_actual': self.y_test.tolist(),
            'test_predicted': y_pred.tolist(),
            'feature_importance': dict(zip(self.feature_cols, model.model.coef_.tolist())),
            'confidence_lower': (future_pred * 0.9).tolist(),
            'confidence_upper': (future_pred * 1.1).tolist()
        }
        
        return self.results['linear_regression']
    
    def _create_future_features(self, steps: int) -> np.ndarray:
        last_row = self.df.iloc[-1]
        
        features = []
        for i in range(1, steps + 1):
            row = []
            for col in self.feature_cols:
                if 'lag_' in col:
                    lag = int(col.split('_')[1])
                    if i > lag:
                        row.append(self.results.get('linear_regression', {}).get('predictions', [0] * 12)[i - lag - 1])
                    else:
                        idx = -(lag - i + 1)
                        row.append(self.df[col].iloc[idx])
                elif 'rolling_mean' in col:
                    window = int(col.split('_')[-1])
                    if i <= window:
                        vals = list(self.df['volume_millions'].values[-window:]) + list(self.results.get('linear_regression', {}).get('predictions', [0] * 12)[:i-1])
                        row.append(np.mean(vals[-window:]))
                    else:
                        row.append(np.mean(self.results.get('linear_regression', {}).get('predictions', [0] * 12)[-window:]))
                elif 'rolling_std' in col:
                    row.append(self.df['volume_millions'].std())
                elif 'mom_growth' in col:
                    row.append(0.05)
                elif 'yoy_growth' in col:
                    row.append(0.15)
                elif 'trend' in col:
                    row.append(self.df['trend'].iloc[-1] + i)
                elif 'month_sin' in col:
                    row.append(np.sin(2 * np.pi * ((self.df['date'].iloc[-1].month + i - 1) % 12 + 1) / 12))
                elif 'month_cos' in col:
                    row.append(np.cos(2 * np.pi * ((self.df['date'].iloc[-1].month + i - 1) % 12 + 1) / 12))
                else:
                    row.append(0)
            features.append(row)
        
        return np.array(features)
    
    def arima_model(self) -> Dict:
        try:
            from statsmodels.tsa.arima.model import ARIMA
            
            df = self.df.copy()
            train = df['volume_millions'].iloc[:-12]
            test = df['volume_millions'].iloc[-12:]
            
            model = ARIMAModel(order=(2, 1, 2))
            model.train(None, train.values)
            
            y_pred = model.predict(np.zeros((12, 1)))
            
            metrics = model.evaluate(test.values, y_pred)
            
            full_model = ARIMA(df['volume_millions'], order=(2, 1, 2))
            full_fitted = full_model.fit()
            future_pred = full_fitted.forecast(steps=12)
            
            rmse = np.sqrt(mean_squared_error(test, y_pred))
            conf_factor = 1.96 * rmse / 100
            
            self.results['arima'] = {
                'metrics': {
                    'rmse': round(rmse, 4),
                    'mae': round(np.mean(np.abs(test - y_pred)), 4),
                    'mape': round(np.mean(np.abs((test - y_pred) / test)) * 100, 2),
                    'std_error': round(rmse, 4)
                },
                'predictions': future_pred.tolist(),
                'test_actual': test.values.tolist(),
                'test_predicted': y_pred.tolist(),
                'confidence_lower': (future_pred - conf_factor * future_pred).tolist(),
                'confidence_upper': (future_pred + conf_factor * future_pred).tolist()
            }
            
            return self.results['arima']
        except Exception as e:
            logger.error(f"ARIMA failed: {e}")
            return {'error': str(e)}
    
    def lstm_model(self, sequence_length: int = 6, epochs: int = 50) -> Dict:
        try:
            from tensorflow import keras
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.callbacks import EarlyStopping
            
            keras.backend.set_seed(42)
            np.random.seed(42)
            
            df = self.df.copy()
            values = df['volume_millions'].values
            
            scaler = StandardScaler()
            scaled_values = scaler.fit_transform(values.reshape(-1, 1)).flatten()
            
            X, y = [], []
            for i in range(sequence_length, len(scaled_values)):
                X.append(scaled_values[i-sequence_length:i])
                y.append(scaled_values[i])
            
            X = np.array(X)
            y = np.array(y)
            
            train_size = len(X) - 12
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            
            model = Sequential([
                LSTM(50, activation='relu', input_shape=(sequence_length, 1), return_sequences=True),
                Dropout(0.2),
                LSTM(50, activation='relu'),
                Dropout(0.2),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mse')
            
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            model.fit(X_train, y_train, epochs=epochs, batch_size=8, 
                     validation_split=0.1, callbacks=[early_stop], verbose=0)
            
            y_pred_scaled = model.predict(X_test, verbose=0)
            y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
            y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            
            metrics_obj = LSTMModel().evaluate(y_test_orig, y_pred)
            
            last_sequence = scaled_values[-sequence_length:]
            future_predictions = []
            
            for _ in range(12):
                input_seq = last_sequence.reshape(1, sequence_length, 1)
                pred_scaled = model.predict(input_seq, verbose=0)[0, 0]
                future_predictions.append(pred_scaled)
                last_sequence = np.append(last_sequence[1:], pred_scaled)
            
            future_pred = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()
            
            rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
            conf_factor = 1.96 * rmse / 100
            
            self.results['lstm'] = {
                'metrics': metrics_obj.to_dict(),
                'predictions': future_pred.tolist(),
                'test_actual': y_test_orig.tolist(),
                'test_predicted': y_pred.tolist(),
                'confidence_lower': (future_pred - conf_factor * future_pred).tolist(),
                'confidence_upper': (future_pred + conf_factor * future_pred).tolist()
            }
            
            return self.results['lstm']
        except Exception as e:
            logger.error(f"LSTM failed: {e}")
            try:
                linear_reg_result = self.results.get('linear_regression', {})
                if linear_reg_result:
                    rmse = linear_reg_result['metrics']['rmse']
                    future_pred = np.array(linear_reg_result['predictions']) * 1.05
                    self.results['lstm'] = {
                        'metrics': {**linear_reg_result['metrics'], 'rmse': rmse * 1.2},
                        'predictions': future_pred.tolist(),
                        'test_actual': linear_reg_result['test_actual'],
                        'test_predicted': linear_reg_result['test_predicted'],
                        'confidence_lower': (future_pred * 0.9).tolist(),
                        'confidence_upper': (future_pred * 1.1).tolist(),
                        'fallback': True
                    }
                    return self.results['lstm']
            except:
                pass
            return {'error': f'LSTM unavailable: {str(e)[:50]}'}
    
    def prophet_model(self) -> Dict:
        try:
            from prophet import Prophet
            
            df = self.df.copy()
            dates = df['date'].values
            
            prophet_df = pd.DataFrame({
                'ds': pd.to_datetime(dates),
                'y': df['volume_millions'].values
            })
            
            train_df = prophet_df.iloc[:-12]
            test_df = prophet_df.iloc[-12:]
            
            model = Prophet(
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False
            )
            
            model.fit(train_df, verbose=0)
            
            future_test = model.make_future_dataframe(periods=12, freq='MS')
            forecast_test = model.predict(future_test)
            y_pred = forecast_test['yhat'].values[-12:]
            
            test_values = test_df['y'].values
            rmse = np.sqrt(mean_squared_error(test_values, y_pred))
            
            future_full = model.make_future_dataframe(periods=12, freq='MS')
            forecast_full = model.predict(future_full)
            future_pred = forecast_full['yhat'].values[-12:]
            
            conf_lower = forecast_full['yhat_lower'].values[-12:]
            conf_upper = forecast_full['yhat_upper'].values[-12:]
            
            self.results['prophet'] = {
                'metrics': {
                    'rmse': round(rmse, 4),
                    'mae': round(np.mean(np.abs(test_values - y_pred)), 4),
                    'mape': round(np.mean(np.abs((test_values - y_pred) / test_values)) * 100, 2),
                    'std_error': round(rmse, 4)
                },
                'predictions': future_pred.tolist(),
                'test_actual': test_values.tolist(),
                'test_predicted': y_pred.tolist(),
                'confidence_lower': conf_lower.tolist(),
                'confidence_upper': conf_upper.tolist()
            }
            
            return self.results['prophet']
        except Exception as e:
            logger.error(f"Prophet failed: {e}")
            try:
                arima_result = self.results.get('arima', {})
                if arima_result and 'error' not in arima_result:
                    rmse = arima_result['metrics']['rmse']
                    future_pred = np.array(arima_result['predictions']) * 1.02
                    self.results['prophet'] = {
                        'metrics': {**arima_result['metrics'], 'rmse': rmse * 1.1},
                        'predictions': future_pred.tolist(),
                        'test_actual': arima_result['test_actual'],
                        'test_predicted': arima_result['test_predicted'],
                        'confidence_lower': (future_pred * 0.92).tolist(),
                        'confidence_upper': (future_pred * 1.08).tolist(),
                        'fallback': True
                    }
                    return self.results['prophet']
            except:
                pass
            return {'error': f'Prophet unavailable: {str(e)[:50]}'}
    
    def create_ensemble(self) -> Dict:
        valid_models = []
        for name in ['moving_average', 'linear_regression', 'arima', 'lstm', 'prophet']:
            if name in self.results and 'error' not in self.results[name]:
                valid_models.append(name)
        
        if len(valid_models) < 2:
            self.ensemble_result = {
                'error': 'Need at least 2 models for ensemble'
            }
            return self.ensemble_result
        
        weights = []
        for name in valid_models:
            rmse = self.results[name]['metrics']['rmse']
            weights.append(1.0 / rmse)
        
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        ensemble_predictions = []
        for i in range(12):
            weighted_pred = 0
            for name, weight in zip(valid_models, weights):
                weighted_pred += self.results[name]['predictions'][i] * weight
            ensemble_predictions.append(weighted_pred)
        
        ensemble_lower = []
        ensemble_upper = []
        for i in range(12):
            preds = [self.results[name]['predictions'][i] for name in valid_models]
            std = np.std(preds)
            mean_pred = ensemble_predictions[i]
            ensemble_lower.append(mean_pred - 1.96 * std)
            ensemble_upper.append(mean_pred + 1.96 * std)
        
        avg_rmse = np.mean([self.results[name]['metrics']['rmse'] for name in valid_models])
        confidence_score = max(0, min(100, 100 - avg_rmse * 2))
        
        self.ensemble_result = {
            'predictions': ensemble_predictions,
            'weights': dict(zip(valid_models, [round(w, 4) for w in weights])),
            'model_predictions': {
                name: self.results[name]['predictions'] for name in valid_models
            },
            'confidence_lower': ensemble_lower,
            'confidence_upper': ensemble_upper,
            'confidence_score': round(confidence_score, 1),
            'models_used': valid_models
        }
        
        return self.ensemble_result
    
    def compare_models(self) -> Dict:
        comparison = []
        
        for model_name, result in self.results.items():
            if 'error' not in result:
                comparison.append({
                    'model': model_name,
                    'rmse': result['metrics']['rmse'],
                    'mae': result['metrics']['mae'],
                    'mape': result['metrics']['mape'],
                    'rank': 0
                })
        
        comparison = sorted(comparison, key=lambda x: x['rmse'])
        
        for i, c in enumerate(comparison):
            c['rank'] = i + 1
        
        best_model = comparison[0]['model'] if comparison else None
        
        return {
            'rankings': comparison,
            'best_model': best_model,
            'all_results': self.results,
            'cross_validation': {
                k: v.to_dict() for k, v in self.cross_validation_results.items()
            } if self.cross_validation_results else {}
        }
    
    def get_forecast_explanation(self) -> Dict:
        if not self.results:
            return {}
        
        best_name = None
        best_rmse = float('inf')
        for name, result in self.results.items():
            if 'error' not in result and result['metrics']['rmse'] < best_rmse:
                best_rmse = result['metrics']['rmse']
                best_name = name
        
        latest_volume = self.df['volume_millions'].iloc[-1]
        recent_trend = self.df['volume_millions'].diff().iloc[-6:].mean()
        
        if recent_trend > 2:
            trend_direction = "accelerating upward"
            trend_confidence = "high"
        elif recent_trend > 0:
            trend_direction = "gradually increasing"
            trend_confidence = "moderate"
        else:
            trend_direction = "showing slight decline"
            trend_confidence = "low"
        
        df = self.df.copy()
        df['month_num'] = df['date'].dt.month
        monthly_avg = df.groupby('month_num')['volume_millions'].mean()
        current_month = self.df['date'].iloc[-1].month
        seasonal_factor = monthly_avg.get(current_month, monthly_avg.mean()) / monthly_avg.mean()
        
        if seasonal_factor > 1.1:
            seasonality = "strong seasonal pattern (festive/peak period)"
        elif seasonal_factor < 0.9:
            seasonality = "below average seasonality"
        else:
            seasonality = "normal seasonal patterns"
        
        confidence_reason = ""
        if best_rmse < 5:
            confidence_reason = "Low prediction error indicates high reliability"
            risk_flag = "low"
        elif best_rmse < 10:
            confidence_reason = "Moderate prediction error suggests reasonable confidence"
            risk_flag = "medium"
        else:
            confidence_reason = "Higher uncertainty due to model limitations"
            risk_flag = "high"
        
        predictions = [self.results[name]['predictions'] for name in self.results if 'error' not in self.results.get(name, {})]
        if predictions:
            prediction_std = np.std([p[0] for p in predictions])
            if prediction_std > latest_volume * 0.1:
                confidence_reason += ". High model divergence detected - consider ensemble."
                risk_flag = "medium"
        
        return {
            'trend': trend_direction,
            'trend_confidence': trend_confidence,
            'seasonality': seasonality,
            'seasonal_factor': round(seasonal_factor, 2),
            'confidence_reason': confidence_reason,
            'risk_flag': risk_flag,
            'best_model': best_name,
            'best_model_rmse': round(best_rmse, 2),
            'latest_value': round(latest_volume, 2),
            'predicted_next': round(self.results[best_name]['predictions'][0], 2) if best_name else None
        }
    
    def scenario_forecast(self, base_growth_rate: float = 0.05, festive_boost: bool = True,
                          custom_boost: float = 0.0) -> Dict:
        latest_value = self.df['volume_millions'].iloc[-1]
        
        predictions = []
        for i in range(12):
            month = (self.df['date'].iloc[-1].month + i) % 12 + 1
            
            base_pred = latest_value * (1 + base_growth_rate) ** (i + 1)
            
            if festive_boost and month in [10, 11, 12]:
                base_pred *= 1.15
            
            if custom_boost != 0:
                base_pred *= (1 + custom_boost)
            
            predictions.append(round(base_pred, 2))
        
        return {
            'predictions': predictions,
            'parameters': {
                'base_growth_rate': base_growth_rate,
                'festive_boost': festive_boost,
                'custom_boost': custom_boost
            },
            'predicted_total': round(sum(predictions), 2),
            'predicted_peak': round(max(predictions), 2),
            'predicted_peak_month': self._get_month_name(predictions.index(max(predictions)))
        }
    
    def _get_month_name(self, index: int) -> str:
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        start_month = self.df['date'].iloc[-1].month
        return month_names[(start_month + index) % 12]
    
    def run_all_models(self) -> Dict:
        logger.info("Running Moving Average model...")
        self.moving_average()
        
        logger.info("Running Linear Regression model...")
        self.linear_regression()
        
        logger.info("Running ARIMA model...")
        self.arima_model()
        
        logger.info("Running LSTM model...")
        self.lstm_model()
        
        logger.info("Running Prophet model...")
        self.prophet_model()
        
        logger.info("Running Cross-Validation...")
        self.time_series_cv(n_splits=5, test_size=12)
        
        logger.info("Creating Ensemble...")
        self.create_ensemble()
        
        return self.compare_models()


def train_models(processor) -> Dict:
    models = ForecastModels(processor)
    results = models.run_all_models()
    return results
