"""
Models Module
Implements all forecasting models: ARIMA, XGBoost, LSTM, Attention-LSTM, and Ensemble.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import joblib
import os

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Metrics container for model evaluation."""
    rmse: float
    mae: float
    mape: float
    r2: float
    std_error: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'rmse': round(self.rmse, 4),
            'mae': round(self.mae, 4),
            'mape': round(self.mape, 4),
            'r2': round(self.r2, 4),
            'std_error': round(self.std_error, 4)
        }


@dataclass
class ModelResult:
    """Complete model result container."""
    name: str
    model_type: str
    metrics: ModelMetrics
    predictions: np.ndarray
    test_actual: np.ndarray
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))
    confidence_lower: np.ndarray = field(default_factory=lambda: np.array([]))
    confidence_upper: np.ndarray = field(default_factory=lambda: np.array([]))
    feature_importance: Optional[Dict[str, float]] = None
    training_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'model_type': self.model_type,
            'metrics': self.metrics.to_dict(),
            'predictions': self.predictions.tolist(),
            'test_actual': self.test_actual.tolist(),
            'residuals': self.residuals.tolist() if len(self.residuals) > 0 else [],
            'confidence_lower': self.confidence_lower.tolist() if len(self.confidence_lower) > 0 else [],
            'confidence_upper': self.confidence_upper.tolist() if len(self.confidence_upper) > 0 else [],
            'feature_importance': self.feature_importance,
            'training_time': round(self.training_time, 2)
        }


class BaseModel(ABC):
    """Abstract base class for all models."""
    
    def __init__(self, name: str, model_type: str):
        self.name = name
        self.model_type = model_type
        self.model = None
        self.feature_names: List[str] = []
    
    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> 'BaseModel':
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def forecast(self, last_values: np.ndarray, steps: int) -> np.ndarray:
        """Generate multi-step forecast."""
        pass
    
    def save(self, path: str):
        """Save model to disk."""
        joblib.dump({'model': self.model, 'feature_names': self.feature_names}, path)
    
    def load(self, path: str):
        """Load model from disk."""
        data = joblib.load(path)
        self.model = data['model']
        self.feature_names = data.get('feature_names', [])


class ARIMAModel(BaseModel):
    """ARIMA model with auto parameter selection."""
    
    def __init__(self, order: Tuple[int, int, int] = (2, 1, 2)):
        super().__init__("ARIMA", "statistical")
        self.order = order
        self.fitted_model = None
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> 'ARIMAModel':
        from statsmodels.tsa.arima.model import ARIMA
        
        self.model = ARIMA(y_train, order=self.order)
        self.fitted_model = self.model.fit()
        return self
    
    def predict(self, steps: int) -> np.ndarray:
        if self.fitted_model is None:
            return np.zeros(steps)
        forecast_result = self.fitted_model.forecast(steps=steps)
        if hasattr(forecast_result, 'values'):
            return forecast_result.values
        return np.array(forecast_result)
    
    def forecast(self, last_values: np.ndarray, steps: int) -> np.ndarray:
        from statsmodels.tsa.arima.model import ARIMA
        
        model = ARIMA(last_values, order=self.order)
        fitted = model.fit()
        forecast_result = fitted.forecast(steps=steps)
        if hasattr(forecast_result, 'values'):
            return forecast_result.values
        return np.array(forecast_result)
    
    def auto_tune(self, y_train: np.ndarray, 
                  max_p: int = 5, max_d: int = 2, max_q: int = 5) -> Tuple[int, int, int]:
        """Find best ARIMA parameters using AIC."""
        from statsmodels.tsa.arima.model import ARIMA
        
        best_aic = float('inf')
        best_order = self.order
        
        for p in range(1, max_p + 1):
            for d in range(0, max_d + 1):
                for q in range(0, max_q + 1):
                    try:
                        model = ARIMA(y_train, order=(p, d, q))
                        fitted = model.fit()
                        aic = fitted.aic
                        
                        if aic < best_aic:
                            best_aic = aic
                            best_order = (p, d, q)
                    except:
                        continue
        
        self.order = best_order
        self.fit(y_train[-len(y_train):], y_train[-len(y_train):])
        logger.info(f"Best ARIMA order: {best_order}, AIC: {best_aic:.2f}")
        return best_order


class SARIMAModel(BaseModel):
    """Seasonal ARIMA model."""
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1), 
                 seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12)):
        super().__init__("SARIMA", "statistical")
        self.order = order
        self.seasonal_order = seasonal_order
        self.fitted_model = None
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> 'SARIMAModel':
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        
        self.model = SARIMAX(y_train, order=self.order, seasonal_order=self.seasonal_order)
        self.fitted_model = self.model.fit(disp=False)
        return self
    
    def predict(self, steps: int) -> np.ndarray:
        if self.fitted_model is None:
            return np.zeros(steps)
        forecast_result = self.fitted_model.forecast(steps=steps)
        if hasattr(forecast_result, 'values'):
            return forecast_result.values
        return np.array(forecast_result)
    
    def forecast(self, last_values: np.ndarray, steps: int) -> np.ndarray:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        
        model = SARIMAX(last_values, order=self.order, seasonal_order=self.seasonal_order)
        fitted = model.fit(disp=False)
        forecast_result = fitted.forecast(steps=steps)
        if hasattr(forecast_result, 'values'):
            return forecast_result.values
        return np.array(forecast_result)


class RidgeRegressionModel(BaseModel):
    """Ridge Regression with regularization."""
    
    def __init__(self, alpha: float = 1.0):
        super().__init__("Ridge", "classical")
        self.alpha = alpha
        self.scaler = None
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            feature_names: List[str] = None, **kwargs) -> 'RidgeRegressionModel':
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)
        
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X_scaled, y_train)
        
        if feature_names:
            self.feature_names = feature_names
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def forecast(self, last_values: np.ndarray, steps: int) -> np.ndarray:
        predictions = []
        history = list(last_values)
        
        for step in range(steps):
            features = self._build_features(history, step)
            if len(features) != len(self.feature_names):
                if len(features) < len(self.feature_names):
                    features.extend([0] * (len(self.feature_names) - len(features)))
                else:
                    features = features[:len(self.feature_names)]
            
            X_pred = self.scaler.transform(np.array(features).reshape(1, -1))
            pred = self.model.predict(X_pred)[0]
            predictions.append(pred)
            history.append(pred)
        
        return np.array(predictions)
    
    def _build_features(self, history: list, step: int) -> list:
        """Build feature vector from history for a single prediction."""
        values = np.array(history)
        features = []
        
        month = (len(history) + step) % 12 + 1
        
        for lag in [1, 3, 6, 12]:
            if len(values) >= lag:
                features.append(values[-lag])
            else:
                features.append(values.mean() if len(values) > 0 else 0)
        
        for window in [3, 6, 12]:
            if len(values) >= window:
                window_vals = values[-window:]
                features.extend([
                    window_vals.mean(),
                    window_vals.std() if len(window_vals) > 1 else 0,
                    window_vals.min(),
                    window_vals.max(),
                    np.median(window_vals)
                ])
            else:
                features.extend([0, 0, 0, 0, 0])
        
        for alpha in [0.3, 0.5]:
            ema = values[0] if len(values) > 0 else 0
            for v in values[1:]:
                ema = alpha * v + (1 - alpha) * ema
            features.append(ema)
        
        if len(values) >= 2:
            features.append(values[-1] - values[-2])
        else:
            features.append(0)
        
        features.extend([
            np.sin(2 * np.pi * month / 12),
            np.cos(2 * np.pi * month / 12),
            np.sin(4 * np.pi * month / 12),
            np.cos(4 * np.pi * month / 12)
        ])
        
        if len(values) >= 2:
            momentum = (values[-1] - values[-2]) / (values[-2] + 1e-10)
            features.append(momentum)
        else:
            features.append(0)
        
        return features


class XGBoostModel(BaseModel):
    """XGBoost gradient boosting model."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6,
                 learning_rate: float = 0.1, subsample: float = 0.8,
                 colsample_bytree: float = 0.8):
        super().__init__("XGBoost", "gradient_boosting")
        
        try:
            from xgboost import XGBRegressor
            self.XGBRegressor = XGBRegressor
        except ImportError:
            logger.warning("XGBoost not installed. Using RandomForest fallback.")
            self.XGBRegressor = None
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.scaler = None
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            feature_names: List[str] = None, **kwargs) -> 'XGBoostModel':
        from sklearn.preprocessing import StandardScaler
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)
        
        if self.XGBRegressor:
            self.model = self.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=42,
                n_jobs=-1
            )
        else:
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42,
                n_jobs=-1
            )
        
        self.model.fit(X_scaled, y_train)
        
        if feature_names:
            self.feature_names = feature_names
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def forecast(self, last_values: np.ndarray, steps: int) -> np.ndarray:
        predictions = []
        history = list(last_values)
        
        for step in range(steps):
            features = self._build_features(history, step)
            if len(features) != len(self.feature_names):
                if len(features) < len(self.feature_names):
                    features.extend([0] * (len(self.feature_names) - len(features)))
                else:
                    features = features[:len(self.feature_names)]
            
            X_pred = self.scaler.transform(np.array(features).reshape(1, -1))
            pred = self.model.predict(X_pred)[0]
            predictions.append(pred)
            history.append(pred)
        
        return np.array(predictions)
    
    def _build_features(self, history: list, step: int) -> list:
        """Build feature vector from history for a single prediction."""
        values = np.array(history)
        features = []
        
        month = (len(history) + step) % 12 + 1
        
        for lag in [1, 3, 6, 12]:
            if len(values) >= lag:
                features.append(values[-lag])
            else:
                features.append(values.mean() if len(values) > 0 else 0)
        
        for window in [3, 6, 12]:
            if len(values) >= window:
                window_vals = values[-window:]
                features.extend([
                    window_vals.mean(),
                    window_vals.std() if len(window_vals) > 1 else 0,
                    window_vals.min(),
                    window_vals.max(),
                    np.median(window_vals)
                ])
            else:
                features.extend([0, 0, 0, 0, 0])
        
        for alpha in [0.3, 0.5]:
            ema = values[0] if len(values) > 0 else 0
            for v in values[1:]:
                ema = alpha * v + (1 - alpha) * ema
            features.append(ema)
        
        if len(values) >= 2:
            features.append(values[-1] - values[-2])
        else:
            features.append(0)
        
        features.extend([
            np.sin(2 * np.pi * month / 12),
            np.cos(2 * np.pi * month / 12),
            np.sin(4 * np.pi * month / 12),
            np.cos(4 * np.pi * month / 12)
        ])
        
        if len(values) >= 2:
            momentum = (values[-1] - values[-2]) / (values[-2] + 1e-10)
            features.append(momentum)
        else:
            features.append(0)
        
        return features
    
    def get_feature_importance(self) -> Dict[str, float]:
        if hasattr(self.model, 'feature_importances_') and self.feature_names:
            importances = self.model.feature_importances_
            return dict(zip(self.feature_names, importances))
        return {}


class RandomForestModel(BaseModel):
    """Random Forest model."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10):
        super().__init__("RandomForest", "ensemble")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.scaler = None
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            feature_names: List[str] = None, **kwargs) -> 'RandomForestModel':
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestRegressor
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)
        
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_scaled, y_train)
        
        if feature_names:
            self.feature_names = feature_names
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def forecast(self, last_values: np.ndarray, steps: int) -> np.ndarray:
        predictions = []
        history = list(last_values)
        
        for step in range(steps):
            features = self._build_features(history, step)
            if len(features) != len(self.feature_names):
                if len(features) < len(self.feature_names):
                    features.extend([0] * (len(self.feature_names) - len(features)))
                else:
                    features = features[:len(self.feature_names)]
            
            X_pred = self.scaler.transform(np.array(features).reshape(1, -1))
            pred = self.model.predict(X_pred)[0]
            predictions.append(pred)
            history.append(pred)
        
        return np.array(predictions)
    
    def _build_features(self, history: list, step: int) -> list:
        """Build feature vector from history for a single prediction."""
        values = np.array(history)
        features = []
        
        month = (len(history) + step) % 12 + 1
        
        for lag in [1, 3, 6, 12]:
            if len(values) >= lag:
                features.append(values[-lag])
            else:
                features.append(values.mean() if len(values) > 0 else 0)
        
        for window in [3, 6, 12]:
            if len(values) >= window:
                window_vals = values[-window:]
                features.extend([
                    window_vals.mean(),
                    window_vals.std() if len(window_vals) > 1 else 0,
                    window_vals.min(),
                    window_vals.max(),
                    np.median(window_vals)
                ])
            else:
                features.extend([0, 0, 0, 0, 0])
        
        for alpha in [0.3, 0.5]:
            ema = values[0] if len(values) > 0 else 0
            for v in values[1:]:
                ema = alpha * v + (1 - alpha) * ema
            features.append(ema)
        
        if len(values) >= 2:
            features.append(values[-1] - values[-2])
        else:
            features.append(0)
        
        features.extend([
            np.sin(2 * np.pi * month / 12),
            np.cos(2 * np.pi * month / 12),
            np.sin(4 * np.pi * month / 12),
            np.cos(4 * np.pi * month / 12)
        ])
        
        if len(values) >= 2:
            momentum = (values[-1] - values[-2]) / (values[-2] + 1e-10)
            features.append(momentum)
        else:
            features.append(0)
        
        return features
    
    def get_feature_importance(self) -> Dict[str, float]:
        if hasattr(self.model, 'feature_importances_') and self.feature_names:
            importances = self.model.feature_importances_
            return dict(zip(self.feature_names, importances))
        return {}


class LSTMModel(BaseModel):
    """LSTM neural network for sequence prediction."""
    
    def __init__(self, sequence_length: int = 6, epochs: int = 50, 
                 lstm_units: int = 50, dropout: float = 0.2):
        super().__init__("LSTM", "deep_learning")
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.keras_model = None
        self.scaler = None
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None, **kwargs) -> 'LSTMModel':
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
            
            tf.random.set_seed(42)
            np.random.seed(42)
            
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            y_scaled = self.scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
            
            X_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            
            self.keras_model = Sequential([
                LSTM(self.lstm_units, activation='relu', input_shape=(self.sequence_length, 1), return_sequences=True),
                Dropout(self.dropout),
                LSTM(self.lstm_units, activation='relu'),
                Dropout(self.dropout),
                Dense(1)
            ])
            
            self.keras_model.compile(optimizer='adam', loss='mse')
            
            validation_data = None
            if X_val is not None and y_val is not None:
                y_val_scaled = self.scaler.transform(y_val.reshape(-1, 1))
                X_val_reshaped = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
                validation_data = (X_val_reshaped, y_val_scaled)
            
            callbacks = [
                EarlyStopping(monitor='val_loss' if validation_data else 'loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
            ]
            
            self.keras_model.fit(
                X_reshaped, y_scaled,
                epochs=self.epochs,
                batch_size=8,
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=0
            )
            
        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.keras_model is None or self.scaler is None:
            return np.zeros(len(X))
        
        X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
        y_pred_scaled = self.keras_model.predict(X_reshaped, verbose=0)
        y_pred = self.scaler.inverse_transform(y_pred_scaled).flatten()
        return y_pred
    
    def forecast(self, last_values: np.ndarray, steps: int) -> np.ndarray:
        if self.keras_model is None or self.scaler is None:
            return np.zeros(steps)
        
        scaled_values = self.scaler.fit_transform(last_values.reshape(-1, 1)).flatten()
        last_sequence = scaled_values[-self.sequence_length:]
        predictions = []
        
        for _ in range(steps):
            input_seq = last_sequence.reshape(1, self.sequence_length, 1)
            pred_scaled = self.keras_model.predict(input_seq, verbose=0)[0, 0]
            predictions.append(pred_scaled)
            last_sequence = np.append(last_sequence[1:], pred_scaled)
        
        pred_array = np.array(predictions).reshape(-1, 1)
        return self.scaler.inverse_transform(pred_array).flatten()


class AttentionLSTMModel(BaseModel):
    """LSTM with Attention mechanism."""
    
    def __init__(self, sequence_length: int = 6, epochs: int = 50,
                 lstm_units: int = 64, num_heads: int = 4, dropout: float = 0.2):
        super().__init__("AttentionLSTM", "deep_learning")
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.lstm_units = lstm_units
        self.num_heads = num_heads
        self.dropout = dropout
        self.keras_model = None
        self.scaler = None
    
    def _build_model(self):
        try:
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import (Input, LSTM, Dense, Dropout, 
                                                MultiHeadAttention, LayerNormalization, Add)
            
            inputs = Input(shape=(self.sequence_length, 1))
            
            x = LSTM(self.lstm_units, activation='relu', return_sequences=True)(inputs)
            x = Dropout(self.dropout)(x)
            
            attention_output = MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.lstm_units,
                dropout=self.dropout
            )(x, x)
            x = Add()([x, attention_output])
            x = LayerNormalization(epsilon=1e-6)(x)
            
            x = LSTM(self.lstm_units // 2, activation='relu', return_sequences=False)(x)
            x = Dropout(self.dropout)(x)
            
            x = Dense(32, activation='relu')(x)
            outputs = Dense(1)(x)
            
            return Model(inputs=inputs, outputs=outputs)
        except Exception as e:
            logger.error(f"Model building failed: {e}")
            return None
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None, **kwargs) -> 'AttentionLSTMModel':
        try:
            import tensorflow as tf
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
            from sklearn.preprocessing import StandardScaler
            
            tf.random.set_seed(42)
            np.random.seed(42)
            
            self.scaler = StandardScaler()
            y_scaled = self.scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
            
            X_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            
            self.keras_model = self._build_model()
            if self.keras_model:
                self.keras_model.compile(optimizer='adam', loss='mse')
                
                validation_data = None
                if X_val is not None and y_val is not None:
                    y_val_scaled = self.scaler.transform(y_val.reshape(-1, 1))
                    X_val_reshaped = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
                    validation_data = (X_val_reshaped, y_val_scaled)
                
                callbacks = [
                    EarlyStopping(monitor='val_loss' if validation_data else 'loss', patience=15, restore_best_weights=True),
                    ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-6)
                ]
                
                self.keras_model.fit(
                    X_reshaped, y_scaled,
                    epochs=self.epochs,
                    batch_size=8,
                    validation_data=validation_data,
                    callbacks=callbacks,
                    verbose=0
                )
                
        except Exception as e:
            logger.error(f"Attention LSTM training failed: {e}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.keras_model is None or self.scaler is None:
            return np.zeros(len(X))
        
        X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
        y_pred_scaled = self.keras_model.predict(X_reshaped, verbose=0)
        y_pred = self.scaler.inverse_transform(y_pred_scaled).flatten()
        return y_pred
    
    def forecast(self, last_values: np.ndarray, steps: int) -> np.ndarray:
        if self.keras_model is None or self.scaler is None:
            return np.zeros(steps)
        
        scaled_values = self.scaler.fit_transform(last_values.reshape(-1, 1)).flatten()
        last_sequence = scaled_values[-self.sequence_length:]
        predictions = []
        
        for _ in range(steps):
            input_seq = last_sequence.reshape(1, self.sequence_length, 1)
            pred_scaled = self.keras_model.predict(input_seq, verbose=0)[0, 0]
            predictions.append(pred_scaled)
            last_sequence = np.append(last_sequence[1:], pred_scaled)
        
        pred_array = np.array(predictions).reshape(-1, 1)
        return self.scaler.inverse_transform(pred_array).flatten()


class EnsembleModel(BaseModel):
    """Weighted ensemble of multiple models."""
    
    def __init__(self, models: List[BaseModel], weights: List[float] = None):
        super().__init__("Ensemble", "ensemble")
        self.models = models
        self.n_models = len(models)
        
        if weights is None:
            weights = [1.0 / self.n_models] * self.n_models
        elif len(weights) != self.n_models:
            weights = [1.0 / self.n_models] * self.n_models
        
        self.weights = np.array(weights) / np.sum(weights)
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> 'EnsembleModel':
        for model in self.models:
            model.fit(X_train, y_train, **kwargs)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = []
        last_values = None
        steps = len(X)
        
        for model, weight in zip(self.models, self.weights):
            try:
                if hasattr(model, 'fitted_model') and model.fitted_model is not None:
                    pred = model.predict(steps)
                elif hasattr(model, 'keras_model') and model.keras_model is not None:
                    if last_values is None:
                        last_values = model.get_last_values_for_forecast()
                    if last_values is not None:
                        pred = model.forecast(last_values, steps)
                    else:
                        continue
                else:
                    pred = model.predict(X)
            except (TypeError, AttributeError, ValueError) as e:
                logger.warning(f"Model {getattr(model, 'name', 'unknown')} predict failed: {e}")
                continue
            predictions.append(pred * weight)
        
        if not predictions:
            return np.zeros(steps)
        return np.sum(predictions, axis=0)
    
    def forecast(self, last_values: np.ndarray, steps: int) -> np.ndarray:
        all_predictions = []
        for model in self.models:
            if hasattr(model, 'forecast'):
                pred = model.forecast(last_values, steps)
            else:
                pred = np.full(steps, np.mean(last_values))
            all_predictions.append(pred)
        
        weighted_pred = np.zeros(steps)
        for pred, weight in zip(all_predictions, self.weights):
            weighted_pred += pred * weight
        
        return weighted_pred
    
    def get_weights(self) -> Dict[str, float]:
        return {model.name: weight for model, weight in zip(self.models, self.weights)}


def create_ensemble_from_results(results: List[ModelResult]) -> Tuple[EnsembleModel, np.ndarray]:
    """Create ensemble from list of model results using inverse RMSE weights."""
    if not results:
        raise ValueError("No results to create ensemble from")
    
    errors = []
    models = []
    
    for result in results:
        if result.metrics.rmse > 0:
            errors.append(result.metrics.rmse)
            models.append(result)
    
    weights = [1.0 / (e + 1e-6) for e in errors]
    total = sum(weights)
    weights = [w / total for w in weights]
    
    ensemble = EnsembleModel(models, weights)
    return ensemble, np.array(weights)


if __name__ == "__main__":
    from data.scraper import fetch_and_store_data
    from preprocessing.cleaner import preprocess_data
    from features.engineering import create_features
    
    df, _ = fetch_and_store_data()
    df_clean, _ = preprocess_data(df)
    df_featured, feature_names = create_features(df_clean)
    
    print(f"Features: {len(feature_names)}")
    print(f"Sample features: {feature_names[:5]}")
