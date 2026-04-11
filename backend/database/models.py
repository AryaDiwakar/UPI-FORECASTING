"""
Database configuration and SQLAlchemy models for UPI Forecasting System.
"""
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/upi_forecast.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class UPITransaction(Base):
    """Raw UPI transaction data from NPCI."""
    __tablename__ = "upi_transactions"

    id = Column(Integer, primary_key=True, index=True)
    month = Column(String(20), nullable=False)
    date = Column(DateTime, nullable=False, index=True)
    volume_millions = Column(Float, nullable=False)
    value_crores = Column(Float, nullable=False)
    source = Column(String(50), default="npsi")  # npci, generated, api
    scraped_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            "id": self.id,
            "month": self.month,
            "date": self.date.strftime("%Y-%m") if self.date else None,
            "volume_millions": self.volume_millions,
            "value_crores": self.value_crores,
            "source": self.source,
            "scraped_at": self.scraped_at.isoformat() if self.scraped_at else None
        }


class ProcessedData(Base):
    """Processed data with features for ML models."""
    __tablename__ = "processed_data"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(Integer, index=True)
    date = Column(DateTime, nullable=False, index=True)
    volume_millions = Column(Float, nullable=False)
    value_crores = Column(Float, nullable=False)
    features = Column(JSON)  # All engineered features as JSON
    is_anomaly = Column(Boolean, default=False)
    anomaly_score = Column(Float, nullable=True)
    processed_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "transaction_id": self.transaction_id,
            "date": self.date.strftime("%Y-%m") if self.date else None,
            "volume_millions": self.volume_millions,
            "value_crores": self.value_crores,
            "features": self.features,
            "is_anomaly": self.is_anomaly,
            "anomaly_score": self.anomaly_score,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None
        }


class TrainedModel(Base):
    """Metadata for trained ML models."""
    __tablename__ = "trained_models"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(50), nullable=False, index=True)
    model_type = Column(String(30), nullable=False)  # classical, deep_learning, ensemble
    version = Column(String(20), default="1.0")
    trained_at = Column(DateTime, default=datetime.utcnow)
    
    # Training details
    train_start_date = Column(DateTime)
    train_end_date = Column(DateTime)
    test_start_date = Column(DateTime)
    test_end_date = Column(DateTime)
    
    # Metrics
    metrics_json = Column(JSON)
    
    # Feature importance
    feature_importance = Column(JSON, nullable=True)
    
    # Model file path
    model_path = Column(String(255))
    
    # Parameters used
    parameters = Column(JSON)
    
    # Active flag
    is_active = Column(Boolean, default=True)

    def to_dict(self):
        return {
            "id": self.id,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "version": self.version,
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
            "metrics": self.metrics_json,
            "feature_importance": self.feature_importance,
            "is_active": self.is_active
        }


class Prediction(Base):
    """Model predictions and forecasts."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, index=True)
    model_name = Column(String(50), nullable=False, index=True)
    
    forecast_date = Column(DateTime, nullable=False, index=True)
    predicted_value = Column(Float, nullable=False)
    confidence_lower = Column(Float)
    confidence_upper = Column(Float)
    
    # Prediction context
    horizon_months = Column(Integer, default=1)
    training_data_end = Column(DateTime)
    
    # Actual value (for backtesting)
    actual_value = Column(Float, nullable=True)
    error = Column(Float, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "model_name": self.model_name,
            "forecast_date": self.forecast_date.strftime("%Y-%m") if self.forecast_date else None,
            "predicted_value": self.predicted_value,
            "confidence_lower": self.confidence_lower,
            "confidence_upper": self.confidence_upper,
            "horizon_months": self.horizon_months,
            "actual_value": self.actual_value,
            "error": self.error
        }


class Anomaly(Base):
    """Detected anomalies in transaction data."""
    __tablename__ = "anomalies"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(Integer, index=True)
    date = Column(DateTime, nullable=False, index=True)
    
    volume = Column(Float, nullable=False)
    z_score = Column(Float)
    iqr_score = Column(Float)
    severity = Column(String(20))  # low, medium, high, critical
    
    # Detection method
    method = Column(String(50))  # zscore, iqr, isolation_forest, prophet
    
    # Context
    expected_value = Column(Float)
    deviation_percent = Column(Float)
    
    # Analysis
    possible_cause = Column(Text, nullable=True)
    is_excluded_from_training = Column(Boolean, default=False)
    
    detected_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "transaction_id": self.transaction_id,
            "date": self.date.strftime("%Y-%m") if self.date else None,
            "volume": self.volume,
            "z_score": self.z_score,
            "severity": self.severity,
            "method": self.method,
            "deviation_percent": self.deviation_percent,
            "possible_cause": self.possible_cause,
            "is_excluded_from_training": self.is_excluded_from_training
        }


class SystemLog(Base):
    """System logs for monitoring."""
    __tablename__ = "system_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    level = Column(String(10))  # INFO, WARNING, ERROR
    component = Column(String(50))  # scraper, processor, model, api
    message = Column(Text)
    details = Column(JSON, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "level": self.level,
            "component": self.component,
            "message": self.message,
            "details": self.details
        }


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
