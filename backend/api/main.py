"""
UPI Transaction Forecasting API - Production Backend
"""
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import pandas as pd
import numpy as np
import logging
import os
import json
import csv
from io import StringIO
from pathlib import Path

from database import init_db, get_db, UPITransaction, TrainedModel, Prediction, Anomaly
from models.forecasting import (
    ForecastingPipeline, 
    train_models,
    ModelResult,
    EnsembleModel,
    AnomalyAwareForecaster
)
from models.features import create_features, AnomalyDetector, DataValidator
from services.scraper import scrape_upi_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="UPI Intelligence Platform",
    description="Production-Grade UPI Transaction Forecasting with Multi-Model Ensemble",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path(__file__).parent / "data"
MODELS_DIR = DATA_DIR / "models"
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


class TrainConfig(BaseModel):
    test_size: int = Field(12, ge=6, le=24)
    exclude_anomalies: bool = False
    anomaly_threshold: float = Field(2.5, ge=1.5, le=4.0)
    run_deep_learning: bool = True


class ForecastConfig(BaseModel):
    model_name: Optional[str] = "best"
    horizon: int = Field(12, ge=3, le=24)
    include_confidence: bool = True


class AnomalyConfig(BaseModel):
    z_threshold: float = Field(2.0, ge=1.0, le=5.0)
    iqr_multiplier: float = Field(1.5, ge=1.0, le=3.0)


class SystemState:
    """Global state for the application."""
    
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.featured_df: Optional[pd.DataFrame] = None
        self.pipeline: Optional[ForecastingPipeline] = None
        self.last_trained: Optional[datetime] = None
        self.data_loaded: bool = False
        self.models_trained: bool = False
        self.stats: Optional[Dict] = None
        
    def reset(self):
        self.df = None
        self.featured_df = None
        self.pipeline = None
        self.last_trained = None
        self.data_loaded = False
        self.models_trained = False
        self.stats = None


state = SystemState()


@app.on_event("startup")
async def startup_event():
    """Initialize database and optionally load data."""
    init_db()
    logger.info("Database initialized")
    
    try:
        await fetch_and_process_data(force=False)
        logger.info("Initial data load complete")
    except Exception as e:
        logger.warning(f"Initial data load skipped: {e}")


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "UPI Intelligence Platform",
        "version": "3.0.0",
        "status": "operational",
        "data_loaded": state.data_loaded,
        "models_trained": state.models_trained,
        "endpoints": {
            "data": ["/fetch-data", "/data", "/stats", "/time-series"],
            "training": ["/train-models"],
            "forecasting": ["/forecast", "/forecast/{model}"],
            "evaluation": ["/models/compare", "/models/feature-importance"],
            "analytics": ["/anomalies", "/insights"],
            "export": ["/export/forecast", "/export/data"]
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "data_loaded": state.data_loaded,
        "models_trained": state.models_trained,
        "records_count": len(state.df) if state.df is not None else 0,
        "last_trained": state.last_trained.isoformat() if state.last_trained else None
    }


@app.post("/fetch-data")
async def fetch_and_process_data(force: bool = True):
    """
    Fetch data from NPCI and process it.
    """
    try:
        logger.info("Fetching UPI data...")
        
        df = scrape_upi_data()
        
        if df is None or len(df) == 0:
            raise HTTPException(status_code=500, detail="Failed to fetch data")
        
        state.df = df
        
        featured_df, feature_metadata = create_features(df, exclude_anomalies=False)
        state.featured_df = featured_df
        
        state.stats = {
            'total_records': len(df),
            'date_range': {
                'start': df['date'].min().strftime('%Y-%m'),
                'end': df['date'].max().strftime('%Y-%m')
            },
            'volume': {
                'mean': round(df['volume_millions'].mean(), 2),
                'std': round(df['volume_millions'].std(), 2),
                'min': round(df['volume_millions'].min(), 2),
                'max': round(df['volume_millions'].max(), 2),
                'latest': round(df['volume_millions'].iloc[-1], 2)
            },
            'feature_metadata': feature_metadata
        }
        
        state.data_loaded = True
        state.models_trained = False
        
        df.to_csv(DATA_DIR / "upi_data.csv", index=False)
        featured_df.to_csv(DATA_DIR / "featured_data.csv", index=False)
        
        logger.info(f"Data loaded: {len(df)} records from {state.stats['date_range']['start']} to {state.stats['date_range']['end']}")
        
        return {
            "status": "success",
            "records": len(df),
            "date_range": state.stats['date_range'],
            "features_created": feature_metadata['feature_count'],
            "anomalies_detected": feature_metadata['anomalies_detected'],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data")
async def get_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    include_features: bool = False
):
    """Get raw or featured data with optional filtering."""
    if state.df is None:
        raise HTTPException(status_code=404, detail="Data not loaded. Call /fetch-data first.")
    
    df = state.featured_df if include_features and state.featured_df else state.df
    
    if start_date:
        df = df[df['date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['date'] <= pd.to_datetime(end_date)]
    
    return {
        "dates": df['date'].dt.strftime('%Y-%m').tolist(),
        "volume": df['volume_millions'].tolist(),
        "value": df['value_crores'].tolist(),
        "record_count": len(df),
        "has_features": include_features and state.featured_df is not None
    }


@app.get("/stats")
async def get_stats():
    """Get dataset statistics."""
    if state.stats is None:
        raise HTTPException(status_code=404, detail="Data not loaded. Call /fetch-data first.")
    
    if state.df is not None:
        state.stats['growth_rate'] = {
            'volume_yoy': round(((state.df['volume_millions'].iloc[-1] - state.df['volume_millions'].iloc[-13]) / 
                                state.df['volume_millions'].iloc[-13] * 100) if len(state.df) >= 13 else 0, 2),
            'volume_mom': round(((state.df['volume_millions'].iloc[-1] - state.df['volume_millions'].iloc[-2]) / 
                                state.df['volume_millions'].iloc[-2] * 100) if len(state.df) >= 2 else 0, 2),
        }
    
    return state.stats


@app.post("/train-models")
async def train_all_models(config: TrainConfig = TrainConfig()):
    """
    Train all forecasting models.
    """
    if state.featured_df is None:
        raise HTTPException(status_code=404, detail="Data not loaded. Call /fetch-data first.")
    
    try:
        logger.info(f"Training models with config: {config}")
        
        featured_df_clean = state.featured_df.copy()
        if config.exclude_anomalies and 'is_anomaly' in featured_df_clean.columns:
            featured_df_clean = featured_df_clean[~featured_df_clean['is_anomaly']]
        
        state.pipeline = ForecastingPipeline(
            test_size=config.test_size,
            sequence_length=6
        )
        
        X_train, X_test, y_train, y_test = state.pipeline.prepare_data(
            featured_df_clean, 
            exclude_anomalies=False
        )
        
        results = state.pipeline.run_all_models(
            X_train, X_test, y_train, y_test,
            run_attention_lstm=config.run_deep_learning
        )
        
        state.last_trained = datetime.now()
        state.models_trained = True
        
        best_model_name, best_result = state.pipeline.get_best_model()
        
        model_summaries = []
        for name, result in results.items():
            model_summaries.append({
                "name": name,
                "model_type": result.model_type,
                "metrics": result.metrics.to_dict(),
                "training_time_seconds": round(result.training_time, 2)
            })
        
        model_summaries = sorted(model_summaries, key=lambda x: x['metrics']['rmse'])
        for i, m in enumerate(model_summaries):
            m['rank'] = i + 1
        
        return {
            "status": "success",
            "models_trained": len(results),
            "best_model": best_model_name,
            "best_metrics": best_result.metrics.to_dict(),
            "models": model_summaries,
            "config": config.model_dump(),
            "trained_at": state.last_trained.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast")
async def get_forecast(config: ForecastConfig = ForecastConfig()):
    """
    Get forecast from all or specified model.
    """
    if state.pipeline is None or not state.models_trained:
        raise HTTPException(status_code=404, detail="Models not trained. Call /train-models first.")
    
    try:
        results = state.pipeline.results
        
        if config.model_name == "best":
            model_name, _ = state.pipeline.get_best_model()
        elif config.model_name and config.model_name in results:
            model_name = config.model_name
        else:
            raise HTTPException(status_code=400, detail=f"Model {config.model_name} not found")
        
        model_result = results[model_name]
        model_obj = model_result.__dict__.get('_ ForecastingPipeline__dict' if hasattr(model_result, '_ ForecastingPipeline__dict') else 'model')
        
        if hasattr(model_result, 'predictions'):
            predictions = model_result.predictions.tolist()[:config.horizon]
        else:
            predictions = list(state.pipeline.pipeline.forecast(model_name, config.horizon))[:config.horizon]
        
        last_date = state.df['date'].iloc[-1]
        forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                        periods=len(predictions), freq='MS')
        
        response = {
            "model": model_name,
            "metrics": model_result.metrics.to_dict(),
            "forecast_dates": [d.strftime('%Y-%m') for d in forecast_dates],
            "predictions": [round(p, 2) for p in predictions]
        }
        
        if config.include_confidence:
            std_error = model_result.metrics.std_error
            conf_factor = 1.96 * std_error / 100
            response["confidence_lower"] = [round(p * (1 - conf_factor), 2) for p in predictions]
            response["confidence_upper"] = [round(p * (1 + conf_factor), 2) for p in predictions]
            response["confidence_level"] = "95%"
        
        return response
        
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast/all")
async def get_all_forecasts(horizon: int = Query(12, ge=3, le=24)):
    """
    Get forecasts from all models for comparison.
    """
    if state.pipeline is None or not state.models_trained:
        raise HTTPException(status_code=404, detail="Models not trained. Call /train-models first.")
    
    last_date = state.df['date'].iloc[-1]
    forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                    periods=horizon, freq='MS')
    
    forecasts = {}
    for name, result in state.pipeline.results.items():
        predictions = result.predictions.tolist()[:horizon]
        forecasts[name] = {
            "predictions": [round(p, 2) for p in predictions],
            "metrics": result.metrics.to_dict(),
            "model_type": result.model_type
        }
    
    best_model, _ = state.pipeline.get_best_model()
    
    return {
        "forecast_dates": [d.strftime('%Y-%m') for d in forecast_dates],
        "forecasts": forecasts,
        "best_model": best_model,
        "horizon": horizon
    }


@app.get("/models/compare")
async def compare_models():
    """
    Compare all trained models with detailed metrics.
    """
    if state.pipeline is None or not state.models_trained:
        raise HTTPException(status_code=404, detail="Models not trained. Call /train-models first.")
    
    comparisons = []
    
    for name, result in state.pipeline.results.items():
        comparisons.append({
            "model": name,
            "model_type": result.model_type,
            "rmse": result.metrics.rmse,
            "mae": result.metrics.mae,
            "mape": result.metrics.mape,
            "std_error": result.metrics.std_error,
            "r2_score": result.metrics.r2_score,
            "rank": 0
        })
    
    comparisons = sorted(comparisons, key=lambda x: x['rmse'])
    for i, c in enumerate(comparisons):
        c['rank'] = i + 1
    
    best = comparisons[0] if comparisons else None
    
    return {
        "rankings": comparisons,
        "best_model": best['model'] if best else None,
        "best_rmse": best['rmse'] if best else None,
        "improvement_pct": round((comparisons[-1]['rmse'] - comparisons[0]['rmse']) / comparisons[-1]['rmse'] * 100, 2) if len(comparisons) > 1 else 0
    }


@app.get("/models/feature-importance")
async def get_feature_importance(top_n: int = Query(10, ge=5, le=30)):
    """
    Get feature importance from all models that support it.
    """
    if state.pipeline is None or not state.models_trained:
        raise HTTPException(status_code=404, detail="Models not trained. Call /train-models first.")
    
    aggregated_importance = state.pipeline.get_feature_importance_summary()
    
    top_features = sorted(aggregated_importance.items(), 
                         key=lambda x: x[1], reverse=True)[:top_n]
    
    model_importance = {}
    for name, result in state.pipeline.results.items():
        if result.feature_importance:
            model_importance[name] = result.feature_importance
    
    return {
        "top_features": [{"feature": f, "importance": round(i, 4)} for f, i in top_features],
        "aggregated_importance": {k: round(v, 4) for k, v in aggregated_importance.items()},
        "model_importance": model_importance,
        "feature_count": len(aggregated_importance)
    }


@app.get("/anomalies")
async def get_anomalies(config: AnomalyConfig = AnomalyConfig()):
    """
    Detect and return anomalies in the dataset.
    """
    if state.df is None:
        raise HTTPException(status_code=404, detail="Data not loaded. Call /fetch-data first.")
    
    detector = AnomalyDetector(z_threshold=config.z_threshold, 
                               iqr_multiplier=config.iqr_multiplier)
    
    result_df = detector.get_anomaly_context(state.df)
    anomalies = result_df[result_df['is_anomaly']]
    
    anomaly_list = []
    for _, row in anomalies.iterrows():
        anomaly_list.append({
            "date": row['date'].strftime('%Y-%m'),
            "volume": round(row['volume_millions'], 2),
            "z_score": round(row['z_score'], 2) if 'z_score' in row else None,
            "severity": row.get('anomaly_severity', 'unknown'),
            "possible_cause": row.get('possible_cause', 'Unknown'),
            "iqr_bounds": {
                "lower": round(row.get('iqr_lower', 0), 2),
                "upper": round(row.get('iqr_upper', 0), 2)
            } if 'iqr_lower' in row else None
        })
    
    return {
        "anomalies": anomaly_list,
        "count": len(anomaly_list),
        "config": config.model_dump(),
        "severity_summary": {
            "low": int((anomalies['anomaly_severity'] == 'low').sum()) if 'anomaly_severity' in anomalies.columns else 0,
            "medium": int((anomalies['anomaly_severity'] == 'medium').sum()) if 'anomaly_severity' in anomalies.columns else 0,
            "high": int((anomalies['anomaly_severity'] == 'high').sum()) if 'anomaly_severity' in anomalies.columns else 0,
            "critical": int((anomalies['anomaly_severity'] == 'critical').sum()) if 'anomaly_severity' in anomalies.columns else 0,
        }
    }


@app.get("/insights")
async def get_insights():
    """
    Generate business insights from data and models.
    """
    if state.df is None:
        raise HTTPException(status_code=404, detail="Data not loaded. Call /fetch-data first.")
    
    df = state.df
    insights = {}
    
    volume_stats = {
        "current": round(df['volume_millions'].iloc[-1], 2),
        "12m_ago": round(df['volume_millions'].iloc[-13], 2) if len(df) >= 13 else None,
        "all_time_high": round(df['volume_millions'].max(), 2),
        "all_time_high_date": df.loc[df['volume_millions'].idxmax(), 'date'].strftime('%Y-%m') if len(df) > 0 else None,
        "mean": round(df['volume_millions'].mean(), 2),
        "trend": "accelerating" if df['volume_millions'].diff().tail(6).mean() > df['volume_millions'].diff().tail(12).head(6).mean() else "decelerating"
    }
    
    yoy_growth = ((df['volume_millions'].iloc[-1] - df['volume_millions'].iloc[-13]) / 
                  df['volume_millions'].iloc[-13] * 100) if len(df) >= 13 else 0
    mom_growth = ((df['volume_millions'].iloc[-1] - df['volume_millions'].iloc[-2]) / 
                  df['volume_millions'].iloc[-2] * 100) if len(df) >= 2 else 0
    
    growth_insights = {
        "yoy_growth_pct": round(yoy_growth, 2),
        "mom_growth_pct": round(mom_growth, 2),
        "interpretation": "explosive" if yoy_growth > 50 else "strong" if yoy_growth > 25 else "moderate"
    }
    
    df['month'] = df['date'].dt.month
    monthly_avg = df.groupby('month')['volume_millions'].mean()
    overall_avg = df['volume_millions'].mean()
    
    seasonality = {}
    for month in range(1, 13):
        if month in monthly_avg.index:
            seasonality[month] = {
                "avg_volume": round(monthly_avg[month], 2),
                "seasonal_factor": round(monthly_avg[month] / overall_avg, 3) if overall_avg > 0 else 1,
                "is_peak": monthly_avg[month] > overall_avg * 1.1
            }
    
    peak_months = [m for m, s in seasonality.items() if s['is_peak']]
    festive_boost = (seasonality.get(11, {}).get('seasonal_factor', 1) - 1) * 100 if 11 in seasonality else 0
    
    seasonality_insights = {
        "peak_months": peak_months,
        "festive_boost_pct": round(festive_boost, 1),
        "interpretation": f"Oct-Dec shows {round(festive_boost, 1)}% higher activity" if festive_boost > 5 else "Seasonal patterns present"
    }
    
    recommendations = []
    
    if yoy_growth > 50:
        recommendations.append({
            "priority": "high",
            "category": "growth",
            "recommendation": "Consider aggressive infrastructure scaling given explosive growth trajectory"
        })
    
    if festive_boost > 10:
        recommendations.append({
            "priority": "high",
            "category": "capacity",
            "recommendation": f"Prepare for {round(festive_boost)}% festive season spike in Q4"
        })
    
    if volume_stats['trend'] == "accelerating":
        recommendations.append({
            "priority": "medium",
            "category": "planning",
            "recommendation": "Momentum is increasing - plan for sustained growth"
        })
    
    if state.pipeline and state.models_trained:
        best_model, _ = state.pipeline.get_best_model()
        recommendations.append({
            "priority": "info",
            "category": "modeling",
            "recommendation": f"Best performing model: {best_model}"
        })
    
    insights = {
        "volume_analysis": volume_stats,
        "growth_analysis": growth_insights,
        "seasonality_analysis": seasonality_insights,
        "seasonality_detail": seasonality,
        "recommendations": recommendations
    }
    
    if state.pipeline and state.models_trained:
        best_model, best_result = state.pipeline.get_best_model()
        insights["model_insights"] = {
            "best_model": best_model,
            "best_rmse": round(best_result.metrics.rmse, 4),
            "confidence": "high" if best_result.metrics.rmse < 5 else "moderate" if best_result.metrics.rmse < 10 else "low"
        }
    
    return insights


@app.get("/export/forecast")
async def export_forecast(format: str = Query("json", enum=["json", "csv"])):
    """
    Export forecast data.
    """
    if state.pipeline is None or not state.models_trained:
        raise HTTPException(status_code=404, detail="Models not trained. Call /train-models first.")
    
    last_date = state.df['date'].iloc[-1]
    horizon = 12
    forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                    periods=horizon, freq='MS')
    
    export_data = {
        "exported_at": datetime.now().isoformat(),
        "date_range": f"{forecast_dates[0].strftime('%Y-%m')} to {forecast_dates[-1].strftime('%Y-%m')}",
        "best_model": None,
        "forecasts": {}
    }
    
    best_model, _ = state.pipeline.get_best_model()
    export_data["best_model"] = best_model
    
    for name, result in state.pipeline.results.items():
        export_data["forecasts"][name] = {
            "predictions": [round(p, 2) for p in result.predictions.tolist()[:horizon]],
            "metrics": result.metrics.to_dict()
        }
    
    if format == "csv":
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(["date", "model", "prediction", "rmse", "mae", "mape"])
        
        for name, data in export_data["forecasts"].items():
            for i, date in enumerate(forecast_dates):
                metrics = data["metrics"]
                writer.writerow([
                    date.strftime('%Y-%m'),
                    name,
                    data["predictions"][i] if i < len(data["predictions"]) else "",
                    metrics.get("rmse", ""),
                    metrics.get("mae", ""),
                    metrics.get("mape", "")
                ])
        
        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=upi_forecasts.csv"}
        )
    
    return export_data


@app.get("/export/data")
async def export_data(format: str = Query("json", enum=["json", "csv"])):
    """
    Export raw data.
    """
    if state.df is None:
        raise HTTPException(status_code=404, detail="Data not loaded. Call /fetch-data first.")
    
    if format == "csv":
        output = StringIO()
        state.df.to_csv(output, index=False)
        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=upi_data.csv"}
        )
    
    return {
        "dates": state.df['date'].dt.strftime('%Y-%m').tolist(),
        "volume": state.df['volume_millions'].tolist(),
        "value": state.df['value_crores'].tolist(),
        "record_count": len(state.df)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
