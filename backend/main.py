from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import logging
import os
import json
import csv
from io import StringIO
from datetime import datetime

from services.scraper import scrape_upi_data
from services.processor import DataProcessor, process_data
from services.models import ForecastModels, train_models
from services.insights import generate_insights

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="UPI Intelligence Platform",
    description="Production-Grade UPI Transaction Forecasting with Multi-Model Ensemble",
    version="2.0.0",
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

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

CACHE_DIR = os.path.join(DATA_DIR, 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

DATA_CACHE_FILE = os.path.join(CACHE_DIR, 'data_cache.json')
MODELS_CACHE_FILE = os.path.join(CACHE_DIR, 'models_cache.json')

CACHE_TTL = 3600

class ProcessorHolder:
    def __init__(self):
        self.df = None
        self.stats = None
        self.processor = None
        self.model_results = None
        self.insights = None
        self.last_updated = None
        self.forecast_explanation = None
        self.scenario_results = None
        
    def is_cache_valid(self) -> bool:
        if self.df is None or self.last_updated is None:
            return False
        age = (datetime.now() - self.last_updated).total_seconds()
        return age < CACHE_TTL

holder = ProcessorHolder()

class ScenarioRequest(BaseModel):
    growth_rate: float = 0.05
    festive_boost: bool = True
    custom_boost: float = 0.0

@app.get("/")
async def root():
    return {
        "message": "UPI Intelligence Platform",
        "version": "2.0.0",
        "tagline": "India's Premier UPI Analytics & Forecasting System",
        "endpoints": {
            "data": ["/fetch-data", "/data", "/stats", "/time-series"],
            "forecasting": ["/forecast", "/ensemble", "/scenario"],
            "evaluation": ["/models", "/cross-validation", "/explanation"],
            "analytics": ["/anomalies", "/insights"],
            "export": ["/export/forecast"]
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "data_loaded": holder.df is not None,
        "models_trained": holder.model_results is not None
    }

@app.get("/fetch-data")
async def fetch_data(force: bool = Query(False, description="Force refresh even if cached")):
    try:
        if not force and holder.is_cache_valid():
            return {
                "status": "cached",
                "message": "Using cached data",
                "records": len(holder.df),
                "age_seconds": (datetime.now() - holder.last_updated).total_seconds()
            }
        
        logger.info("Fetching UPI data...")
        df = scrape_upi_data()
        
        data_path = os.path.join(DATA_DIR, 'upi_data.csv')
        df.to_csv(data_path, index=False)
        logger.info(f"Data saved to {data_path}")
        
        clean_df, stats, processor = process_data(df)
        
        holder.df = clean_df
        holder.stats = stats
        holder.processor = processor
        holder.last_updated = datetime.now()
        holder.model_results = None
        holder.insights = None
        
        return {
            "status": "success",
            "message": "Data fetched and processed successfully",
            "records": len(clean_df),
            "date_range": stats['date_range'],
            "timestamp": holder.last_updated.isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data")
async def get_data():
    if holder.df is None:
        raise HTTPException(status_code=404, detail="Data not loaded. Call /fetch-data first.")
    
    return {
        "dates": holder.df['date'].dt.strftime('%Y-%m').tolist(),
        "volume": holder.df['volume_millions'].tolist(),
        "value": holder.df['value_crores'].tolist(),
        "updated_at": holder.last_updated.isoformat() if holder.last_updated else None
    }

@app.get("/stats")
async def get_stats():
    if holder.stats is None:
        raise HTTPException(status_code=404, detail="Data not loaded. Call /fetch-data first.")
    
    return holder.stats

@app.get("/time-series")
async def get_time_series():
    if holder.processor is None:
        raise HTTPException(status_code=404, detail="Data not loaded. Call /fetch-data first.")
    
    return holder.processor.get_time_series_data()

@app.get("/forecast")
async def get_forecast():
    if holder.processor is None:
        raise HTTPException(status_code=404, detail="Data not loaded. Call /fetch-data first.")
    
    if holder.model_results is None:
        logger.info("Training forecasting models...")
        models = ForecastModels(holder.processor)
        holder.model_results = models.run_all_models()
        holder.forecast_explanation = models.get_forecast_explanation()
    
    last_date = holder.df['date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=12, freq='MS')
    
    result = {
        "forecast_dates": [d.strftime('%Y-%m') for d in future_dates],
        "models": {}
    }
    
    for model_name, model_data in holder.model_results.get('all_results', {}).items():
        if 'error' not in model_data:
            result["models"][model_name] = {
                "predictions": model_data.get('predictions', []),
                "metrics": model_data.get('metrics', {}),
                "test_actual": model_data.get('test_actual', []),
                "test_predicted": model_data.get('test_predicted', []),
                "confidence_lower": model_data.get('confidence_lower', []),
                "confidence_upper": model_data.get('confidence_upper', [])
            }
    
    result["comparison"] = holder.model_results.get('rankings', [])
    result["best_model"] = holder.model_results.get('best_model')
    result["cross_validation"] = holder.model_results.get('cross_validation', {})
    
    return result

@app.get("/ensemble")
async def get_ensemble():
    if holder.processor is None:
        raise HTTPException(status_code=404, detail="Data not loaded. Call /fetch-data first.")
    
    if holder.model_results is None:
        models = ForecastModels(holder.processor)
        holder.model_results = models.run_all_models()
        holder.ensemble_result = models.ensemble_result
    else:
        if not hasattr(holder, 'ensemble_result') or holder.ensemble_result is None:
            models = ForecastModels(holder.processor)
            models.results = holder.model_results.get('all_results', {})
            holder.ensemble_result = models.create_ensemble()
    
    if 'error' in holder.ensemble_result:
        raise HTTPException(status_code=400, detail=holder.ensemble_result['error'])
    
    last_date = holder.df['date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=12, freq='MS')
    
    return {
        "forecast_dates": [d.strftime('%Y-%m') for d in future_dates],
        "ensemble_predictions": holder.ensemble_result.get('predictions', []),
        "model_predictions": holder.ensemble_result.get('model_predictions', {}),
        "weights": holder.ensemble_result.get('weights', {}),
        "confidence_lower": holder.ensemble_result.get('confidence_lower', []),
        "confidence_upper": holder.ensemble_result.get('confidence_upper', []),
        "confidence_score": holder.ensemble_result.get('confidence_score', 0),
        "models_used": holder.ensemble_result.get('models_used', [])
    }

@app.post("/scenario")
async def run_scenario(request: ScenarioRequest):
    if holder.processor is None:
        raise HTTPException(status_code=404, detail="Data not loaded. Call /fetch-data first.")
    
    models = ForecastModels(holder.processor)
    models.results = holder.model_results.get('all_results', {}) if holder.model_results else {}
    
    result = models.scenario_forecast(
        base_growth_rate=request.growth_rate,
        festive_boost=request.festive_boost,
        custom_boost=request.custom_boost
    )
    
    last_date = holder.df['date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=12, freq='MS')
    
    return {
        "forecast_dates": [d.strftime('%Y-%m') for d in future_dates],
        **result
    }

@app.get("/models")
async def get_models():
    if holder.model_results is None:
        raise HTTPException(status_code=404, detail="Models not trained. Call /forecast first.")
    
    rankings = holder.model_results.get('rankings', [])
    
    return {
        "rankings": rankings,
        "best_model": holder.model_results.get('best_model'),
        "total_models": len(rankings),
        "cross_validation_results": holder.model_results.get('cross_validation', {})
    }

@app.get("/cross-validation")
async def get_cross_validation():
    if holder.model_results is None:
        raise HTTPException(status_code=404, detail="Models not trained. Call /forecast first.")
    
    return holder.model_results.get('cross_validation', {})

@app.get("/explanation")
async def get_forecast_explanation():
    if holder.forecast_explanation is None:
        if holder.model_results is None:
            raise HTTPException(status_code=404, detail="Models not trained. Call /forecast first.")
        models = ForecastModels(holder.processor)
        models.results = holder.model_results.get('all_results', {})
        holder.forecast_explanation = models.get_forecast_explanation()
    
    return holder.forecast_explanation

@app.get("/anomalies")
async def get_anomalies(threshold: float = Query(2.0, ge=1.0, le=5.0)):
    if holder.processor is None:
        raise HTTPException(status_code=404, detail="Data not loaded. Call /fetch-data first.")
    
    anomalies = holder.processor.get_anomalies(threshold=threshold)
    
    return {
        "anomalies": anomalies,
        "count": len(anomalies),
        "threshold": threshold
    }

@app.get("/insights")
async def get_insights():
    if holder.df is None:
        raise HTTPException(status_code=404, detail="Data not loaded. Call /fetch-data first.")
    
    if holder.model_results is None:
        models = ForecastModels(holder.processor)
        holder.model_results = models.run_all_models()
    
    if holder.insights is None:
        holder.insights = generate_insights(holder.df, holder.stats, holder.model_results)
    
    if holder.forecast_explanation is None:
        models = ForecastModels(holder.processor)
        models.results = holder.model_results.get('all_results', {})
        holder.forecast_explanation = models.get_forecast_explanation()
    
    holder.insights['forecast_explanation'] = holder.forecast_explanation
    
    return holder.insights

@app.get("/export/forecast")
async def export_forecast(format: str = Query("json", enum=["json", "csv"])):
    if holder.model_results is None:
        raise HTTPException(status_code=404, detail="Models not trained. Call /forecast first.")
    
    last_date = holder.df['date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=12, freq='MS')
    future_dates_str = [d.strftime('%Y-%m') for d in future_dates]
    
    export_data = {
        "forecast_dates": future_dates_str,
        "predictions": {},
        "generated_at": datetime.now().isoformat()
    }
    
    for model_name, model_data in holder.model_results.get('all_results', {}).items():
        if 'error' not in model_data:
            export_data["predictions"][model_name] = {
                "forecast": model_data.get('predictions', []),
                "confidence_lower": model_data.get('confidence_lower', []),
                "confidence_upper": model_data.get('confidence_upper', []),
                "metrics": model_data.get('metrics', {})
            }
    
    if format == "csv":
        output = StringIO()
        writer = csv.writer(output)
        
        writer.writerow(["date", "model", "prediction", "confidence_lower", "confidence_upper"])
        
        for model_name, model_data in export_data["predictions"].items():
            for i, date in enumerate(future_dates_str):
                writer.writerow([
                    date,
                    model_name,
                    model_data["forecast"][i] if i < len(model_data["forecast"]) else "",
                    model_data["confidence_lower"][i] if i < len(model_data["confidence_lower"]) else "",
                    model_data["confidence_upper"][i] if i < len(model_data["confidence_upper"]) else ""
                ])
        
        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=upi_forecast.csv"}
        )
    
    return export_data

@app.get("/dashboard")
async def get_dashboard():
    if holder.df is None or holder.stats is None:
        raise HTTPException(status_code=404, detail="Data not loaded. Call /fetch-data first.")
    
    if holder.model_results is None:
        models = ForecastModels(holder.processor)
        holder.model_results = models.run_all_models()
    
    best_model = holder.model_results.get('best_model')
    best_metrics = None
    if best_model and best_model in holder.model_results.get('all_results', {}):
        best_metrics = holder.model_results['all_results'][best_model].get('metrics', {})
    
    latest_volume = holder.stats['volume']['latest']
    latest_date = holder.stats['date_range']['end']
    
    next_month_pred = None
    if best_model and best_model in holder.model_results.get('all_results', {}):
        predictions = holder.model_results['all_results'][best_model].get('predictions', [])
        if predictions:
            next_month_pred = predictions[0]
    
    confidence_score = 0
    if best_metrics and 'rmse' in best_metrics:
        confidence_score = max(0, min(100, 100 - best_metrics['rmse'] * 3))
    
    return {
        "kpis": {
            "current_volume": latest_volume,
            "current_volume_date": latest_date,
            "yoy_growth": holder.stats['growth_rate']['volume_yoy'],
            "mom_growth": holder.stats['growth_rate']['volume_mom'],
            "predicted_next_month": round(next_month_pred, 2) if next_month_pred else None,
            "model_confidence": round(confidence_score, 1),
            "best_model": best_model,
            "best_model_rmse": best_metrics.get('rmse') if best_metrics else None
        },
        "time_range": {
            "start": holder.stats['date_range']['start'],
            "end": holder.stats['date_range']['end'],
            "total_records": holder.stats['total_records']
        },
        "volume_stats": holder.stats['volume'],
        "last_updated": holder.last_updated.isoformat() if holder.last_updated else None
    }

@app.on_event("startup")
async def startup_event():
    try:
        logger.info("UPI Intelligence Platform v2.0 - Starting...")
        logger.info("Auto-fetching data on startup...")
        
        df = scrape_upi_data()
        clean_df, stats, processor = process_data(df)
        
        holder.df = clean_df
        holder.stats = stats
        holder.processor = processor
        holder.last_updated = datetime.now()
        
        logger.info("Training models...")
        models = ForecastModels(processor)
        holder.model_results = models.run_all_models()
        holder.ensemble_result = models.ensemble_result
        holder.forecast_explanation = models.get_forecast_explanation()
        
        logger.info("Generating insights...")
        holder.insights = generate_insights(holder.df, holder.stats, holder.model_results)
        
        logger.info("=" * 50)
        logger.info("UPI Intelligence Platform Ready!")
        logger.info(f"Best Model: {holder.model_results.get('best_model')}")
        logger.info(f"Confidence Score: {round(holder.forecast_explanation.get('confidence_score', 0), 1) if isinstance(holder.forecast_explanation, dict) else 'N/A'}")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Startup initialization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
