export interface TimeSeriesData {
  dates: string[];
  volume: number[];
  value: number[];
  volume_growth?: number[];
  volume_diff?: number[];
}

export interface Stats {
  total_records: number;
  date_range: {
    start: string;
    end: string;
  };
  volume: {
    mean: number;
    std: number;
    min: number;
    max: number;
    latest: number;
    median?: number;
    q1?: number;
    q3?: number;
  };
  value: {
    mean: number;
    std: number;
    min: number;
    max: number;
    latest: number;
  };
  growth_rate: {
    volume_yoy: number;
    value_yoy: number;
    volume_mom: number;
    value_mom: number;
    cagr?: number;
  };
  volatility?: {
    volume_cv: number;
    volume_monthly_std: number;
  };
}

export interface ModelMetrics {
  rmse: number;
  mae: number;
  mape: number;
  std_error?: number;
}

export interface ModelResult {
  predictions: number[];
  metrics: ModelMetrics;
  test_actual: number[];
  test_predicted: number[];
  confidence_lower?: number[];
  confidence_upper?: number[];
  feature_importance?: Record<string, number>;
}

export interface ModelComparison {
  model: string;
  rmse: number;
  mae: number;
  mape: number;
  rank: number;
}

export interface CrossValidationResult {
  model_name: string;
  fold_metrics: ModelMetrics[];
  mean_rmse: number;
  mean_mae: number;
  mean_mape: number;
  std_rmse: number;
}

export interface ForecastData {
  forecast_dates: string[];
  models: {
    [key: string]: ModelResult;
  };
  comparison: ModelComparison[];
  best_model: string;
  cross_validation?: {
    [key: string]: CrossValidationResult;
  };
}

export interface EnsembleData {
  forecast_dates: string[];
  ensemble_predictions: number[];
  model_predictions: {
    [key: string]: number[];
  };
  weights: {
    [key: string]: number;
  };
  confidence_lower: number[];
  confidence_upper: number[];
  confidence_score: number;
  models_used: string[];
}

export interface ForecastExplanation {
  trend: string;
  trend_confidence: string;
  seasonality: string;
  seasonal_factor: number;
  confidence_reason: string;
  risk_flag: 'low' | 'medium' | 'high';
  best_model: string;
  best_model_rmse: number;
  latest_value: number;
  predicted_next: number;
}

export interface DashboardKPIs {
  current_volume: number;
  current_volume_date: string;
  yoy_growth: number;
  mom_growth: number;
  predicted_next_month: number;
  model_confidence: number;
  best_model: string;
  best_model_rmse: number;
}

export interface Dashboard {
  kpis: DashboardKPIs;
  time_range: {
    start: string;
    end: string;
    total_records: number;
  };
  volume_stats: Stats['volume'];
  last_updated: string;
}

export interface ScenarioRequest {
  growth_rate: number;
  festive_boost: boolean;
  custom_boost: number;
}

export interface ScenarioResult {
  forecast_dates: string[];
  predictions: number[];
  parameters: ScenarioRequest;
  predicted_total: number;
  predicted_peak: number;
  predicted_peak_month: string;
}

export interface Anomaly {
  date: string;
  month: string;
  volume: number;
  value: number;
  volume_zscore: number;
  value_zscore: number;
  is_outlier?: boolean;
  severity?: 'high' | 'medium' | 'low';
}

export interface AIInsight {
  type: 'acceleration' | 'deceleration' | 'divergence' | 'consensus' | 'seasonal' | 'volatility';
  icon: string;
  title: string;
  description: string;
  confidence: 'high' | 'medium' | 'low';
  action: string;
}

export interface GrowthStage {
  period: string;
  event: string;
  description: string;
}

export interface Narrative {
  story: string;
  growth_stages: GrowthStage[];
  key_milestone: string;
  next_prediction: string;
  confidence_level: 'high' | 'medium' | 'low';
}

export interface Insights {
  summary: string;
  trends: string[];
  seasonality: string[];
  model_comparison: {
    best_model: string;
    best_rmse: number;
    best_mae: number;
    best_mape: number;
    rankings: ModelComparison[];
    insights: string[];
  };
  recommendations: string[];
  ai_insights?: AIInsight[];
  forecast_explanation?: ForecastExplanation;
  narrative?: Narrative;
}

export type TimeRange = '1y' | '3y' | 'all';
export type Tab = 'overview' | 'forecast' | 'models' | 'insights' | 'scenarios';
