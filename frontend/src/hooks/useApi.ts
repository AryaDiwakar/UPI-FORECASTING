import { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import type {
  TimeSeriesData,
  Stats,
  ForecastData,
  Anomaly,
  Insights,
  Dashboard,
  EnsembleData,
  ScenarioResult,
  ForecastExplanation
} from '../types';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export function useApi() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [timeSeriesData, setTimeSeriesData] = useState<TimeSeriesData | null>(null);
  const [stats] = useState<Stats | null>(null);
  const [forecast, setForecast] = useState<ForecastData | null>(null);
  const [dashboard, setDashboard] = useState<Dashboard | null>(null);
  const [ensemble, setEnsemble] = useState<EnsembleData | null>(null);
  const [explanation, setExplanation] = useState<ForecastExplanation | null>(null);
  const [anomalies, setAnomalies] = useState<Anomaly[]>([]);
  const [insights, setInsights] = useState<Insights | null>(null);
  const [scenario, setScenario] = useState<ScenarioResult | null>(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [dashboardRes, forecastRes, anomaliesRes, insightsRes] = await Promise.all([
        axios.get(`${API_BASE}/dashboard`),
        axios.get(`${API_BASE}/forecast`),
        axios.get(`${API_BASE}/anomalies`),
        axios.get(`${API_BASE}/insights`),
      ]);

      setDashboard(dashboardRes.data);
      setForecast(forecastRes.data);
      setAnomalies(anomaliesRes.data.anomalies);
      setInsights(insightsRes.data);
      setLastUpdated(new Date());

      if (dashboardRes.data.kpis) {
        const tsData = await axios.get(`${API_BASE}/data`);
        setTimeSeriesData(tsData.data);
      }
    } catch (err) {
      console.error('Error fetching data:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch data');
    } finally {
      setLoading(false);
    }
  }, []);

  const refreshData = useCallback(async (force: boolean = true) => {
    setLoading(true);
    setError(null);
    try {
      await axios.get(`${API_BASE}/fetch-data?force=${force}`);
      await fetchData();
    } catch (err) {
      console.error('Error refreshing data:', err);
      setError(err instanceof Error ? err.message : 'Failed to refresh data');
    } finally {
      setLoading(false);
    }
  }, [fetchData]);

  const fetchEnsemble = useCallback(async () => {
    try {
      const res = await axios.get(`${API_BASE}/ensemble`);
      setEnsemble(res.data);
    } catch (err) {
      console.error('Error fetching ensemble:', err);
    }
  }, []);

  const fetchExplanation = useCallback(async () => {
    try {
      const res = await axios.get(`${API_BASE}/explanation`);
      setExplanation(res.data);
    } catch (err) {
      console.error('Error fetching explanation:', err);
    }
  }, []);

  const runScenario = useCallback(async (params: {
    growth_rate: number;
    festive_boost: boolean;
    custom_boost: number;
  }) => {
    try {
      const res = await axios.post(`${API_BASE}/scenario`, params);
      setScenario(res.data);
      return res.data;
    } catch (err) {
      console.error('Error running scenario:', err);
      throw err;
    }
  }, []);

  const exportForecast = useCallback(async (format: 'json' | 'csv' = 'csv') => {
    try {
      const res = await axios.get(`${API_BASE}/export/forecast?format=${format}`, {
        responseType: format === 'csv' ? 'blob' : 'json',
      });
      return res.data;
    } catch (err) {
      console.error('Error exporting forecast:', err);
      throw err;
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return {
    loading,
    error,
    lastUpdated,
    timeSeriesData,
    stats,
    forecast,
    dashboard,
    ensemble,
    explanation,
    anomalies,
    insights,
    scenario,
    refreshData,
    fetchEnsemble,
    fetchExplanation,
    runScenario,
    exportForecast,
  };
}
