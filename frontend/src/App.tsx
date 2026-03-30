import { useState, useCallback, useEffect } from 'react';
import {
  RefreshCw,
  Activity,
  Brain,
  BarChart3,
  AlertTriangle,
  Lightbulb,
  Zap,
  Download,
  Clock,
  Shield,
  Award
} from 'lucide-react';
import { useApi } from './hooks/useApi';
import { KPICard } from './components/KPICard';
import { TransactionChart, ForecastComparison, ModelComparisonChart, ConfidenceChart } from './components/Charts';
import { InsightsPanel } from './components/Insights';
import { ScenarioSimulator } from './components/ScenarioSimulator';
import { ModelBattle } from './components/ModelBattle';
import type { Tab, TimeRange } from './types';

function App() {
  const {
    loading,
    error,
    lastUpdated,
    timeSeriesData,
    dashboard,
    forecast,
    ensemble,
    anomalies,
    insights,
    refreshData,
    fetchEnsemble,
    runScenario,
  } = useApi();

  const [activeTab, setActiveTab] = useState<Tab>('overview');
  const [selectedModels, setSelectedModels] = useState<string[]>(['lstm', 'arima', 'prophet']);
  const [timeRange, setTimeRange] = useState<TimeRange>('all');
  const [secondsAgo, setSecondsAgo] = useState(0);

  useEffect(() => {
    if (lastUpdated) {
      const interval = setInterval(() => {
        setSecondsAgo(Math.floor((Date.now() - lastUpdated.getTime()) / 1000));
      }, 1000);
      return () => clearInterval(interval);
    }
  }, [lastUpdated]);

  const handleModelToggle = useCallback((model: string) => {
    setSelectedModels((prev) =>
      prev.includes(model)
        ? prev.filter((m) => m !== model)
        : [...prev, model]
    );
  }, []);

  const handleExport = useCallback(async () => {
    try {
      const csvContent = [
        ['Date', 'Model', 'Prediction', 'Confidence_Lower', 'Confidence_Upper'],
        ...(forecast?.forecast_dates || []).flatMap((date, i) =>
          Object.entries(forecast?.models || {}).map(([model, data]) => [
            date,
            model,
            data.predictions[i]?.toFixed(2) || '',
            data.confidence_lower?.[i]?.toFixed(2) || '',
            data.confidence_upper?.[i]?.toFixed(2) || '',
          ])
        ),
      ].map(row => row.join(',')).join('\n');

      const blob = new Blob([csvContent], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `upi_forecast_${new Date().toISOString().split('T')[0]}.csv`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Export failed:', err);
    }
  }, [forecast]);

  useEffect(() => {
    if (activeTab === 'forecast') {
      fetchEnsemble();
    }
  }, [activeTab, fetchEnsemble]);

  if (loading && !dashboard) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="loading-spinner w-16 h-16 mx-auto mb-6" />
          <h2 className="text-2xl font-bold text-white mb-2">UPI Intelligence Platform</h2>
          <p className="text-gray-400">Loading analytics engine...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center p-4">
        <div className="glass-card p-8 max-w-md w-full text-center">
          <AlertTriangle className="w-16 h-16 text-red-400 mx-auto mb-6" />
          <h2 className="text-xl font-bold text-white mb-2">Connection Error</h2>
          <p className="text-gray-400 mb-6">{error}</p>
          <button
            onClick={() => refreshData()}
            className="px-6 py-3 bg-gradient-to-r from-emerald-500 to-emerald-600 text-white rounded-xl font-medium hover:from-emerald-600 hover:to-emerald-700 transition-all"
          >
            Retry Connection
          </button>
        </div>
      </div>
    );
  }

  const tabs = [
    { id: 'overview', label: 'Overview', icon: Activity },
    { id: 'forecast', label: 'Forecast', icon: BarChart3 },
    { id: 'models', label: 'Models', icon: Brain },
    { id: 'scenarios', label: 'Scenarios', icon: Zap },
    { id: 'insights', label: 'Insights', icon: Lightbulb },
  ];

  const formatTimeAgo = (seconds: number) => {
    if (seconds < 60) return `${seconds}s ago`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    return `${Math.floor(seconds / 3600)}h ago`;
  };

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-50 bg-gray-900/80 backdrop-blur-xl border-b border-gray-800">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-4">
              <div className="relative">
                <div className="p-3 bg-gradient-to-br from-emerald-500 to-emerald-600 rounded-xl shadow-lg glow-green">
                  <Activity className="w-7 h-7 text-white" />
                </div>
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-emerald-400 rounded-full animate-pulse" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white">India's UPI Intelligence</h1>
                <p className="text-sm text-gray-400">Real-time analytics & forecasting platform</p>
              </div>
            </div>

            <div className="flex items-center gap-4">
              <div className="hidden md:flex items-center gap-2 px-4 py-2 bg-gray-800/50 rounded-xl">
                <Clock className="w-4 h-4 text-gray-400" />
                <span className="text-sm text-gray-400">
                  Updated {lastUpdated ? formatTimeAgo(secondsAgo) : 'Never'}
                </span>
              </div>

              <button
                onClick={() => refreshData()}
                disabled={loading}
                className="flex items-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 text-white rounded-xl transition-all disabled:opacity-50"
              >
                <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
                <span className="hidden sm:inline">Refresh</span>
              </button>
            </div>
          </div>

          <nav className="flex gap-2 overflow-x-auto pb-2">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as Tab)}
                  className={`nav-button whitespace-nowrap ${
                    activeTab === tab.id ? 'nav-button-active' : 'nav-button-inactive'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  {tab.label}
                </button>
              );
            })}
          </nav>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8">
        {activeTab === 'overview' && (
          <div className="space-y-8">
            <div className="text-center mb-8">
              <h2 className="hero-title mb-4">UPI Transaction Intelligence</h2>
              <p className="hero-subtitle mx-auto">
                AI-powered forecasting and analytics for India's digital payment ecosystem
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <KPICard
                title="This Month's Transactions"
                value={dashboard?.kpis?.current_volume || 0}
                suffix="M"
                decimals={1}
                icon="volume"
                subtitle={`as of ${dashboard?.kpis?.current_volume_date}`}
                isHighlight
                description="Total number of UPI transactions completed in the most recent month"
              />
              <KPICard
                title="Growth vs Last Year"
                value={dashboard?.kpis?.yoy_growth || 0}
                suffix="%"
                decimals={1}
                trend={dashboard?.kpis?.yoy_growth}
                icon="growth"
                description="How much transaction volume has increased compared to the same month last year"
              />
              <KPICard
                title="Next Month Forecast"
                value={dashboard?.kpis?.predicted_next_month || 0}
                suffix="M"
                decimals={1}
                icon="prediction"
                subtitle="AI Ensemble prediction"
                description="Our AI models predict this many transactions for next month based on trends"
              />
              <KPICard
                title="Prediction Accuracy"
                value={dashboard?.kpis?.model_confidence || 0}
                suffix="%"
                decimals={0}
                icon="confidence"
                subtitle={`Best model: ${dashboard?.kpis?.best_model?.replace('_', ' ').toUpperCase()}`}
                description="How reliable our forecasts are - based on how accurate past predictions were"
              />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2">
                {timeSeriesData && (
                  <TransactionChart
                    data={timeSeriesData}
                    timeRange={timeRange}
                    onTimeRangeChange={setTimeRange}
                    anomalies={anomalies}
                  />
                )}
              </div>

              <div className="space-y-4">
                <div className="glass-card p-6">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="p-2 bg-emerald-500/20 rounded-lg">
                      <Shield className="w-5 h-5 text-emerald-400" />
                    </div>
                    <h3 className="font-semibold text-white">Quick Stats</h3>
                  </div>
                  <div className="space-y-4">
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400">Total Records</span>
                      <span className="text-white font-semibold">{dashboard?.time_range?.total_records}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400">Date Range</span>
                      <span className="text-white font-semibold text-sm">
                        {dashboard?.time_range?.start?.split('-')[1]?.slice(0, 3)} '{dashboard?.time_range?.start?.split('-')[0]?.slice(2)} - {dashboard?.time_range?.end?.split('-')[1]?.slice(0, 3)} '{dashboard?.time_range?.end?.split('-')[0]?.slice(2)}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400">Peak Volume</span>
                      <span className="text-emerald-400 font-semibold">{dashboard?.volume_stats?.max?.toFixed(1)}M</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400">Anomalies</span>
                      <span className={`font-semibold ${anomalies.length > 0 ? 'text-amber-400' : 'text-gray-400'}`}>
                        {anomalies.length} detected
                      </span>
                    </div>
                  </div>
                </div>

                <div className="glass-card p-6">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="p-2 bg-purple-500/20 rounded-lg">
                      <Award className="w-5 h-5 text-purple-400" />
                    </div>
                    <h3 className="font-semibold text-white">Best Model</h3>
                  </div>
                  <div className="text-center py-4">
                    <p className="text-3xl font-bold text-gradient-green mb-2">
                      {dashboard?.kpis?.best_model?.replace('_', ' ').toUpperCase()}
                    </p>
                    <p className="text-gray-400 text-sm mb-4">
                      RMSE: {dashboard?.kpis?.best_model_rmse?.toFixed(2)}M
                    </p>
                    <div className="flex justify-center gap-2">
                      <span className="px-3 py-1 bg-emerald-500/20 text-emerald-400 rounded-full text-sm font-medium">
                        {dashboard?.kpis?.model_confidence?.toFixed(0)}% Confidence
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'forecast' && forecast && (
          <div className="space-y-8">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-bold text-white">Multi-Model Forecasting</h2>
                <p className="text-gray-400 mt-1">
                  Compare predictions across LSTM, ARIMA, Prophet, and ensemble models
                </p>
              </div>
              <div className="flex gap-3">
                <button
                  onClick={handleExport}
                  className="flex items-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 text-white rounded-xl transition-all"
                >
                  <Download className="w-4 h-4" />
                  Export CSV
                </button>
              </div>
            </div>

            <ForecastComparison
              historical={timeSeriesData!}
              forecast={forecast}
              ensemble={ensemble || undefined}
              selectedModels={selectedModels}
              onModelToggle={handleModelToggle}
              showConfidence
            />

            {ensemble && <ConfidenceChart forecast={forecast} ensemble={ensemble} />}

            <div className="glass-card p-6">
              <h3 className="text-lg font-semibold text-white mb-6">12-Month Forecast Summary</h3>
              <div className="grid grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
                {forecast.forecast_dates.map((date, i) => {
                  const ensemblePred = ensemble?.ensemble_predictions?.[i];
                  const bestPred = forecast.models[forecast.best_model]?.predictions?.[i];
                  return (
                    <div
                      key={date}
                      className="bg-gray-800/50 rounded-xl p-4 text-center hover:bg-gray-700/50 transition-colors"
                    >
                      <p className="text-xs text-gray-400 mb-2">
                        {date.split('-')[1]?.slice(0, 3)}'{date.split('-')[0]?.slice(2)}
                      </p>
                      <p className="text-lg font-bold text-emerald-400">
                        {ensemblePred?.toFixed(1) || bestPred?.toFixed(1) || 'N/A'}M
                      </p>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'models' && forecast && (
          <div className="space-y-8">
            <div>
              <h2 className="text-2xl font-bold text-white">Model Battle</h2>
              <p className="text-gray-400 mt-1">
                Compare model performance and choose the best for your use case
              </p>
            </div>

            <ModelBattle
              comparison={forecast.comparison || []}
              crossValidation={forecast.cross_validation}
            />

            <ModelComparisonChart comparison={forecast.comparison || []} />
          </div>
        )}

        {activeTab === 'scenarios' && (
          <div className="space-y-8">
            <div>
              <h2 className="text-2xl font-bold text-white">Scenario Simulator</h2>
              <p className="text-gray-400 mt-1">
                Test different growth scenarios and their impact on future projections
              </p>
            </div>

            <ScenarioSimulator
              onRunScenario={runScenario}
              onExport={handleExport}
            />
          </div>
        )}

        {activeTab === 'insights' && insights && (
          <div className="space-y-8">
            <div>
              <h2 className="text-2xl font-bold text-white">AI-Powered Insights</h2>
              <p className="text-gray-400 mt-1">
                Deep analysis and intelligent recommendations for your data
              </p>
            </div>

            <InsightsPanel insights={insights} />
          </div>
        )}
      </main>

      <footer className="bg-gray-900/50 border-t border-gray-800 mt-16">
        <div className="max-w-7xl mx-auto px-4 py-8">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-emerald-500/20 rounded-lg">
                <Activity className="w-5 h-5 text-emerald-400" />
              </div>
              <div>
                <p className="font-semibold text-white">UPI Intelligence Platform</p>
                <p className="text-sm text-gray-400">v2.0 - Production Grade</p>
              </div>
            </div>
            <div className="flex items-center gap-6 text-sm text-gray-400">
              <span>Powered by LSTM, ARIMA, Prophet</span>
              <span className="hidden md:inline">•</span>
              <span>Real-time Analytics</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
