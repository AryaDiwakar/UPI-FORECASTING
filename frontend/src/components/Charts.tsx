import {
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  BarChart,
  Bar,
  Legend,
  ComposedChart,
  ReferenceLine,
  Scatter,
} from 'recharts';
import type { TimeSeriesData, ForecastData, EnsembleData, Anomaly } from '../types';

interface TransactionChartProps {
  data: TimeSeriesData;
  timeRange?: '1y' | '3y' | 'all';
  anomalies?: Anomaly[];
  onTimeRangeChange?: (range: '1y' | '3y' | 'all') => void;
}

export function TransactionChart({
  data,
  timeRange = 'all',
  anomalies = [],
  onTimeRangeChange
}: TransactionChartProps) {
  const anomalyDates = new Set(anomalies.map(a => a.date));

  let filteredData = data.dates.map((date, i) => ({
    date,
    volume: data.volume[i],
    value: data.value[i] / 100,
    isAnomaly: anomalyDates.has(date),
  }));

  if (timeRange === '1y') {
    filteredData = filteredData.slice(-12);
  } else if (timeRange === '3y') {
    filteredData = filteredData.slice(-36);
  }

  const anomalyData = filteredData.filter(d => d.isAnomaly);

  return (
    <div className="chart-container">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-semibold text-white">UPI Transaction Trends</h3>
          <p className="text-sm text-gray-400 mt-1">Monthly volume and value analysis</p>
        </div>
        {onTimeRangeChange && (
          <div className="flex gap-2">
            {(['1y', '3y', 'all'] as const).map((range) => (
              <button
                key={range}
                onClick={() => onTimeRangeChange(range)}
                className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                  timeRange === range
                    ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                    : 'bg-gray-800/50 text-gray-400 border border-gray-700/50 hover:text-white'
                }`}
              >
                {range === '1y' ? '1Y' : range === '3y' ? '3Y' : 'All'}
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={filteredData}>
            <defs>
              <linearGradient id="volumeGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#10b981" stopOpacity={0.4} />
                <stop offset="95%" stopColor="#10b981" stopOpacity={0.05} />
              </linearGradient>
              <linearGradient id="valueGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#f59e0b" stopOpacity={0.05} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.5} />
            <XAxis
              dataKey="date"
              stroke="#9ca3af"
              tick={{ fill: '#9ca3af', fontSize: 11 }}
              tickFormatter={(value) => {
                const parts = value.split('-');
                return `${parts[1]?.slice(0, 3)} '${parts[0]?.slice(2)}`;
              }}
              interval="preserveStartEnd"
            />
            <YAxis
              yAxisId="left"
              stroke="#9ca3af"
              tick={{ fill: '#9ca3af', fontSize: 11 }}
              tickFormatter={(value) => `${value}M`}
            />
            <YAxis
              yAxisId="right"
              orientation="right"
              stroke="#9ca3af"
              tick={{ fill: '#9ca3af', fontSize: 11 }}
              tickFormatter={(value) => `₹${value}K`}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: 'rgba(17, 24, 39, 0.95)',
                border: '1px solid rgba(75, 85, 99, 0.5)',
                borderRadius: '12px',
                backdropFilter: 'blur(10px)',
              }}
              labelStyle={{ color: '#f3f4f6', fontWeight: 600 }}
              itemStyle={{ color: '#9ca3af' }}
              formatter={(value, name) => [
                name === 'volume' ? `${Number(value).toFixed(1)}M` : `₹${Number(value).toFixed(0)}K Cr`,
                name === 'volume' ? 'Volume' : 'Value'
              ]}
            />
            <Legend
              wrapperStyle={{ paddingTop: '20px' }}
              formatter={(value) => <span className="text-gray-300">{value}</span>}
            />
            <Area
              yAxisId="left"
              type="monotone"
              dataKey="volume"
              stroke="#10b981"
              fill="url(#volumeGradient)"
              name="Volume (M)"
              strokeWidth={2}
            />
            <Line
              yAxisId="right"
              type="monotone"
              dataKey="value"
              stroke="#f59e0b"
              strokeWidth={2}
              dot={false}
              name="Value (₹K Cr)"
            />
            <Scatter
              yAxisId="left"
              dataKey="volume"
              data={anomalyData}
              fill="#ef4444"
              shape={(props: any) => {
                const { cx, cy } = props;
                return (
                  <g>
                    <circle cx={cx} cy={cy} r={8} fill="#ef4444" opacity={0.8} />
                    <circle cx={cx} cy={cy} r={4} fill="#fff" />
                  </g>
                );
              }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

interface ForecastComparisonProps {
  historical: TimeSeriesData;
  forecast: ForecastData;
  ensemble?: EnsembleData;
  selectedModels?: string[];
  onModelToggle?: (model: string) => void;
  showConfidence?: boolean;
}

const modelColors: Record<string, string> = {
  lstm: '#10b981',
  arima: '#3b82f6',
  linear_regression: '#f59e0b',
  moving_average: '#8b5cf6',
  prophet: '#ec4899',
  ensemble: '#06b6d4',
};

const modelLabels: Record<string, string> = {
  lstm: 'LSTM',
  arima: 'ARIMA',
  linear_regression: 'Ridge Reg',
  moving_average: 'MA',
  prophet: 'Prophet',
  ensemble: 'Ensemble',
};

export function ForecastComparison({
  historical,
  forecast,
  ensemble,
  selectedModels = ['lstm', 'arima', 'prophet'],
  onModelToggle,
  showConfidence = true,
}: ForecastComparisonProps) {
  const historicalChartData = historical.dates.slice(-24).map((date) => {
    const idx = historical.dates.indexOf(date);
    return {
      date,
      actual: historical.volume[idx],
    };
  });

  const forecastChartData = forecast.forecast_dates.map((date, i) => {
    const data: Record<string, unknown> = { date };

    if (ensemble && selectedModels.includes('ensemble')) {
      data['ensemble'] = ensemble.ensemble_predictions[i];
      if (showConfidence) {
        data['ensemble_lower'] = ensemble.confidence_lower[i];
        data['ensemble_upper'] = ensemble.confidence_upper[i];
      }
    }

    Object.entries(forecast.models).forEach(([modelName, modelData]) => {
      if (selectedModels.includes(modelName)) {
        data[modelName] = modelData.predictions[i];
        if (showConfidence && modelData.confidence_lower && modelData.confidence_upper) {
          data[`${modelName}_lower`] = modelData.confidence_lower[i];
          data[`${modelName}_upper`] = modelData.confidence_upper[i];
        }
      }
    });

    return data;
  });

  const allDates = [...historicalChartData.map(d => d.date), ...forecast.forecast_dates];

  return (
    <div className="chart-container">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-semibold text-white">Multi-Model Forecast Comparison</h3>
          <p className="text-sm text-gray-400 mt-1">Compare predictions across different models</p>
        </div>
      </div>

      {onModelToggle && (
        <div className="flex flex-wrap gap-2 mb-4">
          {Object.keys(forecast.models).map((model) => (
            <button
              key={model}
              onClick={() => onModelToggle(model)}
              className={`model-toggle ${selectedModels.includes(model) ? 'model-toggle-active' : ''}`}
              style={{
                borderColor: selectedModels.includes(model) ? modelColors[model] : undefined,
              }}
            >
              <span
                className="inline-block w-3 h-3 rounded-full mr-2"
                style={{ backgroundColor: modelColors[model] }}
              />
              {modelLabels[model] || model.replace('_', ' ')}
            </button>
          ))}
          {ensemble && (
            <button
              onClick={() => onModelToggle('ensemble')}
              className={`model-toggle ${selectedModels.includes('ensemble') ? 'model-toggle-active' : ''}`}
              style={{
                borderColor: selectedModels.includes('ensemble') ? modelColors.ensemble : undefined,
              }}
            >
              <span
                className="inline-block w-3 h-3 rounded-full mr-2"
                style={{ backgroundColor: modelColors.ensemble }}
              />
              Ensemble
            </button>
          )}
        </div>
      )}

      <div className="h-96">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={[...historicalChartData, ...forecastChartData]}>
            <defs>
              {Object.entries(modelColors).map(([model, color]) => (
                <linearGradient key={model} id={`gradient-${model}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={color} stopOpacity={0.2} />
                  <stop offset="95%" stopColor={color} stopOpacity={0.05} />
                </linearGradient>
              ))}
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.5} />
            <XAxis
              dataKey="date"
              stroke="#9ca3af"
              tick={{ fill: '#9ca3af', fontSize: 10 }}
              tickFormatter={(value) => {
                if (allDates.indexOf(value) < historicalChartData.length) {
                  return value.split('-')[1]?.slice(0, 3) + "'" + value.split('-')[0]?.slice(2);
                }
                return value.split('-')[1]?.slice(0, 3) + "'" + value.split('-')[0]?.slice(2);
              }}
              interval="preserveStartEnd"
            />
            <YAxis
              stroke="#9ca3af"
              tick={{ fill: '#9ca3af', fontSize: 11 }}
              tickFormatter={(value) => `${value}M`}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: 'rgba(17, 24, 39, 0.95)',
                border: '1px solid rgba(75, 85, 99, 0.5)',
                borderRadius: '12px',
              }}
              labelStyle={{ color: '#f3f4f6', fontWeight: 600 }}
            />
            <Legend
              wrapperStyle={{ paddingTop: '10px' }}
              formatter={(value) => <span className="text-gray-300">{modelLabels[value] || value}</span>}
            />
            <ReferenceLine
              x={forecast.forecast_dates[0]}
              stroke="#6b7280"
              strokeDasharray="5 5"
              label={{
                value: 'Forecast',
                position: 'top',
                fill: '#6b7280',
                fontSize: 11,
              }}
            />

            <Line
              type="monotone"
              dataKey="actual"
              stroke="#fff"
              strokeWidth={2}
              dot={false}
              name="Actual"
              connectNulls={false}
            />

            {selectedModels.includes('ensemble') && ensemble && (
              <>
                <Area
                  type="monotone"
                  dataKey="ensemble_upper"
                  stroke="transparent"
                  fill={modelColors.ensemble}
                  fillOpacity={0.1}
                  name="Ensemble Upper"
                  connectNulls={false}
                />
                <Area
                  type="monotone"
                  dataKey="ensemble_lower"
                  stroke="transparent"
                  fill={modelColors.ensemble}
                  fillOpacity={0.1}
                  name="Ensemble Lower"
                  connectNulls={false}
                />
                <Line
                  type="monotone"
                  dataKey="ensemble"
                  stroke={modelColors.ensemble}
                  strokeWidth={3}
                  dot={false}
                  name="ensemble"
                  strokeDasharray="5 5"
                  connectNulls={false}
                />
              </>
            )}

            {Object.entries(forecast.models).map(([modelName]) => {
              if (!selectedModels.includes(modelName)) return null;
              const color = modelColors[modelName] || '#6b7280';

              return (
                <Line
                  key={modelName}
                  type="monotone"
                  dataKey={modelName}
                  stroke={color}
                  strokeWidth={2}
                  dot={false}
                  name={modelName}
                  connectNulls={false}
                />
              );
            })}
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-4 p-4 bg-gray-800/50 rounded-xl">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-6">
            <div>
              <span className="text-xs text-gray-400 uppercase">Best Model</span>
              <p className="text-emerald-400 font-semibold">
                {forecast.best_model?.replace('_', ' ').toUpperCase()}
              </p>
            </div>
            <div>
              <span className="text-xs text-gray-400 uppercase">RMSE</span>
              <p className="text-white font-semibold">
                {forecast.comparison?.[0]?.rmse?.toFixed(2) || 'N/A'}
              </p>
            </div>
            {ensemble && (
              <div>
                <span className="text-xs text-gray-400 uppercase">Confidence</span>
                <p className="text-cyan-400 font-semibold">{ensemble.confidence_score}%</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

interface ModelComparisonChartProps {
  comparison: { model: string; rmse: number; mae: number; mape: number; rank?: number }[];
}

export function ModelComparisonChart({ comparison }: ModelComparisonChartProps) {
  const data = comparison.map((c) => ({
    name: modelLabels[c.model] || c.model.replace('_', ' '),
    RMSE: c.rmse,
    MAE: c.mae,
    MAPE: c.mape,
    rank: c.rank || 0,
  }));

  return (
    <div className="chart-container">
      <h3 className="text-lg font-semibold text-white mb-6">Model Performance Comparison</h3>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.5} />
            <XAxis type="number" stroke="#9ca3af" tick={{ fill: '#9ca3af', fontSize: 11 }} />
            <YAxis
              type="category"
              dataKey="name"
              stroke="#9ca3af"
              tick={{ fill: '#9ca3af', fontSize: 11 }}
              width={80}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: 'rgba(17, 24, 39, 0.95)',
                border: '1px solid rgba(75, 85, 99, 0.5)',
                borderRadius: '12px',
              }}
            />
            <Legend formatter={(value) => <span className="text-gray-300">{value}</span>} />
            <Bar dataKey="RMSE" fill="#10b981" radius={[0, 4, 4, 0]} name="RMSE" />
            <Bar dataKey="MAE" fill="#3b82f6" radius={[0, 4, 4, 0]} name="MAE" />
            <Bar dataKey="MAPE" fill="#8b5cf6" radius={[0, 4, 4, 0]} name="MAPE %" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

interface ConfidenceChartProps {
  forecast: ForecastData;
  ensemble?: EnsembleData;
}

export function ConfidenceChart({ forecast, ensemble }: ConfidenceChartProps) {
  const bestModel = forecast.best_model;
  const bestData = bestModel ? forecast.models[bestModel] : null;

  if (!bestData) return null;

  const data = forecast.forecast_dates.map((date, i) => ({
    date,
    prediction: bestData.predictions[i],
    upper: bestData.confidence_upper?.[i] || bestData.predictions[i] * 1.1,
    lower: bestData.confidence_lower?.[i] || bestData.predictions[i] * 0.9,
    ensembleUpper: ensemble?.confidence_upper?.[i],
    ensembleLower: ensemble?.confidence_lower?.[i],
    ensemble: ensemble?.ensemble_predictions?.[i],
  }));

  return (
    <div className="chart-container">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-semibold text-white">Prediction Confidence Interval</h3>
          <p className="text-sm text-gray-400 mt-1">
            {bestModel?.toUpperCase()} - 95% Confidence Band
          </p>
        </div>
        <div className="badge badge-confidence-high">
          <span className="w-2 h-2 rounded-full bg-emerald-400 mr-2" />
          High Confidence
        </div>
      </div>

      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data}>
            <defs>
              <linearGradient id="confidenceGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#10b981" stopOpacity={0.4} />
                <stop offset="95%" stopColor="#10b981" stopOpacity={0.1} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.5} />
            <XAxis
              dataKey="date"
              stroke="#9ca3af"
              tick={{ fill: '#9ca3af', fontSize: 10 }}
              tickFormatter={(value) => value.split('-')[1]?.slice(0, 3)}
            />
            <YAxis
              stroke="#9ca3af"
              tick={{ fill: '#9ca3af', fontSize: 11 }}
              tickFormatter={(value) => `${value}M`}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: 'rgba(17, 24, 39, 0.95)',
                border: '1px solid rgba(75, 85, 99, 0.5)',
                borderRadius: '12px',
              }}
              formatter={(value, name) => [
                `${Number(value).toFixed(1)}M`,
                name === 'prediction' ? 'Forecast' : name === 'upper' ? 'Upper Bound' : 'Lower Bound'
              ]}
            />
            <Legend formatter={(value) => <span className="text-gray-300">{value}</span>} />
            <Area
              type="monotone"
              dataKey="upper"
              stroke="transparent"
              fill="#10b981"
              fillOpacity={0.15}
              name="Upper Bound"
            />
            <Area
              type="monotone"
              dataKey="lower"
              stroke="transparent"
              fill="#10b981"
              fillOpacity={0.15}
              name="Lower Bound"
            />
            <Line
              type="monotone"
              dataKey="prediction"
              stroke="#10b981"
              strokeWidth={3}
              dot={{ fill: '#10b981', strokeWidth: 2, r: 4 }}
              name="Forecast"
            />
            {ensemble && (
              <Line
                type="monotone"
                dataKey="ensemble"
                stroke="#06b6d4"
                strokeWidth={2}
                strokeDasharray="5 5"
                dot={false}
                name="Ensemble"
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
