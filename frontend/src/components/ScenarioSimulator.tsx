import { useState, useCallback } from 'react';
import { Play, Download, RefreshCw, Zap, Calendar, TrendingUp, HelpCircle, Target } from 'lucide-react';
import type { ScenarioResult } from '../types';

interface ScenarioSimulatorProps {
  onRunScenario: (params: {
    growth_rate: number;
    festive_boost: boolean;
    custom_boost: number;
  }) => Promise<ScenarioResult>;
  onExport?: () => Promise<void>;
}

const scenarioPresets = [
  { name: 'Conservative', growthRate: 3, festiveBoost: true, customBoost: 0, description: 'Slow but steady growth' },
  { name: 'Moderate', growthRate: 8, festiveBoost: true, customBoost: 5, description: 'Expected realistic growth' },
  { name: 'Aggressive', growthRate: 15, festiveBoost: true, customBoost: 10, description: 'High adoption scenario' },
  { name: 'Festival Special', growthRate: 10, festiveBoost: true, customBoost: 15, description: 'Heavy festive season impact' },
];

export function ScenarioSimulator({ onRunScenario, onExport }: ScenarioSimulatorProps) {
  const [growthRate, setGrowthRate] = useState(8);
  const [festiveBoost, setFestiveBoost] = useState(true);
  const [customBoost, setCustomBoost] = useState(5);
  const [scenario, setScenario] = useState<ScenarioResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [showHelp, setShowHelp] = useState(false);

  const runScenario = useCallback(async () => {
    setLoading(true);
    try {
      const result = await onRunScenario({
        growth_rate: growthRate / 100,
        festive_boost: festiveBoost,
        custom_boost: customBoost / 100,
      });
      setScenario(result);
    } catch (error) {
      console.error('Scenario failed:', error);
    } finally {
      setLoading(false);
    }
  }, [growthRate, festiveBoost, customBoost, onRunScenario]);

  const applyPreset = (preset: typeof scenarioPresets[0]) => {
    setGrowthRate(preset.growthRate);
    setFestiveBoost(preset.festiveBoost);
    setCustomBoost(preset.customBoost);
  };

  const handleExport = useCallback(async () => {
    if (!scenario) return;
    try {
      const csvContent = [
        ['Month', 'Predicted Volume (M)'],
        ...scenario.forecast_dates.map((date, i) => [date, scenario.predictions[i].toFixed(2)]),
      ].map(row => row.join(',')).join('\n');

      const blob = new Blob([csvContent], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `upi_scenario_${Date.now()}.csv`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Export failed:', error);
    }
  }, [scenario]);

  return (
    <div className="space-y-6">
      <div className="glass-card p-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 className="text-lg font-semibold text-white flex items-center gap-2">
              <Zap className="w-5 h-5 text-amber-400" />
              What-If Scenario Simulator
            </h3>
            <p className="text-sm text-gray-400 mt-1">
              Adjust parameters to see how different conditions affect future UPI transactions
            </p>
          </div>
          <button
            onClick={() => setShowHelp(!showHelp)}
            className="flex items-center gap-2 px-3 py-2 bg-gray-800 hover:bg-gray-700 text-gray-400 rounded-lg transition-colors"
          >
            <HelpCircle className="w-4 h-4" />
            <span className="text-sm">How it works</span>
          </button>
        </div>

        {showHelp && (
          <div className="mb-6 p-4 bg-blue-500/10 border border-blue-500/20 rounded-xl">
            <h4 className="font-medium text-blue-400 mb-3">Understanding Scenario Parameters</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
              <div>
                <p className="font-medium text-white mb-1">Base Growth Rate</p>
                <p className="text-gray-400">The standard monthly growth percentage. 8% means each month is 8% higher than the previous month.</p>
              </div>
              <div>
                <p className="font-medium text-white mb-1">Custom Adjustment</p>
                <p className="text-gray-400">Additional boost or reduction. Use positive for promotions, negative for economic downturns.</p>
              </div>
              <div>
                <p className="font-medium text-white mb-1">Festive Boost</p>
                <p className="text-gray-400">Adds 15% extra during Oct-Dec (Diwali, Christmas). Indians traditionally spend more during festivals.</p>
              </div>
            </div>
          </div>
        )}

        <div className="mb-6">
          <p className="text-sm text-gray-400 mb-3">Quick Presets</p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {scenarioPresets.map((preset) => (
              <button
                key={preset.name}
                onClick={() => applyPreset(preset)}
                className="p-3 bg-gray-800/50 hover:bg-gray-700/50 border border-gray-700 hover:border-emerald-500/30 rounded-xl text-left transition-all group"
              >
                <p className="font-medium text-white group-hover:text-emerald-400 transition-colors">{preset.name}</p>
                <p className="text-xs text-gray-500 mt-1">{preset.description}</p>
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <label className="text-sm font-medium text-gray-300">Monthly Growth Rate</label>
              <span className="text-lg font-bold text-emerald-400">{growthRate}%</span>
            </div>
            <div className="slider-track">
              <div className="slider-fill" style={{ width: `${growthRate}%` }} />
              <input
                type="range"
                min="0"
                max="20"
                value={growthRate}
                onChange={(e) => setGrowthRate(Number(e.target.value))}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
            </div>
            <p className="text-xs text-gray-500">Expected month-over-month increase</p>
          </div>

          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <label className="text-sm font-medium text-gray-300">Custom Adjustment</label>
              <span className={`text-lg font-bold ${customBoost >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                {customBoost >= 0 ? '+' : ''}{customBoost}%
              </span>
            </div>
            <div className="slider-track">
              <div
                className="slider-fill"
                style={{
                  width: `${50 + customBoost}%`,
                  left: customBoost < 0 ? `${50 + customBoost}%` : '50%',
                  background: customBoost >= 0 ? 'linear-gradient(90deg, #10b981, #059669)' : 'linear-gradient(90deg, #ef4444, #dc2626)',
                }}
              />
              <input
                type="range"
                min="-20"
                max="20"
                value={customBoost}
                onChange={(e) => setCustomBoost(Number(e.target.value))}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
            </div>
            <p className="text-xs text-gray-500">Additional factor (promotions, events)</p>
          </div>

          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <label className="text-sm font-medium text-gray-300">Festive Season (Oct-Dec)</label>
              <span className={`text-lg font-semibold ${festiveBoost ? 'text-emerald-400' : 'text-gray-500'}`}>
                {festiveBoost ? 'ON' : 'OFF'}
              </span>
            </div>
            <div className="flex items-center gap-4">
              <button
                onClick={() => setFestiveBoost(!festiveBoost)}
                className={`toggle-switch ${festiveBoost ? 'toggle-switch-active' : ''}`}
              />
              <p className="text-xs text-gray-500">
                +15% boost for festival months
              </p>
            </div>
          </div>
        </div>

        <div className="flex justify-center">
          <button
            onClick={runScenario}
            disabled={loading}
            className="flex items-center gap-2 px-8 py-3 bg-gradient-to-r from-emerald-500 to-emerald-600 hover:from-emerald-600 hover:to-emerald-700 text-white rounded-xl transition-all disabled:opacity-50 shadow-lg shadow-emerald-500/20"
          >
            {loading ? (
              <RefreshCw className="w-5 h-5 animate-spin" />
            ) : (
              <Play className="w-5 h-5" />
            )}
            <span className="font-medium">Run Simulation</span>
          </button>
        </div>
      </div>

      {scenario && (
        <div className="glass-card p-6 border-amber-500/20">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 bg-amber-500/20 rounded-lg">
              <Target className="w-5 h-5 text-amber-400" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-white">Simulation Results</h3>
              <p className="text-sm text-gray-400">Based on your selected parameters</p>
            </div>
            {scenario && onExport && (
              <button
                onClick={handleExport}
                className="ml-auto flex items-center gap-2 px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors"
              >
                <Download className="w-4 h-4" />
                Export CSV
              </button>
            )}
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-gray-800/50 rounded-xl p-4">
              <div className="flex items-center gap-2 text-gray-400 mb-2">
                <TrendingUp className="w-4 h-4" />
                <span className="text-xs uppercase">Total Year Volume</span>
              </div>
              <p className="text-2xl font-bold text-white">{scenario.predicted_total.toFixed(0)}M</p>
              <p className="text-xs text-gray-500 mt-1">Next 12 months combined</p>
            </div>
            <div className="bg-gray-800/50 rounded-xl p-4">
              <div className="flex items-center gap-2 text-amber-400 mb-2">
                <Calendar className="w-4 h-4" />
                <span className="text-xs uppercase">Peak Month</span>
              </div>
              <p className="text-2xl font-bold text-white">{scenario.predicted_peak_month}</p>
              <p className="text-xs text-gray-500 mt-1">{scenario.predicted_peak.toFixed(1)}M transactions</p>
            </div>
            <div className="bg-gray-800/50 rounded-xl p-4">
              <div className="flex items-center gap-2 text-emerald-400 mb-2">
                <Zap className="w-4 h-4" />
                <span className="text-xs uppercase">Monthly Growth</span>
              </div>
              <p className="text-2xl font-bold text-white">{(scenario.parameters.growth_rate * 100).toFixed(0)}%</p>
              <p className="text-xs text-gray-500 mt-1">Base rate applied</p>
            </div>
            <div className="bg-gray-800/50 rounded-xl p-4">
              <div className="flex items-center gap-2 text-purple-400 mb-2">
                <Zap className="w-4 h-4" />
                <span className="text-xs uppercase">Festive Impact</span>
              </div>
              <p className="text-2xl font-bold text-white">{scenario.parameters.festive_boost ? '+15%' : 'None'}</p>
              <p className="text-xs text-gray-500 mt-1">Q4 2026 boost</p>
            </div>
          </div>

          <div className="bg-gray-800/30 rounded-xl p-4">
            <h4 className="text-sm font-medium text-gray-300 mb-4">Monthly Forecast Breakdown</h4>
            <div className="grid grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
              {scenario.forecast_dates.map((date, i) => {
                const month = parseInt(date.split('-')[1]);
                const isFestive = [10, 11, 12].includes(month) && scenario.parameters.festive_boost;
                const isPeak = scenario.predicted_peak_month?.includes(date.split('-')[1]?.slice(0, 3));
                return (
                  <div
                    key={date}
                    className={`p-3 rounded-lg text-center transition-all ${
                      isPeak 
                        ? 'bg-amber-500/20 border border-amber-500/40' 
                        : isFestive
                        ? 'bg-purple-500/10 border border-purple-500/20'
                        : 'bg-gray-700/30 border border-transparent'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <p className="text-xs text-gray-400">
                        {date.split('-')[1]?.slice(0, 3)}
                      </p>
                      {isPeak && <span className="text-[10px] text-amber-400">Peak</span>}
                    </div>
                    <p className={`font-semibold ${isPeak ? 'text-amber-400' : isFestive ? 'text-purple-400' : 'text-white'}`}>
                      {scenario.predictions[i].toFixed(1)}M
                    </p>
                    {isFestive && <p className="text-[10px] text-purple-400 mt-1">+15%</p>}
                  </div>
                );
              })}
            </div>
          </div>

          <div className="mt-4 flex items-center justify-center gap-6 text-xs text-gray-500">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded bg-purple-500/20 border border-purple-500/20" />
              <span>Festive months (Oct-Dec)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded bg-amber-500/20 border border-amber-500/40" />
              <span>Peak transaction month</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
