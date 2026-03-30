import { Trophy, Medal, Award, TrendingDown, Target, Clock, Zap } from 'lucide-react';
import type { ModelComparison, CrossValidationResult } from '../types';

interface ModelBattleProps {
  comparison: ModelComparison[];
  crossValidation?: Record<string, CrossValidationResult>;
}

const rankIcons = {
  1: Trophy,
  2: Medal,
  3: Award,
};

const modelDescriptions: Record<string, { strength: string; bestFor: string; icon: string }> = {
  lstm: {
    strength: 'Deep Learning',
    bestFor: 'Complex non-linear patterns',
    icon: '🧠',
  },
  prophet: {
    strength: 'Meta AI',
    bestFor: 'Seasonality & holidays',
    icon: '📈',
  },
  arima: {
    strength: 'Statistical',
    bestFor: 'Autoregressive patterns',
    icon: '📊',
  },
  linear_regression: {
    strength: 'Ridge Regularization',
    bestFor: 'Stable baseline',
    icon: '📉',
  },
  moving_average: {
    strength: 'Simple MA',
    bestFor: 'Short-term smoothing',
    icon: '〰️',
  },
};

export function ModelBattle({ comparison, crossValidation }: ModelBattleProps) {
  const sortedModels = [...comparison].sort((a, b) => a.rmse - b.rmse);

  return (
    <div className="glass-card p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-semibold text-white flex items-center gap-2">
            <Trophy className="w-5 h-5 text-amber-400" />
            Model Battle Leaderboard
          </h3>
          <p className="text-sm text-gray-400 mt-1">
            Ranked by RMSE (Root Mean Square Error)
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="px-3 py-1 bg-amber-500/20 text-amber-400 rounded-full text-sm font-medium">
            {sortedModels[0]?.model?.replace('_', ' ').toUpperCase() || 'N/A'}
          </span>
          <span className="text-gray-500">wins</span>
        </div>
      </div>

      <div className="space-y-4">
        {sortedModels.map((model, index) => {
          const rank = index + 1;
          const RankIcon = rankIcons[rank as keyof typeof rankIcons];
          const description = modelDescriptions[model.model];
          const cvResult = crossValidation?.[model.model];
          const isBest = rank === 1;

          return (
            <div
              key={model.model}
              className={`relative p-4 rounded-xl transition-all ${
                isBest
                  ? 'bg-gradient-to-r from-amber-500/10 to-transparent border border-amber-500/30 glow-green'
                  : 'bg-gray-800/50 border border-gray-700/50 hover:border-gray-600'
              }`}
            >
              {isBest && (
                <div className="absolute -top-3 left-4">
                  <div className="bg-gradient-to-r from-amber-400 to-amber-600 text-black text-xs font-bold px-3 py-1 rounded-full flex items-center gap-1">
                    <Trophy className="w-3 h-3" />
                    BEST MODEL
                  </div>
                </div>
              )}

              <div className="flex items-center gap-4">
                <div
                  className={`rank-badge ${
                    isBest
                      ? 'rank-1'
                      : rank === 2
                        ? 'rank-2'
                        : rank === 3
                          ? 'rank-3'
                          : 'rank-default'
                  }`}
                >
                  {RankIcon ? (
                    <RankIcon className="w-4 h-4" />
                  ) : (
                    <span className="text-sm font-bold">#{rank}</span>
                  )}
                </div>

                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-1">
                    <h4 className={`font-semibold text-lg ${isBest ? 'text-emerald-400' : 'text-white'}`}>
                      {description?.icon} {model.model.replace('_', ' ').toUpperCase()}
                    </h4>
                    {description && (
                      <span className="text-xs text-gray-500 bg-gray-700/50 px-2 py-0.5 rounded">
                        {description.strength}
                      </span>
                    )}
                  </div>
                  {description && (
                    <p className="text-sm text-gray-400">{description.bestFor}</p>
                  )}
                </div>

                <div className="flex gap-6">
                  <div className="text-center">
                    <div className="flex items-center gap-1 text-red-400 mb-1">
                      <TrendingDown className="w-3 h-3" />
                      <span className="text-xs uppercase">RMSE</span>
                    </div>
                    <p className={`text-xl font-bold ${isBest ? 'text-emerald-400' : 'text-white'}`}>
                      {model.rmse.toFixed(2)}
                    </p>
                  </div>
                  <div className="text-center">
                    <div className="flex items-center gap-1 text-blue-400 mb-1">
                      <Target className="w-3 h-3" />
                      <span className="text-xs uppercase">MAE</span>
                    </div>
                    <p className="text-xl font-bold text-white">{model.mae.toFixed(2)}</p>
                  </div>
                  <div className="text-center">
                    <div className="flex items-center gap-1 text-purple-400 mb-1">
                      <Clock className="w-3 h-3" />
                      <span className="text-xs uppercase">MAPE</span>
                    </div>
                    <p className="text-xl font-bold text-white">{model.mape.toFixed(1)}%</p>
                  </div>
                </div>
              </div>

              {cvResult && (
                <div className="mt-4 pt-4 border-t border-gray-700/50">
                  <div className="flex items-center gap-2 mb-2">
                    <Zap className="w-4 h-4 text-amber-400" />
                    <span className="text-sm font-medium text-gray-300">Cross-Validation Results</span>
                  </div>
                  <div className="grid grid-cols-4 gap-4">
                    <div>
                      <p className="text-xs text-gray-500">Mean RMSE</p>
                      <p className="text-sm font-semibold text-white">{cvResult.mean_rmse.toFixed(2)}</p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-500">Std RMSE</p>
                      <p className="text-sm font-semibold text-white">{cvResult.std_rmse.toFixed(2)}</p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-500">Folds</p>
                      <p className="text-sm font-semibold text-white">{cvResult.fold_metrics.length}</p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-500">Consistency</p>
                      <p className={`text-sm font-semibold ${
                        cvResult.std_rmse < 2 ? 'text-emerald-400' : cvResult.std_rmse < 4 ? 'text-amber-400' : 'text-red-400'
                      }`}>
                        {cvResult.std_rmse < 2 ? 'High' : cvResult.std_rmse < 4 ? 'Medium' : 'Low'}
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {isBest && (
                <div className="mt-4">
                  <div className="flex gap-2">
                    <div className="flex-1 h-2 bg-gray-700 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-emerald-500 to-emerald-400 rounded-full"
                        style={{ width: `${Math.max(20, 100 - sortedModels[0].rmse * 5)}%` }}
                      />
                    </div>
                    <span className="text-xs text-emerald-400 font-medium">
                      {Math.max(20, 100 - sortedModels[0].rmse * 5).toFixed(0)}% accuracy
                    </span>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
