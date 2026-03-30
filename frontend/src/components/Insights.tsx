import { Lightbulb, TrendingUp, Calendar, Award, AlertCircle, ArrowRight, Rocket, GitBranch, CheckCircle, Gift, Activity, BookOpen, Sparkles } from 'lucide-react';
import type { Insights, AIInsight } from '../types';

interface InsightsPanelProps {
  insights: Insights;
}

const iconMap: Record<string, typeof Rocket> = {
  acceleration: Rocket,
  deceleration: TrendingUp,
  divergence: GitBranch,
  consensus: CheckCircle,
  seasonal: Gift,
  volatility: Activity,
};

const insightColors = {
  acceleration: { bg: 'bg-emerald-500/10', border: 'border-emerald-500/30', icon: 'text-emerald-400' },
  deceleration: { bg: 'bg-amber-500/10', border: 'border-amber-500/30', icon: 'text-amber-400' },
  divergence: { bg: 'bg-purple-500/10', border: 'border-purple-500/30', icon: 'text-purple-400' },
  consensus: { bg: 'bg-blue-500/10', border: 'border-blue-500/30', icon: 'text-blue-400' },
  seasonal: { bg: 'bg-amber-500/10', border: 'border-amber-500/30', icon: 'text-amber-400' },
  volatility: { bg: 'bg-red-500/10', border: 'border-red-500/30', icon: 'text-red-400' },
};

export function InsightsPanel({ insights }: InsightsPanelProps) {
  const renderAIInsight = (insight: AIInsight) => {
    const colors = insightColors[insight.type] || insightColors.consensus;
    const Icon = iconMap[insight.type] || Sparkles;

    return (
      <div
        key={insight.title}
        className={`${colors.bg} border ${colors.border} rounded-xl p-4 hover:scale-[1.02] transition-transform`}
      >
        <div className="flex items-start gap-3">
          <div className={`p-2 rounded-lg bg-black/20 ${colors.icon}`}>
            <Icon className="w-5 h-5" />
          </div>
          <div className="flex-1">
            <div className="flex items-center justify-between mb-1">
              <h4 className="font-semibold text-white">{insight.title}</h4>
              <span className={`text-xs px-2 py-0.5 rounded-full ${
                insight.confidence === 'high'
                  ? 'bg-emerald-500/20 text-emerald-400'
                  : insight.confidence === 'medium'
                    ? 'bg-amber-500/20 text-amber-400'
                    : 'bg-red-500/20 text-red-400'
              }`}>
                {insight.confidence}
              </span>
            </div>
            <p className="text-sm text-gray-300 mb-2">{insight.description}</p>
            <div className="flex items-center gap-2 text-xs text-gray-400 bg-black/20 rounded-lg px-3 py-2">
              <Sparkles className="w-3 h-3 text-emerald-400" />
              <span>{insight.action}</span>
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      {insights.narrative && insights.narrative.story && (
        <div className="glass-card-glow p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-gradient-to-br from-amber-500 to-amber-600 rounded-lg">
              <BookOpen className="w-5 h-5 text-white" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-white">UPI Growth Story</h3>
              <p className="text-sm text-gray-400">The complete journey of India's digital payments</p>
            </div>
          </div>

          <div className="prose prose-invert max-w-none">
            <p className="text-gray-300 leading-relaxed">{insights.narrative.story}</p>
          </div>

          <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-black/20 rounded-xl p-4 text-center">
              <p className="text-xs text-gray-400 uppercase mb-2">Key Milestone</p>
              <p className="text-2xl font-bold text-gradient-green">
                {insights.narrative.key_milestone}
              </p>
            </div>
            <div className="bg-black/20 rounded-xl p-4 text-center">
              <p className="text-xs text-gray-400 uppercase mb-2">Next Prediction</p>
              <p className="text-2xl font-bold text-cyan-400">
                {insights.narrative.next_prediction}
              </p>
            </div>
            <div className="bg-black/20 rounded-xl p-4 text-center">
              <p className="text-xs text-gray-400 uppercase mb-2">Confidence</p>
              <p className={`text-2xl font-bold ${
                insights.narrative.confidence_level === 'high'
                  ? 'text-emerald-400'
                  : insights.narrative.confidence_level === 'medium'
                    ? 'text-amber-400'
                    : 'text-red-400'
              }`}>
                {insights.narrative.confidence_level.toUpperCase()}
              </p>
            </div>
          </div>

          {insights.narrative.growth_stages.length > 0 && (
            <div className="mt-6">
              <h4 className="text-sm font-medium text-gray-300 mb-3">Growth Milestones</h4>
              <div className="space-y-3">
                {insights.narrative?.growth_stages?.map((stage, i) => (
                  <div key={i} className="flex items-start gap-4">
                    <div className="relative">
                      <div className="w-8 h-8 rounded-full bg-emerald-500/20 border-2 border-emerald-500/50 flex items-center justify-center">
                        <span className="text-xs font-bold text-emerald-400">{i + 1}</span>
                      </div>
                      {i < (insights.narrative?.growth_stages?.length || 0) - 1 && (
                        <div className="absolute top-8 left-1/2 w-0.5 h-8 bg-emerald-500/30 -translate-x-1/2" />
                      )}
                    </div>
                    <div className="flex-1 pb-4">
                      <p className="text-xs text-gray-500">{stage.period}</p>
                      <p className="font-medium text-white">{stage.event}</p>
                      <p className="text-sm text-gray-400 mt-1">{stage.description}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {insights.ai_insights && insights.ai_insights.length > 0 && (
        <div className="glass-card p-6">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 bg-gradient-to-br from-purple-500 to-purple-600 rounded-lg">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-white">AI-Powered Insights</h3>
              <p className="text-sm text-gray-400">Real-time analysis of your data</p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {insights.ai_insights.map(renderAIInsight)}
          </div>
        </div>
      )}

      <div className="glass-card p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 bg-gradient-to-br from-emerald-500 to-emerald-600 rounded-lg">
            <Lightbulb className="w-5 h-5 text-white" />
          </div>
          <h3 className="text-lg font-semibold text-white">Executive Summary</h3>
        </div>
        <p className="text-gray-300 leading-relaxed">{insights.summary}</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="insight-card">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-green-500/20 rounded-lg">
              <TrendingUp className="w-5 h-5 text-green-400" />
            </div>
            <h3 className="text-lg font-semibold text-white">Growth Trends</h3>
          </div>
          <ul className="space-y-3">
            {insights.trends.map((trend, i) => (
              <li key={i} className="flex items-start gap-3 text-gray-300">
                <ArrowRight className="w-5 h-5 text-emerald-400 mt-0.5 flex-shrink-0" />
                <span>{trend}</span>
              </li>
            ))}
          </ul>
        </div>

        <div className="insight-card insight-card-accent">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-blue-500/20 rounded-lg">
              <Calendar className="w-5 h-5 text-blue-400" />
            </div>
            <h3 className="text-lg font-semibold text-white">Seasonality Analysis</h3>
          </div>
          <ul className="space-y-3">
            {insights.seasonality.map((item, i) => (
              <li key={i} className="flex items-start gap-3 text-gray-300">
                <ArrowRight className="w-5 h-5 text-blue-400 mt-0.5 flex-shrink-0" />
                <span>{item}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      {insights.forecast_explanation && (
        <div className="insight-card">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-cyan-500/20 rounded-lg">
              <Award className="w-5 h-5 text-cyan-400" />
            </div>
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-white">Forecast Explanation</h3>
            </div>
            <span className={`badge ${
              insights.forecast_explanation.risk_flag === 'low'
                ? 'badge-confidence-high'
                : insights.forecast_explanation.risk_flag === 'medium'
                  ? 'badge-confidence-medium'
                  : 'badge-confidence-low'
            }`}>
              {insights.forecast_explanation.risk_flag.toUpperCase()} RISK
            </span>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
            <div className="bg-black/20 rounded-lg p-3">
              <p className="text-xs text-gray-400 mb-1">Trend</p>
              <p className="text-sm font-medium text-white capitalize">{insights.forecast_explanation.trend}</p>
            </div>
            <div className="bg-black/20 rounded-lg p-3">
              <p className="text-xs text-gray-400 mb-1">Seasonality</p>
              <p className="text-sm font-medium text-white">
                {insights.forecast_explanation.seasonal_factor > 1 ? '+' : ''}
                {((insights.forecast_explanation.seasonal_factor - 1) * 100).toFixed(0)}%
              </p>
            </div>
            <div className="bg-black/20 rounded-lg p-3">
              <p className="text-xs text-gray-400 mb-1">Best Model</p>
              <p className="text-sm font-medium text-emerald-400 uppercase">
                {insights.forecast_explanation.best_model?.replace('_', ' ')}
              </p>
            </div>
            <div className="bg-black/20 rounded-lg p-3">
              <p className="text-xs text-gray-400 mb-1">Confidence</p>
              <p className="text-sm font-medium text-white">
                {insights.forecast_explanation.confidence_reason}
              </p>
            </div>
          </div>

          <p className="text-sm text-gray-400">{insights.forecast_explanation.confidence_reason}</p>
        </div>
      )}

      {insights.model_comparison && (
        <div className="insight-card insight-card-accent">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-purple-500/20 rounded-lg">
              <Award className="w-5 h-5 text-purple-400" />
            </div>
            <h3 className="text-lg font-semibold text-white">Model Performance</h3>
            <span className="ml-auto badge badge-best">
              #{insights.model_comparison.best_model?.replace('_', ' ').toUpperCase()}
            </span>
          </div>
          <ul className="space-y-3">
            {insights.model_comparison.insights?.map((item, i) => (
              <li key={i} className="flex items-start gap-3 text-gray-300">
                <ArrowRight className="w-5 h-5 text-purple-400 mt-0.5 flex-shrink-0" />
                <span>{item}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="insight-card insight-card-warning">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 bg-amber-500/20 rounded-lg">
            <AlertCircle className="w-5 h-5 text-amber-400" />
          </div>
          <h3 className="text-lg font-semibold text-white">Recommendations</h3>
        </div>
        <ul className="space-y-3">
          {insights.recommendations.map((rec, i) => (
            <li key={i} className="flex items-start gap-3 text-gray-300">
              <ArrowRight className="w-5 h-5 text-amber-400 mt-0.5 flex-shrink-0" />
              <span>{rec}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
