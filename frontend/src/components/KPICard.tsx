import { useEffect, useState, useRef } from 'react';
import { TrendingUp, TrendingDown, DollarSign, BarChart3, Brain, Target, Activity, Award, Info } from 'lucide-react';

interface AnimatedNumberProps {
  value: number;
  prefix?: string;
  suffix?: string;
  decimals?: number;
  duration?: number;
}

function AnimatedNumber({ value, prefix = '', suffix = '', decimals = 0, duration = 1000 }: AnimatedNumberProps) {
  const [displayValue, setDisplayValue] = useState(0);
  const startTimeRef = useRef<number | null>(null);
  const animationRef = useRef<number | null>(null);

  useEffect(() => {
    startTimeRef.current = null;

    const animate = (timestamp: number) => {
      if (!startTimeRef.current) {
        startTimeRef.current = timestamp;
      }

      const progress = Math.min((timestamp - startTimeRef.current) / duration, 1);
      const easeOut = 1 - Math.pow(1 - progress, 3);
      const current = value * easeOut;

      setDisplayValue(current);

      if (progress < 1) {
        animationRef.current = requestAnimationFrame(animate);
      }
    };

    animationRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [value, duration]);

  const formatted = decimals > 0 ? displayValue.toFixed(decimals) : Math.round(displayValue).toLocaleString();

  return (
    <span className="counter-animation">
      {prefix}{formatted}{suffix}
    </span>
  );
}

interface KPICardProps {
  title: string;
  value: number;
  subtitle?: string;
  trend?: number;
  icon: 'volume' | 'value' | 'growth' | 'period' | 'model' | 'confidence' | 'prediction' | 'best';
  prefix?: string;
  suffix?: string;
  decimals?: number;
  isHighlight?: boolean;
  description?: string;
}

const iconMap = {
  volume: BarChart3,
  value: DollarSign,
  growth: TrendingUp,
  period: Activity,
  model: Brain,
  confidence: Target,
  prediction: Award,
  best: Award,
};

const iconColors = {
  volume: 'from-emerald-500 to-emerald-600',
  value: 'from-amber-500 to-amber-600',
  growth: 'from-emerald-500 to-emerald-600',
  period: 'from-blue-500 to-blue-600',
  model: 'from-purple-500 to-purple-600',
  confidence: 'from-cyan-500 to-cyan-600',
  prediction: 'from-indigo-500 to-indigo-600',
  best: 'from-yellow-500 to-yellow-600',
};

const cardDescriptions: Record<string, string> = {
  volume: "Total UPI transactions in the most recent month",
  growth: "Year-over-year growth compared to same month last year",
  prediction: "AI-predicted transaction volume for next month",
  confidence: "How accurate our models are based on historical data",
};

export function KPICard({
  title,
  value,
  subtitle,
  trend,
  icon,
  prefix = '',
  suffix = '',
  decimals = 0,
  isHighlight = false,
  description
}: KPICardProps) {
  const Icon = iconMap[icon];
  const colorClass = iconColors[icon];
  const isTrendUp = trend !== undefined && trend >= 0;
  const cardDescription = description || cardDescriptions[icon] || '';

  return (
    <div className={`kpi-card ${isHighlight ? 'border-emerald-500/30 glow-green' : ''} relative group`}>
      <div className="flex items-start justify-between mb-3">
        <div className={`p-3 rounded-xl bg-gradient-to-br ${colorClass} shadow-lg`}>
          <Icon className="w-6 h-6 text-white" />
        </div>
        {trend !== undefined && (
          <div className={`flex items-center gap-1 text-sm font-semibold px-3 py-1.5 rounded-full ${
            isTrendUp ? 'trend-up' : 'trend-down'
          }`}>
            {isTrendUp ? (
              <TrendingUp className="w-4 h-4" />
            ) : (
              <TrendingDown className="w-4 h-4" />
            )}
            <span>{Math.abs(trend).toFixed(1)}%</span>
          </div>
        )}
      </div>

      <h3 className="text-sm font-medium text-gray-300 mb-1">{title}</h3>

      <div className={`kpi-value mb-2 ${isHighlight ? 'kpi-value-accent' : ''}`}>
        <AnimatedNumber
          value={value}
          prefix={prefix}
          suffix={suffix}
          decimals={decimals}
          duration={1200}
        />
      </div>

      {subtitle && (
        <p className="text-xs text-gray-500 mb-2">{subtitle}</p>
      )}

      {cardDescription && (
        <div className="flex items-start gap-2 pt-2 border-t border-gray-700/50 mt-2">
          <Info className="w-3.5 h-3.5 text-gray-500 mt-0.5 flex-shrink-0" />
          <p className="text-xs text-gray-400 leading-relaxed">{cardDescription}</p>
        </div>
      )}

      {isHighlight && (
        <div className="absolute -top-px left-4 right-4 h-px bg-gradient-to-r from-transparent via-emerald-500 to-transparent opacity-50" />
      )}
    </div>
  );
}

interface MiniStatProps {
  label: string;
  value: string | number;
  color?: 'green' | 'blue' | 'purple' | 'amber';
}

export function MiniStat({ label, value, color = 'green' }: MiniStatProps) {
  const colorClasses = {
    green: 'text-emerald-400',
    blue: 'text-blue-400',
    purple: 'text-purple-400',
    amber: 'text-amber-400',
  };

  return (
    <div className="flex flex-col">
      <span className="text-xs text-gray-500 uppercase tracking-wide">{label}</span>
      <span className={`text-lg font-semibold ${colorClasses[color]}`}>{value}</span>
    </div>
  );
}
