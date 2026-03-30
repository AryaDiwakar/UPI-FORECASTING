import { AlertTriangle } from 'lucide-react';
import type { Anomaly } from '../types';

interface AnomaliesProps {
  anomalies: Anomaly[];
}

export function AnomaliesPanel({ anomalies }: AnomaliesProps) {
  if (anomalies.length === 0) {
    return (
      <div className="bg-fintech-800 rounded-xl p-6 border border-fintech-700">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 bg-green-500/20 rounded-lg">
            <AlertTriangle className="w-5 h-5 text-green-400" />
          </div>
          <h3 className="text-lg font-semibold text-white">Anomaly Detection</h3>
        </div>
        <p className="text-fintech-400">No anomalies detected within threshold.</p>
      </div>
    );
  }

  return (
    <div className="bg-fintech-800 rounded-xl p-6 border border-fintech-700">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 bg-amber-500/20 rounded-lg">
          <AlertTriangle className="w-5 h-5 text-amber-400" />
        </div>
        <h3 className="text-lg font-semibold text-white">Anomaly Detection</h3>
        <span className="ml-auto bg-amber-500/20 text-amber-400 px-3 py-1 rounded-full text-sm">
          {anomalies.length} detected
        </span>
      </div>
      
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-fintech-700">
              <th className="text-left py-3 px-4 text-fintech-400 font-medium">Month</th>
              <th className="text-right py-3 px-4 text-fintech-400 font-medium">Volume (M)</th>
              <th className="text-right py-3 px-4 text-fintech-400 font-medium">Value (Cr)</th>
              <th className="text-right py-3 px-4 text-fintech-400 font-medium">Z-Score</th>
            </tr>
          </thead>
          <tbody>
            {anomalies.map((anomaly, i) => (
              <tr key={i} className="border-b border-fintech-700/50 hover:bg-fintech-700/30">
                <td className="py-3 px-4 text-white font-medium">{anomaly.month}</td>
                <td className="py-3 px-4 text-right text-fintech-300">
                  {anomaly.volume.toFixed(2)}M
                </td>
                <td className="py-3 px-4 text-right text-fintech-300">
                  ₹{anomaly.value.toFixed(0)}Cr
                </td>
                <td className="py-3 px-4 text-right">
                  <span className={`px-2 py-1 rounded text-sm ${
                    Math.abs(anomaly.volume_zscore) > 2.5 
                      ? 'bg-red-500/20 text-red-400' 
                      : 'bg-amber-500/20 text-amber-400'
                  }`}>
                    {anomaly.volume_zscore > 0 ? '+' : ''}{anomaly.volume_zscore.toFixed(2)}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
