import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime


class InsightGenerator:
    def __init__(self, df: pd.DataFrame, stats: Dict, model_results: Dict):
        self.df = df
        self.stats = stats
        self.model_results = model_results
    
    def generate_insights(self) -> Dict:
        insights = {
            'summary': self._generate_summary(),
            'trends': self._analyze_trends(),
            'seasonality': self._analyze_seasonality(),
            'model_comparison': self._compare_models(),
            'recommendations': self._generate_recommendations(),
            'ai_insights': self._generate_ai_insights(),
            'narrative': self._generate_narrative()
        }
        
        return insights
    
    def _generate_summary(self) -> str:
        latest = self.df['volume_millions'].iloc[-1]
        oldest = self.df['volume_millions'].iloc[0]
        total_growth = ((latest - oldest) / oldest) * 100
        
        cagr = ((latest / oldest) ** (12 / len(self.df)) - 1) * 100 if len(self.df) > 0 else 0
        
        summary = f"""
UPI has revolutionized India's digital payments landscape, growing from {oldest:.2f}M to {latest:.2f}M monthly transactions - a {total_growth:,.0f}% transformation over {self.stats['total_records']} months.

Current trajectory: ~{cagr:.0f}% CAGR with Month-over-Month growth of {self.stats['growth_rate']['volume_mom']:.1f}%
        """.strip()
        
        return summary
    
    def _analyze_trends(self) -> List[str]:
        trends = []
        
        df = self.df.copy()
        df['yoy_growth'] = df['volume_millions'].pct_change(12) * 100
        avg_yoy = df['yoy_growth'].dropna().mean()
        
        if avg_yoy > 100:
            trends.append(f"Explosive adoption: UPI transactions growing at {avg_yoy:.0f}% YoY on average")
        elif avg_yoy > 50:
            trends.append(f"Hypergrowth phase: Sustained {avg_yoy:.0f}% year-over-year expansion")
        elif avg_yoy > 20:
            trends.append(f"Strong momentum: Consistent {avg_yoy:.0f}% YoY growth trajectory")
        else:
            trends.append(f"Maturing market: {avg_yoy:.0f}% YoY growth indicates steady expansion")
        
        recent_vol = self.df['volume_millions'].iloc[-6:].mean()
        older_vol = self.df['volume_millions'].iloc[-12:-6].mean()
        accel = ((recent_vol - older_vol) / older_vol) * 100 if older_vol > 0 else 0
        
        if accel > 25:
            trends.append(f"Acceleration detected: Recent 6-months show {accel:.0f}% higher growth than prior period")
        elif accel > 10:
            trends.append(f"Growth is picking up: {accel:.0f}% improvement in recent growth rate")
        elif accel < -5:
            trends.append(f"Slowing growth: Recent period shows {abs(accel):.0f}% deceleration")
        
        recent_vol_latest = self.df['volume_millions'].iloc[-1]
        if recent_vol_latest > self.stats['volume']['max'] * 0.95:
            trends.append("Near all-time high: Current volume approaching historical peak")
        
        return trends
    
    def _analyze_seasonality(self) -> List[str]:
        df = self.df.copy()
        df['month'] = df['date'].dt.month
        
        monthly_avg = df.groupby('month')['volume_millions'].mean()
        overall_avg = df['volume_millions'].mean()
        
        peak_month = monthly_avg.idxmax()
        low_month = monthly_avg.idxmin()
        
        month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April',
                      5: 'May', 6: 'June', 7: 'July', 8: 'August',
                      9: 'September', 10: 'October', 11: 'November', 12: 'December'}
        
        seasonality = []
        
        peak_factor = monthly_avg[peak_month] / overall_avg if overall_avg > 0 else 1
        if peak_factor > 1.15:
            seasonality.append(f"Festive surge: {month_names[peak_month]} shows {((peak_factor-1)*100):.0f}% above average - driven by Diwali & year-end shopping")
        else:
            seasonality.append(f"Peak activity in {month_names[peak_month]}")
        
        low_factor = monthly_avg[low_month] / overall_avg if overall_avg > 0 else 1
        if low_factor < 0.85:
            seasonality.append(f"Low season: {month_names[low_month]} sees {((1-low_factor)*100):.0f}% below average")
        
        festive_months = [10, 11, 12]
        festive_avg = df[df['month'].isin(festive_months)]['volume_millions'].mean()
        
        if festive_avg > overall_avg * 1.1:
            seasonality.append(f"Q4 effect: Oct-Dec period shows {((festive_avg/overall_avg)-1)*100:.0f}% festive boost vs annual average")
        
        q1_avg = df[df['month'].isin([1, 2, 3])]['volume_millions'].mean()
        if q1_avg < overall_avg * 0.95:
            seasonality.append(f"Post-festive dip: Q1 shows typical normalization after year-end surge")
        
        return seasonality
    
    def _compare_models(self) -> Dict:
        if not self.model_results or 'rankings' not in self.model_results:
            return {}
        
        rankings = self.model_results['rankings']
        
        if not rankings:
            return {}
        
        best = rankings[0]
        worst = rankings[-1] if len(rankings) > 1 else best
        
        comparison = {
            'best_model': best['model'],
            'best_rmse': best['rmse'],
            'best_mae': best['mae'],
            'best_mape': best['mape'],
            'rankings': rankings,
            'insights': []
        }
        
        model_descriptions = {
            'lstm': ('Deep Learning (LSTM)', 'captures complex non-linear patterns and long-term dependencies'),
            'prophet': ('Meta Prophet', 'decomposes trend, seasonality, and holiday effects'),
            'arima': ('Statistical ARIMA', 'models autoregressive patterns with differencing'),
            'linear_regression': ('Ridge Regression', 'provides stable baseline with regularization'),
            'moving_average': ('Simple MA', 'offers naive but robust short-term smoothing')
        }
        
        model_name, description = model_descriptions.get(best['model'], (best['model'], ''))
        comparison['insights'].append(f"{model_name} leads: {description}")
        
        if best['model'] == 'lstm':
            comparison['insights'].append("LSTM excels at capturing the exponential growth curve of UPI adoption")
        elif best['model'] == 'prophet':
            comparison['insights'].append("Prophet effectively models the strong yearly seasonality in UPI usage")
        elif best['model'] == 'arima':
            comparison['insights'].append("ARIMA captures the autoregressive momentum in transaction volumes")
        
        if len(rankings) > 1:
            rmse_diff_pct = ((worst['rmse'] - best['rmse']) / best['rmse']) * 100
            if rmse_diff_pct > 50:
                comparison['insights'].append(f"Significant model gap: {rmse_diff_pct:.0f}% RMSE difference between best and baseline")
            else:
                comparison['insights'].append(f"Models are consistent: {rmse_diff_pct:.0f}% spread indicates stable predictions")
        
        if 'cross_validation' in self.model_results:
            cv = self.model_results['cross_validation']
            if cv and best['model'] in cv:
                cv_std = cv[best['model']].get('std_rmse', 0)
                if cv_std < 2:
                    comparison['insights'].append(f"Robust predictions: {best['model']} shows low variance across validation folds (std: {cv_std:.2f})")
        
        return comparison
    
    def _generate_ai_insights(self) -> List[Dict]:
        ai_insights = []
        
        df = self.df.copy()
        df['yoy_growth'] = df['volume_millions'].pct_change(12) * 100
        
        latest_yoy = df['yoy_growth'].iloc[-1]
        avg_yoy = df['yoy_growth'].dropna().mean()
        
        if latest_yoy > avg_yoy * 1.2:
            ai_insights.append({
                'type': 'acceleration',
                'icon': 'rocket',
                'title': 'Growth Acceleration Detected',
                'description': f'UPI is growing {((latest_yoy/avg_yoy)-1)*100:.0f}% faster than historical average',
                'confidence': 'high',
                'action': 'Prepare for higher transaction volumes than trend projections'
            })
        elif latest_yoy < avg_yoy * 0.8:
            ai_insights.append({
                'type': 'deceleration',
                'icon': 'chart-decreasing',
                'title': 'Growth Normalization',
                'description': f'YoY growth at {latest_yoy:.0f}% is below the {avg_yoy:.0f}% historical average',
                'confidence': 'medium',
                'action': 'Consider revised growth projections for next quarter'
            })
        
        predictions = []
        if self.model_results and 'all_results' in self.model_results:
            for name, result in self.model_results['all_results'].items():
                if 'error' not in result and 'predictions' in result:
                    predictions.append((name, result['predictions']))
        
        if len(predictions) >= 2:
            preds_first_month = [p[1][0] for p in predictions]
            model_std = np.std(preds_first_month)
            latest_val = df['volume_millions'].iloc[-1]
            
            if model_std > latest_val * 0.15:
                ai_insights.append({
                    'type': 'divergence',
                    'icon': 'git-branch',
                    'title': 'Model Divergence Detected',
                    'description': f'Models disagree on near-term outlook (std: {model_std:.1f}M). Ensemble recommended.',
                    'confidence': 'medium',
                    'action': 'Use ensemble forecast for more reliable predictions'
                })
            else:
                ai_insights.append({
                    'type': 'consensus',
                    'icon': 'check-circle',
                    'title': 'High Model Consensus',
                    'description': f'All models align on ~{np.mean(preds_first_month):.0f}M for next month',
                    'confidence': 'high',
                    'action': 'High confidence in near-term forecast accuracy'
                })
        
        festive_months = [10, 11, 12]
        df['month'] = df['date'].dt.month
        next_month = (df['date'].iloc[-1].month % 12) + 1
        
        if next_month in festive_months:
            avg_festive = df[df['month'].isin(festive_months)]['volume_millions'].mean()
            expected_growth = ((avg_festive / df['volume_millions'].iloc[-1]) - 1) * 100
            ai_insights.append({
                'type': 'seasonal',
                'icon': 'gift',
                'title': 'Festive Season Incoming',
                'description': f'Expected {expected_growth:.0f}% volume boost in upcoming festive period',
                'confidence': 'high',
                'action': 'Scale infrastructure for {expected_growth:.0f}% surge in transactions'
            })
        
        df['mom_volatility'] = df['volume_millions'].pct_change().rolling(6).std()
        recent_volatility = df['mom_volatility'].iloc[-1]
        
        if recent_volatility > 0.15:
            ai_insights.append({
                'type': 'volatility',
                'icon': 'activity',
                'title': 'Increased Volatility',
                'description': f'Monthly volatility at {recent_volatility*100:.1f}% - higher than usual',
                'confidence': 'medium',
                'action': 'Widen confidence intervals for forecast uncertainty'
            })
        
        return ai_insights
    
    def _generate_narrative(self) -> Dict:
        df = self.df.copy()
        
        latest_volume = df['volume_millions'].iloc[-1]
        first_volume = df['volume_millions'].iloc[0]
        
        growth_factor = latest_volume / first_volume if first_volume > 0 else 1
        
        stages = []
        
        covid_months = df[(df['date'] >= '2020-03') & (df['date'] <= '2021-03')]
        if not covid_months.empty:
            covid_impact = (covid_months['volume_millions'].iloc[-1] / covid_months['volume_millions'].iloc[0] - 1) * 100
            stages.append({
                'period': '2020-2021',
                'event': 'Pandemic Acceleration',
                'description': f'COVID-19 drove {abs(covid_impact):.0f}% growth as digital payments became essential'
            })
        
        recent_12m_growth = ((df['volume_millions'].iloc[-1] / df['volume_millions'].iloc[-12]) - 1) * 100 if len(df) >= 12 else 0
        stages.append({
            'period': '2024-2025',
            'event': 'Current Phase',
            'description': f'{recent_12m_growth:.0f}% YoY growth - UPI becoming mainstream payment mode'
        })
        
        best_model = self.model_results.get('best_model', 'Unknown') if self.model_results else 'Unknown'
        next_pred = None
        if self.model_results and 'all_results' in self.model_results:
            best_result = self.model_results['all_results'].get(best_model, {})
            if best_result and 'predictions' in best_result:
                next_pred = best_result['predictions'][0]
        
        story = f"""
India's UPI story is one of the world's fastest digital payment adoptions. From humble beginnings 
in 2016 with just {first_volume:.2f}M monthly transactions, UPI has grown {growth_factor:.0f}x to reach {latest_volume:.0f}M 
transactions per month.

The journey has been marked by:
• Explosive post-COVID adoption (2020-2021)
• Festive season surges (Oct-Dec peaks)
• Infrastructure scaling challenges
• Expanding merchant acceptance

Looking ahead, our models predict UPI reaching {next_pred:.0f}M next month, with the platform 
poised to handle even larger transaction volumes as rural adoption increases.
        """.strip()
        
        return {
            'story': story,
            'growth_stages': stages,
            'key_milestone': f'{latest_volume:.0f}M monthly transactions',
            'next_prediction': f'{next_pred:.0f}M' if next_pred else 'TBD',
            'confidence_level': self._get_confidence_narrative()
        }
    
    def _get_confidence_narrative(self) -> str:
        if not self.model_results or 'rankings' not in self.model_results:
            return 'medium'
        
        best_rmse = self.model_results['rankings'][0]['rmse'] if self.model_results['rankings'] else 10
        
        if best_rmse < 5:
            return 'high'
        elif best_rmse < 10:
            return 'medium'
        else:
            return 'low'
    
    def _generate_recommendations(self) -> List[str]:
        recommendations = []
        
        if self.stats['growth_rate']['volume_yoy'] > 50:
            recommendations.append("Infrastructure scaling: Prepare for 50%+ YoY growth with auto-scaling payment infrastructure")
        
        latest_vol = self.df['volume_millions'].iloc[-1]
        if latest_vol > self.stats['volume']['mean'] * 2:
            recommendations.append("Volume alert: Current transaction load is 2x historical average - review capacity planning")
        
        recommendations.append("Model selection: Use ensemble (LSTM + Prophet + ARIMA) for most robust forecasts")
        recommendations.append("Seasonal preparation: Q4 infrastructure stress-testing for festive transaction peaks")
        recommendations.append("Anomaly monitoring: Set up real-time alerts for >15% deviation from predicted volumes")
        recommendations.append("Data quality: Continue feeding NPCI official data for model accuracy")
        
        return recommendations


def generate_insights(df: pd.DataFrame, stats: Dict, model_results: Dict) -> Dict:
    generator = InsightGenerator(df, stats, model_results)
    return generator.generate_insights()
