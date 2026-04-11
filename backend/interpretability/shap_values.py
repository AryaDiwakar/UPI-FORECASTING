"""
Interpretability Module
Feature importance, SHAP values, and model explanations.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FeatureImportanceResult:
    """Container for feature importance analysis."""
    feature: str
    importance: float
    rank: int
    cumulative_importance: float


class FeatureImportanceAnalyzer:
    """Analyze and rank feature importance."""
    
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
    
    def analyze(self, importance_dict: Dict[str, float]) -> pd.DataFrame:
        """Create ranked DataFrame of feature importance."""
        df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in importance_dict.items()
        ]).sort_values('importance', ascending=False)
        
        df['rank'] = range(1, len(df) + 1)
        df['cumulative_importance'] = df['importance'].cumsum()
        
        return df
    
    def get_top_features(self, importance_dict: Dict[str, float], 
                       n: int = 10) -> List[Dict[str, Any]]:
        """Get top N most important features."""
        sorted_features = sorted(
            importance_dict.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:n]
        
        total = sum(importance_dict.values())
        cumulative = 0
        
        results = []
        for i, (feature, importance) in enumerate(sorted_features):
            cumulative += importance
            results.append({
                'rank': i + 1,
                'feature': feature,
                'importance': importance,
                'importance_pct': (importance / total * 100) if total > 0 else 0,
                'cumulative_pct': (cumulative / total * 100) if total > 0 else 0
            })
        
        return results
    
    def categorize_features(self, importance_dict: Dict[str, float]) -> Dict[str, List[str]]:
        """Categorize features by type."""
        categories = {
            'lag': [],
            'rolling': [],
            'growth': [],
            'temporal': [],
            'decomposition': [],
            'fourier': [],
            'event': [],
            'momentum': [],
            'other': []
        }
        
        for feature in importance_dict.keys():
            if feature.startswith('lag_'):
                categories['lag'].append(feature)
            elif feature.startswith('rolling_'):
                categories['rolling'].append(feature)
            elif 'growth' in feature or 'momentum' in feature:
                categories['growth'].append(feature)
            elif 'temporal' in feature or any(x in feature for x in ['month', 'quarter', 'year', 'sin', 'cos']):
                categories['temporal'].append(feature)
            elif 'decomp' in feature:
                categories['decomposition'].append(feature)
            elif 'fourier' in feature:
                categories['fourier'].append(feature)
            elif 'is_' in feature or 'festive' in feature or 'lockdown' in feature:
                categories['event'].append(feature)
            else:
                categories['other'].append(feature)
        
        return categories
    
    def get_category_importance(self, importance_dict: Dict[str, float]) -> Dict[str, float]:
        """Calculate total importance by category."""
        categories = self.categorize_features(importance_dict)
        
        category_importance = {}
        for category, features in categories.items():
            total = sum(importance_dict.get(f, 0) for f in features)
            if features:
                category_importance[category] = round(total, 4)
        
        return dict(sorted(
            category_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))


class SHAPExplainer:
    """SHAP-based model explanation (for tree-based models)."""
    
    def __init__(self):
        self.explainer = None
        self.shap_values = None
    
    def fit(self, model, X_train: np.ndarray, feature_names: List[str] = None):
        """Fit SHAP explainer."""
        try:
            import shap
            
            if hasattr(model, 'predict_proba'):
                explainer = shap.TreeExplainer(model)
                self.shap_values = explainer.shap_values(X_train)
            else:
                logger.warning("Model doesn't support SHAP")
                return False
            
            self.feature_names = feature_names
            return True
            
        except ImportError:
            logger.warning("SHAP not installed. Using permutation importance instead.")
            return False
        except Exception as e:
            logger.error(f"SHAP fitting failed: {e}")
            return False
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from SHAP values."""
        if self.shap_values is None:
            return {}
        
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        
        if hasattr(self, 'feature_names') and self.feature_names:
            return dict(zip(self.feature_names, mean_abs_shap.tolist()))
        
        return {f'feature_{i}': v for i, v in enumerate(mean_abs_shap)}


class PermutationImportanceAnalyzer:
    """Permutation importance for any model."""
    
    def __init__(self, n_repeats: int = 10, random_state: int = 42):
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.results = None
    
    def analyze(self, model, X: np.ndarray, y: np.ndarray,
                feature_names: List[str] = None) -> Dict[str, float]:
        """Calculate permutation importance."""
        from sklearn.inspection import permutation_importance
        
        result = permutation_importance(
            model, X, y,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.results = result
        
        if feature_names:
            importance_dict = dict(zip(feature_names, result.importances_mean.tolist()))
        else:
            importance_dict = {f'feature_{i}': v 
                              for i, v in enumerate(result.importances_mean)}
        
        return importance_dict


class ModelExplainer:
    """Generate natural language explanations of model behavior."""
    
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
    
    def explain(self, importance_dict: Dict[str, float],
               top_n: int = 5) -> Dict[str, str]:
        """Generate explanations for model predictions."""
        top_features = sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        explanations = {}
        
        lag_features = [f for f, _ in top_features if 'lag' in f]
        if lag_features:
            explanations['lag_importance'] = (
                f"Past values significantly influence predictions. "
                f"Key lags: {', '.join(lag_features[:3])}"
            )
        
        rolling_features = [f for f, _ in top_features if 'rolling' in f]
        if rolling_features:
            explanations['rolling_importance'] = (
                f"Moving averages are important for predictions. "
                f"Key windows: {', '.join(rolling_features[:2])}"
            )
        
        growth_features = [f for f, _ in top_features if 'growth' in f]
        if growth_features:
            explanations['growth_importance'] = (
                f"Growth rates are predictive of future values. "
                f"Features: {', '.join(growth_features[:2])}"
            )
        
        temporal_features = [f for f, _ in top_features if 'month' in f or 'sin' in f or 'cos' in f]
        if temporal_features:
            explanations['temporal_importance'] = (
                f"Seasonality plays a role in predictions. "
                f"Calendar features help capture periodic patterns."
            )
        
        if not explanations:
            explanations['general'] = (
                f"Multiple features contribute to predictions. "
                f"Top feature: {top_features[0][0] if top_features else 'N/A'}"
            )
        
        return explanations
    
    def explain_prediction(self, feature_values: Dict[str, float],
                          importance_dict: Dict[str, float]) -> str:
        """Explain a specific prediction."""
        top_features = sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        explanations = []
        for feature, importance in top_features:
            value = feature_values.get(feature, 'N/A')
            explanations.append(
                f"{feature}={value:.2f} (importance: {importance:.4f})"
            )
        
        return "Key factors: " + "; ".join(explanations)


def generate_interpretability_report(
    importance_dict: Dict[str, float],
    feature_names: List[str],
    top_n: int = 10
) -> Dict[str, Any]:
    """Generate comprehensive interpretability report."""
    
    importance_analyzer = FeatureImportanceAnalyzer(feature_names)
    explainer = ModelExplainer(feature_names)
    
    top_features = importance_analyzer.get_top_features(importance_dict, top_n)
    category_importance = importance_analyzer.get_category_importance(importance_dict)
    explanations = explainer.explain(importance_dict, top_n)
    
    ranked_df = importance_analyzer.analyze(importance_dict)
    
    return {
        'top_features': top_features,
        'category_importance': category_importance,
        'ranked_features': ranked_df.to_dict('records'),
        'explanations': explanations,
        'summary': generate_summary(top_features, category_importance)
    }


def generate_summary(top_features: List[Dict],
                    category_importance: Dict[str, float]) -> str:
    """Generate natural language summary."""
    if not top_features:
        return "No significant features identified."
    
    top_feature = top_features[0]
    total_importance = sum(f['importance'] for f in top_features)
    top_pct = (top_feature['importance'] / total_importance * 100) if total_importance > 0 else 0
    
    summary_parts = []
    
    summary_parts.append(
        f"The most important feature is '{top_feature['feature']}' "
        f"with {top_feature['importance_pct']:.1f}% relative importance."
    )
    
    lag_importance = category_importance.get('lag', 0)
    if lag_importance > 0:
        summary_parts.append(
            f"Lag features contribute {lag_importance:.4f} to the model."
        )
    
    rolling_importance = category_importance.get('rolling', 0)
    if rolling_importance > 0:
        summary_parts.append(
            f"Rolling statistics contribute {rolling_importance:.4f}."
        )
    
    temporal_importance = category_importance.get('temporal', 0)
    if temporal_importance > 0:
        summary_parts.append(
            f"Seasonal/temporal patterns contribute {temporal_importance:.4f}."
        )
    
    return " ".join(summary_parts)


if __name__ == "__main__":
    from backend.data.scraper import fetch_and_store_data
    from backend.preprocessing.cleaner import preprocess_data
    from backend.features.engineering import create_features
    
    df, _ = fetch_and_store_data()
    df_clean, _ = preprocess_data(df)
    df_featured, feature_names = create_features(df_clean)
    
    print("Generating interpretability report...")
    print(f"Feature count: {len(feature_names)}")
    
    mock_importance = {f: np.random.random() for f in feature_names[:20]}
    
    report = generate_interpretability_report(mock_importance, feature_names[:20])
    
    print("\n=== Top Features ===")
    for f in report['top_features'][:5]:
        print(f"{f['rank']}. {f['feature']}: {f['importance_pct']:.2f}%")
    
    print("\n=== Category Importance ===")
    for cat, imp in report['category_importance'].items():
        print(f"{cat}: {imp:.4f}")
    
    print("\n=== Summary ===")
    print(report['summary'])
