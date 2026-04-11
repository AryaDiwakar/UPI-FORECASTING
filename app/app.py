"""
UPI Intelligence Platform - Streamlit Frontend
A research-grade time series forecasting dashboard.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.models.pipeline import ForecastingPipeline
from backend.evaluation.metrics import MetricsCalculator, ModelComparator

st.set_page_config(
    page_title="UPI Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize Streamlit session state."""
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False


def sidebar_controls():
    """Render sidebar controls."""
    st.sidebar.header("⚙️ Configuration")
    
    test_size = st.sidebar.slider("Test Size (months)", 6, 24, 12)
    forecast_horizon = st.sidebar.slider("Forecast Horizon", 3, 24, 12)
    sequence_length = st.sidebar.slider("LSTM Sequence Length", 3, 12, 6)
    
    st.sidebar.header("🔧 Options")
    exclude_anomalies = st.sidebar.checkbox("Exclude Anomalies from Training", value=False)
    run_deep_learning = st.sidebar.checkbox("Run Deep Learning (LSTM)", value=True)
    
    st.sidebar.header("🎯 Actions")
    run_pipeline = st.sidebar.button("🚀 Run Full Pipeline", type="primary", use_container_width=True)
    
    if st.sidebar.button("🔄 Reset", use_container_width=True):
        st.session_state.pipeline = None
        st.session_state.results = None
        st.session_state.data_loaded = False
        st.rerun()
    
    return {
        'test_size': test_size,
        'forecast_horizon': forecast_horizon,
        'sequence_length': sequence_length,
        'exclude_anomalies': exclude_anomalies,
        'run_deep_learning': run_deep_learning,
        'run_pipeline': run_pipeline
    }


def render_header():
    """Render main header."""
    st.markdown('<p class="main-header">📊 UPI Intelligence Platform</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Research-Grade UPI Transaction Forecasting System</p>', unsafe_allow_html=True)


def render_metrics_cards(results):
    """Render key metrics as cards."""
    if not results:
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Records Analyzed",
            value=results['data_info']['records'],
            delta=f"{results['data_info']['date_range']['start']} to {results['data_info']['date_range']['end']}"
        )
    
    with col2:
        best_model = results.get('best_model', 'N/A')
        st.metric(label="Best Model", value=best_model)
    
    with col3:
        models_trained = results.get('models_trained', 0)
        st.metric(label="Models Trained", value=models_trained)
    
    with col4:
        duration = results.get('duration_seconds', 0)
        st.metric(label="Processing Time", value=f"{duration:.1f}s")


def render_eda_tab(eda_report):
    """Render EDA tab with visualizations."""
    st.header("📈 Exploratory Data Analysis")
    
    if not eda_report:
        st.warning("Run the pipeline first to see EDA results.")
        return
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview", 
        "📉 Trend Analysis", 
        "🔄 Seasonality",
        "📐 Stationarity",
        "📋 Auto Insights"
    ])
    
    with tab1:
        st.subheader("Distribution Analysis")
        dist = eda_report.get('distribution_analysis', {})
        
        if dist:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Basic Statistics**")
                stats = dist.get('basic_stats', {})
                for key, value in stats.items():
                    st.write(f"- {key.capitalize()}: {value}")
            
            with col2:
                st.write("**Shape Statistics**")
                shape = dist.get('shape_stats', {})
                st.write(f"- Skewness: {shape.get('skewness', 'N/A')}")
                st.write(f"- Interpretation: {shape.get('skewness_interpretation', 'N/A')}")
                st.write(f"- Kurtosis: {shape.get('kurtosis', 'N/A')}")
    
    with tab2:
        st.subheader("Trend Analysis")
        trend = eda_report.get('trend_analysis', {})
        
        if trend:
            trend_type = trend.get('trend_type', {})
            st.write(f"**Trend Type:** {trend_type.get('trend_type', 'Unknown')}")
            st.write(f"**Interpretation:** {trend_type.get('interpretation', '')}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Linear R²: {trend_type.get('linear_r2', 'N/A')}")
            with col2:
                st.write(f"Exponential R²: {trend_type.get('exponential_r2', 'N/A')}")
    
    with tab3:
        st.subheader("Seasonality Analysis")
        seasonal = eda_report.get('seasonality_analysis', {})
        
        if seasonal:
            monthly = seasonal.get('monthly_patterns', {})
            st.write(f"**Seasonality Strength:** {monthly.get('strength', 'Unknown')}")
            st.write(f"**Peak Months:** {monthly.get('peak_months', [])}")
            st.write(f"**Low Months:** {monthly.get('low_months', [])}")
            
            decomp = seasonal.get('decomposition', {})
            if decomp:
                st.write(f"**Trend Direction:** {decomp.get('trend_direction', 'Unknown')}")
    
    with tab4:
        st.subheader("Stationarity Tests")
        stationarity = eda_report.get('stationarity_analysis', {})
        
        if stationarity:
            conclusion = stationarity.get('overall_conclusion', {})
            st.write(f"**Is Stationary:** {conclusion.get('is_stationary', 'Unknown')}")
            st.write(f"**Confidence:** {conclusion.get('confidence', 'Unknown')}")
            st.write(f"**Recommendation:** {conclusion.get('recommendation', '')}")
            
            adf = stationarity.get('adf_test', {})
            if adf:
                st.write(f"\n**ADF Test:**")
                st.write(f"- Test Statistic: {adf.get('test_statistic', 'N/A')}")
                st.write(f"- P-Value: {adf.get('p_value', 'N/A')}")
                st.write(f"- Conclusion: {adf.get('conclusion', 'N/A')}")
    
    with tab5:
        st.subheader("Auto-Generated Insights")
        insights = eda_report.get('auto_insights', [])
        
        if insights:
            for i, insight in enumerate(insights, 1):
                st.write(f"{i}. {insight}")
        else:
            st.info("No insights generated yet.")


def render_models_tab(results):
    """Render model comparison tab."""
    st.header("🤖 Model Comparison")
    
    if not results or 'model_comparison' not in results:
        st.warning("Run the pipeline first to see model results.")
        return
    
    comparison = results['model_comparison']
    rankings = comparison.get('rankings', [])
    
    st.subheader("Model Rankings (by RMSE)")
    
    if rankings:
        df_rankings = pd.DataFrame(rankings)
        df_rankings = df_rankings[['rank', 'model', 'rmse', 'mae', 'mape', 'r2']]
        df_rankings.columns = ['Rank', 'Model', 'RMSE', 'MAE', 'MAPE (%)', 'R²']
        
        st.dataframe(df_rankings, use_container_width=True, hide_index=True)
        
        st.subheader("📊 Performance Visualization")
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("RMSE Comparison", "MAPE Comparison"),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        models = [r['model'] for r in rankings]
        rmse_values = [r['rmse'] for r in rankings]
        mape_values = [r['mape'] for r in rankings]
        
        fig.add_trace(
            go.Bar(x=models, y=rmse_values, marker_color='indianred', name='RMSE'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=models, y=mape_values, marker_color='lightgreen', name='MAPE'),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Model Details")
    
    model_results = results.get('model_results', {})
    if model_results:
        for model_name in rankings[:3] if rankings else list(model_results.keys())[:3]:
            model_key = model_name['model'] if isinstance(model_name, dict) else model_name
            result = model_results.get(model_key, {})
            
            # Handle both dict and ModelResult types
            if hasattr(result, 'metrics'):
                metrics = result.metrics if isinstance(result.metrics, dict) else result.metrics.to_dict()
            elif isinstance(result, dict) and 'metrics' in result:
                metrics = result.get('metrics', {})
            else:
                metrics = {}
            
            if metrics:
                with st.expander(f"📌 {model_key}"):
                    metrics = result.get('metrics', {}) if isinstance(result, dict) else {}
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("RMSE", f"{metrics.get('rmse', 'N/A')}")
                    col2.metric("MAE", f"{metrics.get('mae', 'N/A')}")
                    col3.metric("MAPE", f"{metrics.get('mape', 'N/A')}%")
                    col4.metric("R²", f"{metrics.get('r2', 'N/A')}")


def render_forecast_tab(results):
    """Render forecast visualization tab."""
    st.header("📈 Forecast Analysis")
    
    if not results or 'forecast' not in results:
        st.warning("Run the pipeline first to see forecast results.")
        return
    
    forecast = results['forecast']
    model_name = forecast.get('model', 'Unknown')
    predictions = forecast.get('predictions', [])
    dates = forecast.get('forecast_dates', [])
    lower = forecast.get('confidence_lower', [])
    upper = forecast.get('confidence_upper', [])
    
    st.subheader(f"Forecast from {model_name}")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=predictions,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))
    
    if lower and upper:
        fig.add_trace(go.Scatter(
            x=dates + dates[::-1],
            y=upper + lower[::-1],
            fill='toself',
            fillcolor='rgba(0,100,255,0.2)',
            line=dict(color='rgba(0,100,255,0)'),
            name='95% Confidence Interval'
        ))
    
    fig.update_layout(
        title=f"{model_name} - {len(predictions)}-Month Forecast",
        xaxis_title="Date",
        yaxis_title="Transaction Volume (Millions)",
        height=500,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Forecast Table")
    
    forecast_df = pd.DataFrame({
        'Month': dates,
        'Predicted (M)': predictions,
        'Lower Bound': lower if lower else [''] * len(predictions),
        'Upper Bound': upper if upper else [''] * len(predictions)
    })
    
    st.dataframe(forecast_df, use_container_width=True, hide_index=True)


def render_interpretability_tab(results):
    """Render model interpretability tab."""
    st.header("🔍 Model Interpretability")
    
    if not results:
        st.warning("Run the pipeline first to see interpretability analysis.")
        return
    
    pipeline = st.session_state.get('pipeline')
    
    if pipeline and pipeline.models:
        model_options = list(pipeline.models.keys())
        selected_model = st.selectbox("Select Model", model_options)
        
        interpretability = pipeline.get_interpretability_report(selected_model)
        
        if 'error' not in interpretability:
            st.subheader(f"Feature Importance for {selected_model}")
            
            top_features = interpretability.get('top_features', [])
            
            if top_features:
                df_features = pd.DataFrame(top_features)
                df_features.columns = ['Rank', 'Feature', 'Importance', 'Importance %', 'Cumulative %']
                
                fig = go.Figure(go.Bar(
                    x=df_features['Importance %'].head(15),
                    y=df_features['Feature'].head(15),
                    orientation='h',
                    marker_color='steelblue'
                ))
                
                fig.update_layout(
                    title="Top 15 Features by Importance",
                    height=500,
                    yaxis=dict(autorange="reversed")
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Category Importance")
            
            category_imp = interpretability.get('category_importance', {})
            
            if category_imp:
                categories = list(category_imp.keys())
                values = list(category_imp.values())
                
                fig = go.Figure(go.Bar(
                    x=categories,
                    y=values,
                    marker_color='coral'
                ))
                
                fig.update_layout(
                    title="Feature Importance by Category",
                    xaxis_title="Category",
                    yaxis_title="Total Importance"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Model Explanations")
            
            explanations = interpretability.get('explanations', {})
            for key, explanation in explanations.items():
                st.write(f"**{key.replace('_', ' ').title()}:** {explanation}")
        else:
            st.info(interpretability.get('error', 'No interpretability data available.'))
    else:
        st.info("Train models first to see interpretability analysis.")


def render_data_tab(pipeline):
    """Render raw data exploration tab."""
    st.header("📊 Data Explorer")
    
    if pipeline and pipeline.df_featured is not None:
        df = pipeline.df_featured.copy()
        
        st.subheader("Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", len(df))
        col1.metric("Features", len(pipeline.feature_names))
        col1.metric("Date Range", f"{df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}")
        
        st.subheader("Sample Data")
        
        display_cols = ['date', 'volume_millions', 'value_crores']
        if 'is_anomaly' in df.columns:
            display_cols.append('is_anomaly')
        
        st.dataframe(
            df[display_cols].tail(20),
            use_container_width=True,
            hide_index=True
        )
        
        st.subheader("Time Series Plot")
        
        fig = px.line(
            df, 
            x='date', 
            y='volume_millions',
            title='UPI Transaction Volume Over Time',
            labels={'volume_millions': 'Volume (Millions)', 'date': 'Date'}
        )
        
        if 'is_anomaly' in df.columns:
            anomalies = df[df['is_anomaly'] == True]
            fig.add_trace(go.Scatter(
                x=anomalies['date'],
                y=anomalies['volume_millions'],
                mode='markers',
                marker=dict(size=10, color='red'),
                name='Anomalies'
            ))
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Load data first by running the pipeline.")


def main():
    """Main Streamlit application."""
    init_session_state()
    
    render_header()
    
    config = sidebar_controls()
    
    if config['run_pipeline']:
        with st.spinner("Running forecasting pipeline..."):
            try:
                pipeline = ForecastingPipeline(
                    test_size=config['test_size'],
                    forecast_horizon=config['forecast_horizon'],
                    sequence_length=config['sequence_length']
                )
                
                results = pipeline.run_full_pipeline(
                    exclude_anomalies=config['exclude_anomalies'],
                    run_deep_learning=config['run_deep_learning']
                )
                
                st.session_state.pipeline = pipeline
                st.session_state.results = results
                st.session_state.data_loaded = True
                
                st.success("Pipeline completed successfully!")
                
            except Exception as e:
                st.error(f"Pipeline failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    results = st.session_state.get('results')
    pipeline = st.session_state.get('pipeline')
    
    if results:
        render_metrics_cards(results)
        
        if pipeline:
            eda_report = pipeline.eda_report
        else:
            eda_report = {}
        
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 EDA",
            "🤖 Models",
            "📈 Forecast",
            "🔍 Interpretability",
            "📋 Data",
            "📝 Summary"
        ])
        
        with tab1:
            render_eda_tab(eda_report)
        
        with tab2:
            render_models_tab(results)
        
        with tab3:
            render_forecast_tab(results)
        
        with tab4:
            render_interpretability_tab(results)
        
        with tab5:
            render_data_tab(pipeline)
        
        with tab6:
            render_summary_tab(results)
    
    else:
        st.info("👈 Configure settings in the sidebar and click 'Run Full Pipeline' to begin.")
        
        st.markdown("""
        ### How to Use
        
        1. **Configure Settings** - Use the sidebar to adjust test size, forecast horizon, and model options
        2. **Run Pipeline** - Click 'Run Full Pipeline' to execute the forecasting workflow
        3. **Explore Results** - Navigate through tabs to view EDA, model comparisons, and forecasts
        
        ### What This Tool Does
        
        - **Data Loading**: Scrapes UPI transaction data from NPCI
        - **Preprocessing**: Cleans data, handles missing values, detects outliers
        - **EDA**: Performs statistical analysis including stationarity tests
        - **Feature Engineering**: Creates lag, rolling, and seasonal features
        - **Model Training**: Trains ARIMA, XGBoost, LSTM, and other models
        - **Ensemble**: Combines models using inverse-RMSE weighting
        - **Forecasting**: Generates predictions with confidence intervals
        """)


def render_summary_tab(results):
    """Render summary tab with key findings."""
    st.header("📋 Executive Summary")
    
    if not results:
        st.warning("Run the pipeline first.")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Key Findings")
        
        findings = []
        
        if results.get('best_model'):
            findings.append(f"**Best Performing Model:** {results['best_model']}")
        
        comparison = results.get('model_comparison', {})
        rankings = comparison.get('rankings', [])
        
        if rankings:
            best = rankings[0]
            worst = rankings[-1]
            
            if best.get('rmse') and worst.get('rmse'):
                improvement = ((worst['rmse'] - best['rmse']) / worst['rmse']) * 100
                findings.append(f"**Best vs Worst RMSE Improvement:** {improvement:.1f}%")
        
        eda_insights = results.get('eda_insights', [])
        if eda_insights:
            findings.append(f"**Key Trend:** {eda_insights[0] if eda_insights else 'N/A'}")
        
        for finding in findings:
            st.write(finding)
    
    with col2:
        st.subheader("Recommendations")
        
        recommendations = []
        
        if results.get('best_model') == 'XGBoost' or results.get('best_model') == 'LSTM':
            recommendations.append("Consider ensemble for improved robustness")
        
        eda_insights = results.get('eda_insights', [])
        if any('stationary' in str(i).lower() for i in eda_insights):
            recommendations.append("Consider differencing for ARIMA models")
        
        recommendations.append("Monitor for structural breaks in the time series")
        recommendations.append("Retrain models periodically with new data")
        
        for rec in recommendations:
            st.write(f"- {rec}")
    
    st.subheader("Technical Details")
    
    with st.expander("Pipeline Configuration"):
        st.json({
            'test_size': results.get('data_info', {}).get('test_size', 'N/A'),
            'forecast_horizon': results.get('forecast', {}).get('horizon', 'N/A'),
            'models_trained': results.get('models_trained', 0),
            'feature_count': results.get('feature_info', {}).get('feature_count', 0)
        })
    
    with st.expander("Export Results"):
        if st.button("📥 Download Forecast as CSV"):
            forecast = results.get('forecast', {})
            df = pd.DataFrame({
                'month': forecast.get('forecast_dates', []),
                'prediction': forecast.get('predictions', []),
                'lower_bound': forecast.get('confidence_lower', []),
                'upper_bound': forecast.get('confidence_upper', [])
            })
            
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="upi_forecast.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main()
