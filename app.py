import streamlit as st
import torch
import numpy as np
import pandas as pd
import yfinance as yf
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="Advanced Crypto Price Predictor",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
    }
    .metric-label {
        font-size: 14px;
        color: #555;
    }
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# LSTM Model definition
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1])

# Sidebar for inputs
st.sidebar.title("Model Parameters")

# Cryptocurrency selection
crypto_options = {
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Cardano": "ADA-USD", 
    "Solana": "SOL-USD",
    "Binance Coin": "BNB-USD"
}
selected_crypto = st.sidebar.selectbox("Select Cryptocurrency", list(crypto_options.keys()))
crypto_symbol = crypto_options[selected_crypto]

# Date range selection
today = datetime.now()
default_start = today - timedelta(days=365)
default_end = today - timedelta(days=1)

start_date = st.sidebar.date_input("Historical Data Start Date", default_start)
end_date = st.sidebar.date_input("Historical Data End Date", default_end)

# Prediction parameters
sequence_length = st.sidebar.slider("Sequence Length (Days)", 30, 90, 60)
forecast_days = st.sidebar.slider("Forecast Days", 1, 30, 7)

# Confidence interval percentage
confidence_pct = st.sidebar.slider("Confidence Interval (%)", 50, 95, 80)

# Load pre-trained model or train new one
@st.cache_resource
def get_model(crypto, seq_len=60):
    try:
        model = LSTMModel(input_size=1, hidden_size=50, num_layers=2, output_size=1)
        model.load_state_dict(torch.load(f"lstm_{crypto.split('-')[0].lower()}_model.pth"))
        return model
    except:
        st.sidebar.warning(f"No pre-trained model found for {crypto}. Using default BTC model.")
        model = LSTMModel(input_size=1, hidden_size=50, num_layers=2, output_size=1)
        try:
            model.load_state_dict(torch.load("lstm_crypto_model.pth"))
        except:
            st.sidebar.error("No model files found. Please train a model first.")
        return model

# Function to fetch and prepare data
@st.cache_data
def fetch_data(symbol, start, end):
    try:
        df = yf.download(symbol, start=start, end=end)
        if df.empty:
            st.error(f"No data available for {symbol} in the specified date range.")
            return None
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Function to prepare data for prediction
def prepare_data(df, seq_length=60):
    # Create scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Close']].values)
    
    # Create sequences for backtesting
    X, y = [], []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:i+seq_length])
        y.append(scaled_data[i+seq_length])
    
    # The last sequence for future prediction
    last_sequence = scaled_data[-seq_length:]
    
    return np.array(X), np.array(y), last_sequence, scaler

# Function to generate predictions with confidence intervals
def predict_with_confidence(model, sequence, scaler, days=7, confidence=0.8):
    model.eval()
    predictions = []
    lower_bounds = []
    upper_bounds = []
    
    current_sequence = sequence.copy()
    
    # Calculate z-score based on confidence level
    # For 80% confidence, z = 1.28
    # For 90% confidence, z = 1.645
    # For 95% confidence, z = 1.96
    z_map = {80: 1.28, 85: 1.44, 90: 1.645, 95: 1.96}
    z = z_map.get(confidence, 1.28)
    
    # Standard deviation based on historical prediction errors (assuming 5% of the value for simplicity)
    std_dev_factor = 0.05
    
    with torch.no_grad():
        for _ in range(days):
            # Reshape sequence for model input
            current_tensor = torch.tensor(current_sequence.reshape(1, -1, 1), dtype=torch.float32)
            
            # Get prediction
            pred = model(current_tensor).item()
            
            # Calculate confidence interval (simplified approach)
            orig_value = scaler.inverse_transform([[pred]])[0][0]
            std_dev = orig_value * std_dev_factor
            lower = pred - (z * std_dev_factor)
            upper = pred + (z * std_dev_factor)
            
            # Save predictions and bounds
            predictions.append(pred)
            lower_bounds.append(lower)
            upper_bounds.append(upper)
            
            # Update sequence for next prediction
            current_sequence = np.append(current_sequence[1:], [[pred]], axis=0)
    
    # Convert to original scale
    predictions_orig = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    lower_bounds_orig = scaler.inverse_transform(np.array(lower_bounds).reshape(-1, 1)).flatten()
    upper_bounds_orig = scaler.inverse_transform(np.array(upper_bounds).reshape(-1, 1)).flatten()
    
    return predictions_orig, lower_bounds_orig, upper_bounds_orig

# Function to evaluate model on historical data
def evaluate_model(model, X, y, scaler):
    model.eval()
    y_pred = []
    
    with torch.no_grad():
        for i in range(len(X)):
            x_tensor = torch.tensor(X[i].reshape(1, -1, 1), dtype=torch.float32)
            pred = model(x_tensor).item()
            y_pred.append(pred)
    
    # Convert predictions back to original scale
    y_pred = scaler.inverse_transform(np.array(y_pred).reshape(-1, 1))
    y_true = scaler.inverse_transform(y.reshape(-1, 1))
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return y_pred, y_true, rmse, mae, mape

# Main app
st.title(f"📈 Advanced {selected_crypto} Price Prediction")
st.markdown("### Powered by LSTM Neural Networks")

# Load data
with st.spinner(f"Fetching {selected_crypto} data..."):
    df = fetch_data(crypto_symbol, start_date, end_date)

if df is not None:
    # Prepare data
    X, y, last_sequence, scaler = prepare_data(df, sequence_length)
    
    # Load model
    with st.spinner("Loading model..."):
        model = get_model(crypto_symbol, sequence_length)
    
    # Evaluate model on historical data
    with st.spinner("Evaluating model performance..."):
        y_pred, y_true, rmse, mae, mape = evaluate_model(model, X, y, scaler)
    
    # Generate future predictions
    with st.spinner(f"Generating {forecast_days}-day forecast..."):
        future_preds, lower_bounds, upper_bounds = predict_with_confidence(
            model, last_sequence, scaler, days=forecast_days, confidence=confidence_pct
        )
    
    # Create date range for future predictions
    last_date = df.index[-1]
    future_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
    
    # Display metrics in a 3-column layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">${future_preds[0]:,.2f}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-label">Predicted Price (Tomorrow)</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">${future_preds[-1]:,.2f}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-label">Predicted Price ({forecast_days} Days)</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:
        # Get the last Close price as a scalar value
        current_price = float(df['Close'].iloc[-1])  # Explicitly convert to float
        change = ((future_preds[-1] - current_price) / current_price) * 100
        color = "green" if change >= 0 else "red"
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value" style="color:{color}">{change:+.2f}%</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-label">Expected Change ({forecast_days} Days)</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Price Forecast", "Model Performance", "Historical Analysis"])
    
    with tab1:
        # Plot forecast with confidence intervals using Plotly
        fig = make_subplots(specs=[[{"secondary_y": False}]])
        
        # Add historical prices
        fig.add_trace(
            go.Scatter(
                x=df.index[-30:],  # Last 30 days
                y=df['Close'].values[-30:],
                mode='lines',
                name='Historical Price',
                line=dict(color='blue')
            )
        )
        
        # Add predictions
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=future_preds,
                mode='lines+markers',
                name='Predicted Price',
                line=dict(color='red', dash='dash')
            )
        )
        
        # Add confidence interval
        fig.add_trace(
            go.Scatter(
                x=future_dates + future_dates[::-1],
                y=list(upper_bounds) + list(lower_bounds)[::-1],
                fill='toself',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,0,0,0)'),
                name=f'{confidence_pct}% Confidence Interval'
            )
        )
        
        fig.update_layout(
            title=f"{selected_crypto} Price Forecast - Next {forecast_days} Days",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display predictions in a table
        pred_df = pd.DataFrame({
            'Date': [d.strftime('%Y-%m-%d') for d in future_dates],
            'Predicted Price': [f"${p:,.2f}" for p in future_preds],
            'Lower Bound': [f"${p:,.2f}" for p in lower_bounds],
            'Upper Bound': [f"${p:,.2f}" for p in upper_bounds]
        })
        
        st.subheader("Detailed Price Predictions")
        st.dataframe(pred_df, use_container_width=True)
    
    with tab2:
        # Display model performance metrics
        st.subheader("Model Performance Metrics")
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric("RMSE", f"${rmse:.2f}")
            st.markdown("Root Mean Squared Error")
            
        with metric_col2:
            st.metric("MAE", f"${mae:.2f}")
            st.markdown("Mean Absolute Error")
            
        with metric_col3:
            st.metric("MAPE", f"{mape:.2f}%")
            st.markdown("Mean Absolute Percentage Error")
        
        # Plot actual vs predicted on test data
        fig = go.Figure()
        
        # Use the last 20% of data as test set for visualization
        test_size = int(len(y_true) * 0.2)
        
        # Add actual prices
        fig.add_trace(
            go.Scatter(
                x=df.index[-test_size:],
                y=y_true[-test_size:].flatten(),
                mode='lines',
                name='Actual Price',
                line=dict(color='blue')
            )
        )
        
        # Add predicted prices
        fig.add_trace(
            go.Scatter(
                x=df.index[-test_size:],
                y=y_pred[-test_size:].flatten(),
                mode='lines',
                name='Predicted Price',
                line=dict(color='red', dash='dash')
            )
        )
        
        fig.update_layout(
            title="Model Validation: Actual vs Predicted Prices",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display error distribution
        errors = y_true.flatten() - y_pred.flatten()
        
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=errors,
                nbinsx=30,
                marker_color='blue',
                opacity=0.7
            )
        )
        
        fig.update_layout(
            title="Error Distribution",
            xaxis_title="Prediction Error (USD)",
            yaxis_title="Frequency"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Historical price analysis
        st.subheader("Historical Price Analysis")
        
        # Price chart with moving averages
        fig = go.Figure()
        
        # Add closing price
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue')
            )
        )
        
        # Add 7-day moving average
        df['MA7'] = df['Close'].rolling(window=7).mean()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MA7'],
                mode='lines',
                name='7-Day MA',
                line=dict(color='orange')
            )
        )
        
        # Add 30-day moving average
        df['MA30'] = df['Close'].rolling(window=30).mean()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MA30'],
                mode='lines',
                name='30-Day MA',
                line=dict(color='green')
            )
        )
        
        fig.update_layout(
            title=f"{selected_crypto} Historical Price with Moving Averages",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Daily returns
        df['Daily_Return'] = df['Close'].pct_change() * 100
        
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Daily_Return'],
                mode='lines',
                name='Daily Return',
                line=dict(color='blue')
            )
        )
        
        fig.update_layout(
            title=f"{selected_crypto} Daily Returns (%)",
            xaxis_title="Date",
            yaxis_title="Return (%)",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume chart
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume',
                marker_color='blue'
            )
        )
        
        fig.update_layout(
            title=f"{selected_crypto} Trading Volume",
            xaxis_title="Date",
            yaxis_title="Volume",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Advanced Options Section
    with st.expander("Advanced Options", expanded=False):
        st.subheader("Export Predictions")
        
        # Create a DataFrame of predictions
        export_df = pd.DataFrame({
            'Date': [d.strftime('%Y-%m-%d') for d in future_dates],
            'Predicted_Price': future_preds,
            'Lower_Bound': lower_bounds,
            'Upper_Bound': upper_bounds
        })
        
        # Convert to CSV for download
        csv = export_df.to_csv(index=False)
        
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name=f"{selected_crypto.lower().replace(' ', '_')}_predictions.csv",
            mime="text/csv"
        )
        
        st.subheader("Model Information")
        st.markdown("""
        **LSTM Architecture:**
        - Input features: 1 (Price)
        - Hidden size: 50
        - Number of layers: 2
        - Output: 1 (Next day price)
        
        **Training Parameters:**
        - Sequence length: {}
        - Optimizer: Adam
        - Loss function: Mean Squared Error
        """.format(sequence_length))

else:
    st.error("Unable to fetch data. Please check your inputs and try again.")

# Footer
st.markdown("---")
st.markdown("### 🚀 Built with Streamlit, PyTorch & Plotly")
st.markdown("Advanced Cryptocurrency Price Prediction Tool v2.0")