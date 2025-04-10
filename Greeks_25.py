# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 16:16:11 2025

@author: Hemal
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Set page config
st.set_page_config(page_title="Black-Scholes Option Pricing Model", layout="wide")
st.title("Black-Scholes Option Pricing Model")

# Function to calculate Black-Scholes option price
def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    """
    Calculate the Black-Scholes price for an option
    
    Parameters:
    S: Underlying asset price
    K: Strike price
    T: Time to expiration in years
    r: Risk-free interest rate (decimal)
    sigma: Volatility (decimal)
    option_type: "call" or "put"
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price

# Function to calculate Black-Scholes Greeks
def black_scholes_greeks(S, K, T, r, sigma, option_type="call"):
    """
    Calculate Black-Scholes Greeks
    
    Parameters:
    S: Underlying asset price
    K: Strike price
    T: Time to expiration in years
    r: Risk-free interest rate (decimal)
    sigma: Volatility (decimal)
    option_type: "call" or "put"
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                 - r * K * np.exp(-r * T) * norm.cdf(d2))
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        delta = -norm.cdf(-d1)
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                 + r * K * np.exp(-r * T) * norm.cdf(-d2))
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    
    # Second-order Greeks
    vanna = -norm.pdf(d1) * d2 / sigma
    charm = -norm.pdf(d1) * (2 * r * T - d2 * sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T))
    vomma = vega * (d1 * d2) / sigma
    veta = -vega * ((r * d1) / (sigma * np.sqrt(T)) - ((1 + d1 * d2) / (2 * T)))
    speed = -gamma / S * (d1 / (sigma * np.sqrt(T)) + 1)
    zomma = gamma * ((d1 * d2 - 1) / sigma)
    color = -gamma / (2 * T)
    ultima = -vega * (d1 * d2 * (d1 * d2 - 1)) / (sigma ** 2)
    
    return {
        "Price": price,
        "Delta": delta, 
        "Gamma": gamma, 
        "Vega": vega, 
        "Theta": theta, 
        "Rho": rho, 
        "Vanna": vanna, 
        "Charm": charm, 
        "Vomma": vomma, 
        "Veta": veta, 
        "Speed": speed, 
        "Zomma": zomma, 
        "Color": color, 
        "Ultima": ultima
    }

# Function to find ATM strike price
def find_atm_strike(underlying_price, strikes):
    """Find the strike price closest to the underlying price (ATM)"""
    strikes = np.array(strikes)
    atm_idx = np.abs(strikes - underlying_price).argmin()
    return strikes[atm_idx]

# Sidebar for file upload and parameters
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Default parameters
default_risk_free_rate = 0.06  # 6% annualized
default_implied_volatility = 0.25  # 25% estimated

# Main content
if uploaded_file is not None:
    # Load the data
    df = pd.read_csv(uploaded_file)
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Check if required columns exist
    required_columns = ["Date", "Expiry", "Underlying Value", "Strike Price"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
    else:
        # Convert expiry date to datetime and compute time to expiry in years
        try:
            df["Date"] = pd.to_datetime(df["Date"], format="%d-%b-%Y")
            df["Expiry"] = pd.to_datetime(df["Expiry"], format="%d-%b-%Y")
            df["Time to Expiry"] = (df["Expiry"] - df["Date"]).dt.days / 365
            
            # Get unique underlying values and dates for selection
            unique_underlyings = df["Underlying Value"].unique()
            unique_dates = df["Date"].unique()
            
            # Sidebar for selecting specific underlying and date
            st.sidebar.header("Data Selection")
            selected_underlying = st.sidebar.selectbox("Select Underlying Value", unique_underlyings)
            selected_date = st.sidebar.selectbox("Select Date", sorted(unique_dates))
            
            # Filter data based on selection
            filtered_df = df[(df["Underlying Value"] == selected_underlying) & 
                             (df["Date"] == selected_date)]
            
            if len(filtered_df) == 0:
                st.warning("No data available for the selected combination.")
            else:
                # Get the time to expiry from filtered data
                time_to_expiry = filtered_df["Time to Expiry"].iloc[0]
                strike_prices = filtered_df["Strike Price"].tolist()
                
                # Find ATM strike price
                atm_strike = find_atm_strike(selected_underlying, strike_prices)
                
                # Display the ATM strike price
                st.sidebar.markdown(f"**ATM Strike Price:** {atm_strike:.2f}")
                
                # Sidebar for BSM parameters with sliders
                st.sidebar.header("BSM Model Parameters")
                
                option_type = st.sidebar.radio("Option Type", ["call", "put"])
                
                # The underlying price is now a slider, initialized with the selected value
                underlying_price = st.sidebar.slider(
                    "Underlying Price (S)", 
                    float(selected_underlying * 0.5), 
                    float(selected_underlying * 1.5), 
                    float(selected_underlying),
                    step=0.01
                )
                
                # Strike price is now a slider, initialized with the ATM value
                strike_price = st.sidebar.slider(
                    "Strike Price (K)", 
                    float(atm_strike * 0.5), 
                    float(atm_strike * 1.5), 
                    float(atm_strike),
                    step=0.01
                )
                
                # Time to expiry slider (in days)
                expiry_days = int(time_to_expiry * 365)
                time_to_expiry_days = st.sidebar.slider(
                    "Time to Expiry (days)", 
                    1, 
                    365, 
                    expiry_days
                )
                time_to_expiry = time_to_expiry_days / 365
                
                # Risk-free rate and volatility sliders
                risk_free_rate = st.sidebar.slider(
                    "Risk-Free Rate (%)", 
                    0.0, 
                    20.0, 
                    default_risk_free_rate * 100
                ) / 100
                
                implied_volatility = st.sidebar.slider(
                    "Implied Volatility (%)", 
                    1.0, 
                    100.0, 
                    default_implied_volatility * 100
                ) / 100
                
                # Calculate Greeks with current parameters
                greeks = black_scholes_greeks(
                    underlying_price, 
                    strike_price, 
                    time_to_expiry, 
                    risk_free_rate, 
                    implied_volatility,
                    option_type
                )
                
                # Display the calculated Greeks
                st.header("Option Pricing Results")
                
                # Create a layout with columns
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("BSM Model Values")
                    price_df = pd.DataFrame({
                        "Parameter": ["Option Price", "Delta", "Gamma", "Vega", "Theta", "Rho"],
                        "Value": [
                            f"INR{greeks['Price']:.4f}",
                            f"{greeks['Delta']:.4f}",
                            f"{greeks['Gamma']:.6f}",
                            f"{greeks['Vega']:.4f}",
                            f"{greeks['Theta']:.4f}",
                            f"{greeks['Rho']:.4f}"
                        ]
                    })
                    st.table(price_df)
                    
                    st.subheader("Second-Order Greeks")
                    second_order_df = pd.DataFrame({
                        "Greek": ["Vanna", "Charm", "Vomma", "Veta", "Speed", "Zomma", "Color", "Ultima"],
                        "Value": [
                            f"{greeks['Vanna']:.6f}",
                            f"{greeks['Charm']:.6f}",
                            f"{greeks['Vomma']:.6f}",
                            f"{greeks['Veta']:.6f}",
                            f"{greeks['Speed']:.6f}",
                            f"{greeks['Zomma']:.6f}",
                            f"{greeks['Color']:.6f}",
                            f"{greeks['Ultima']:.6f}"
                        ]
                    })
                    st.table(second_order_df)
                
                with col2:
                    # Generate data for sensitivity analysis
                    st.subheader("Sensitivity Analysis")
                    
                    # Select which sensitivity to plot
                    sensitivity_type = st.selectbox(
                        "Select Sensitivity Analysis",
                        ["Price vs Strike", "Greeks vs Strike", "Price vs Volatility", "Price vs Time to Expiry"]
                    )
                    
                    if sensitivity_type == "Price vs Strike":
                        # Generate range of strikes
                        strike_range = np.linspace(strike_price * 0.7, strike_price * 1.3, 100)
                        call_prices = []
                        put_prices = []
                        
                        for k in strike_range:
                            call_price = black_scholes_price(
                                underlying_price, k, time_to_expiry, risk_free_rate, implied_volatility, "call"
                            )
                            put_price = black_scholes_price(
                                underlying_price, k, time_to_expiry, risk_free_rate, implied_volatility, "put"
                            )
                            call_prices.append(call_price)
                            put_prices.append(put_price)
                        
                        # Create plot
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=strike_range, y=call_prices, mode='lines', name='Call Price'))
                        fig.add_trace(go.Scatter(x=strike_range, y=put_prices, mode='lines', name='Put Price'))
                        fig.add_vline(x=underlying_price, line_dash="dash", line_color="green", 
                                     annotation_text="Underlying Price")
                        
                        fig.update_layout(
                            title="Option Price vs Strike Price",
                            xaxis_title="Strike Price",
                            yaxis_title="Option Price",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    elif sensitivity_type == "Greeks vs Strike":
                        # Generate range of strikes
                        strike_range = np.linspace(strike_price * 0.7, strike_price * 1.3, 100)
                        
                        # Calculate Greeks for each strike
                        greek_to_plot = st.selectbox("Select Greek", ["Delta", "Gamma", "Vega", "Theta"])
                        
                        call_values = []
                        put_values = []
                        
                        for k in strike_range:
                            call_greeks = black_scholes_greeks(
                                underlying_price, k, time_to_expiry, risk_free_rate, implied_volatility, "call"
                            )
                            put_greeks = black_scholes_greeks(
                                underlying_price, k, time_to_expiry, risk_free_rate, implied_volatility, "put"
                            )
                            call_values.append(call_greeks[greek_to_plot])
                            put_values.append(put_greeks[greek_to_plot])
                        
                        # Create plot
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=strike_range, y=call_values, mode='lines', name=f'Call {greek_to_plot}'))
                        fig.add_trace(go.Scatter(x=strike_range, y=put_values, mode='lines', name=f'Put {greek_to_plot}'))
                        fig.add_vline(x=underlying_price, line_dash="dash", line_color="green", 
                                     annotation_text="Underlying Price")
                        
                        fig.update_layout(
                            title=f"{greek_to_plot} vs Strike Price",
                            xaxis_title="Strike Price",
                            yaxis_title=greek_to_plot,
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    elif sensitivity_type == "Price vs Volatility":
                        # Generate range of volatilities
                        vol_range = np.linspace(0.05, 1.0, 100)
                        call_prices = []
                        put_prices = []
                        
                        for vol in vol_range:
                            call_price = black_scholes_price(
                                underlying_price, strike_price, time_to_expiry, risk_free_rate, vol, "call"
                            )
                            put_price = black_scholes_price(
                                underlying_price, strike_price, time_to_expiry, risk_free_rate, vol, "put"
                            )
                            call_prices.append(call_price)
                            put_prices.append(put_price)
                        
                        # Create plot
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=vol_range * 100, y=call_prices, mode='lines', name='Call Price'))
                        fig.add_trace(go.Scatter(x=vol_range * 100, y=put_prices, mode='lines', name='Put Price'))
                        fig.add_vline(x=implied_volatility * 100, line_dash="dash", line_color="green", 
                                     annotation_text="Current Volatility")
                        
                        fig.update_layout(
                            title="Option Price vs Implied Volatility",
                            xaxis_title="Implied Volatility (%)",
                            yaxis_title="Option Price",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    elif sensitivity_type == "Price vs Time to Expiry":
                        # Generate range of times to expiry
                        time_range = np.linspace(1/365, 2, 100)  # 1 day to 2 years
                        call_prices = []
                        put_prices = []
                        
                        for t in time_range:
                            call_price = black_scholes_price(
                                underlying_price, strike_price, t, risk_free_rate, implied_volatility, "call"
                            )
                            put_price = black_scholes_price(
                                underlying_price, strike_price, t, risk_free_rate, implied_volatility, "put"
                            )
                            call_prices.append(call_price)
                            put_prices.append(put_price)
                        
                        # Create plot
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=time_range * 365, y=call_prices, mode='lines', name='Call Price'))
                        fig.add_trace(go.Scatter(x=time_range * 365, y=put_prices, mode='lines', name='Put Price'))
                        fig.add_vline(x=time_to_expiry * 365, line_dash="dash", line_color="green", 
                                     annotation_text="Current Time")
                        
                        fig.update_layout(
                            title="Option Price vs Time to Expiry",
                            xaxis_title="Time to Expiry (days)",
                            yaxis_title="Option Price",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Display data table
                st.header("Filtered Dataset")
                st.dataframe(filtered_df)
                
                # Calculate Greeks for all strikes in the filtered dataset
                greeks_data = []
                for _, row in filtered_df.iterrows():
                    S = row["Underlying Value"]
                    K = row["Strike Price"]
                    T = row["Time to Expiry"]
                    
                    if pd.notna(S) and pd.notna(K) and T > 0:  # Ensure valid values
                        greeks_result = black_scholes_greeks(
                            S, K, T, risk_free_rate, implied_volatility, option_type
                        )
                        greeks_data.append([
                            K, greeks_result["Price"], greeks_result["Delta"], greeks_result["Gamma"], 
                            greeks_result["Vega"], greeks_result["Theta"], greeks_result["Rho"],
                            greeks_result["Vanna"], greeks_result["Charm"], greeks_result["Vomma"], 
                            greeks_result["Veta"], greeks_result["Speed"], greeks_result["Zomma"],
                            greeks_result["Color"], greeks_result["Ultima"]
                        ])
                
                # Create DataFrame for Greeks
                greeks_df = pd.DataFrame(
                    greeks_data, 
                    columns=[
                        "Strike Price", "Price", "Delta", "Gamma", "Vega", "Theta", "Rho", 
                        "Vanna", "Charm", "Vomma", "Veta", "Speed", "Zomma", "Color", "Ultima"
                    ]
                )
                
                # Display Greeks table
                st.header("Calculated Greeks for All Strikes")
                st.dataframe(greeks_df)
                
                # Download button for Greeks data
                csv = greeks_df.to_csv(index=False)
                st.download_button(
                    label="Download Greeks as CSV",
                    data=csv,
                    file_name="greeks.csv",
                    mime="text/csv",
                )
                
        except Exception as e:
            st.error(f"Error processing data: {e}")
else:
    st.info("Please upload a CSV file with options data to begin.")
    
    # Sample data format
    st.header("Expected CSV Format")
sample_data = pd.DataFrame({
    "Date": ["01-Jan-2023", "01-Jan-2023"],
    "Expiry": ["01-Mar-2023", "01-Mar-2023"],
    "Underlying Value": [100.0, 100.0],
    "Strike Price": [95.0, 105.0]
})

st.dataframe(sample_data)

# Generate CSV string directly from the sample data
sample_csv_string = sample_data.to_csv(index=False)

# Download button for sample data
st.download_button(
    label="Download Sample CSV",
    data=sample_csv_string,
    file_name="sample_option_data.csv",
    mime="text/csv",
)
    # Create a buffer
    csv_string = sample_data.to_csv(index=False)

# Download button for sample data
st.download_button(
    label="Download Sample CSV",
    data=csv_string,
    file_name="sample_option_data.csv",
    mime="text/csv",
    )
