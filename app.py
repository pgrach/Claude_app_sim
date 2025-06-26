import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yaml
from datetime import datetime, timedelta
import openpyxl
from hydro_miner_model import (
    load_hydro_data, load_btc_data, analyze_power_profile,
    run_monte_carlo_simulation, calculate_optimal_fleet,
    project_mining_economics
)

# Page configuration
st.set_page_config(
    page_title="Hydro Bitcoin Mining Optimizer",
    page_icon="⚡",
    layout="wide"
)

# Load configuration
@st.cache_data
def load_config():
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)

# Initialize session state
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None

# Header
st.title("⚡ Hydro-Powered Bitcoin Mining Optimizer")
st.markdown("Optimize ASIC fleet size for run-of-river hydroelectric power with dynamic throttling")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Load default config
    config = load_config()
    
    st.subheader("ASIC Specifications")
    asic_model = st.text_input("Model", value=config['asic']['model'])
    asic_hashrate = st.number_input("Hashrate (TH/s)", value=config['asic']['hash_rate_th'], min_value=1)
    asic_power = st.number_input("Power per TH (W)", value=config['asic']['watts_per_th'], min_value=1.0)
    asic_price = st.number_input("Price per TH ($)", value=config['asic']['price_usd_per_th'], min_value=1.0)
    
    st.subheader("Financial Assumptions")
    pool_fee = st.number_input(
        "Mining Pool Fee (%)",
        value=config['financial']['pool_fee_percent'] * 100,
        min_value=0.0,
        max_value=10.0,
        step=0.1,
        format="%.1f"
    ) / 100.0

    st.subheader("Operating Costs")
    annual_opex = st.number_input("Annual Operating Costs ($)", value=config['operating_costs']['annual_usd'], min_value=0)
    
    st.subheader("Simulation Parameters")
    n_simulations = st.number_input("Monte Carlo Simulations", value=config['simulation']['n_simulations'], min_value=100, max_value=10000)
    fleet_step = st.number_input("Fleet Size Step", value=config['simulation']['fleet_step'], min_value=5)
    projection_years = st.number_input("Projection Years", value=config['simulation']['projection_years'], min_value=1, max_value=5)
    
    st.subheader("Economic Scenarios")
    scenario_key = st.selectbox(
        "Scenario",
        options=list(config['scenarios'].keys()),
        format_func=lambda k: config['scenarios'][k]['name']
    )
    selected_scenario = config['scenarios'][scenario_key]

    with st.expander("View Scenario Details", expanded=True):
        st.markdown(f"*{selected_scenario['description']}*")
        col1, col2 = st.columns(2)
        col1.metric("Annual Difficulty Growth", f"{selected_scenario['difficulty_growth_annual']:.1%}")
        col2.metric("Annual Price Change", f"{selected_scenario['price_change_annual']:.1%}")
    
    run_simulation = st.button("🚀 Run Optimization", type="primary")

# Main content area
col1, col2 = st.columns([3, 2])

with col1:
    st.header("📊 Power Profile Analysis")
    
    # Load and analyze hydro data
    @st.cache_data
    def get_hydro_analysis():
        hydro_data = load_hydro_data('hydro_flow.xlsx')
        return analyze_power_profile(hydro_data)
    
    hydro_stats = get_hydro_analysis()
    
    # Display power statistics
    st.subheader("Power Availability Statistics")
    
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    with metrics_col1:
        st.metric("Average Power", f"{hydro_stats['avg_power_kw']:.0f} kW")
    with metrics_col2:
        st.metric("Uptime", f"{hydro_stats['uptime_percent']:.1f}%")
    with metrics_col3:
        st.metric("P50 Power", f"{hydro_stats['p50_power_kw']:.0f} kW")
    with metrics_col4:
        st.metric("P90 Power", f"{hydro_stats['p90_power_kw']:.0f} kW")
      # Power duration curve
    st.subheader("Power Duration Curve")
    fig_duration = go.Figure()
    percentiles = list(hydro_stats['power_percentiles'].keys())
    power_values = list(hydro_stats['power_percentiles'].values())
    
    fig_duration.add_trace(go.Scatter(
        x=percentiles,
        y=power_values,
        mode='lines',
        name='Available Power',
        fill='tozeroy'
    ))
    
    # Add ASIC power requirements
    asic_power_kw = asic_hashrate * asic_power / 1000
    for n_asics in range(fleet_step, int(hydro_stats['max_power_kw'] / asic_power_kw) + fleet_step, fleet_step):
        fig_duration.add_hline(
            y=n_asics * asic_power_kw,
            line_dash="dash",
            annotation_text=f"{n_asics} ASICs",
            annotation_position="right"
        )
    
    fig_duration.update_layout(
        xaxis_title="Exceedance Probability (%)",
        yaxis_title="Power (kW)",
        height=400,
        margin=dict(r=80)
    )
    st.plotly_chart(fig_duration, use_container_width=True)
    
    # Monthly variation
    st.subheader("Monthly Power Variation")
    monthly_data = pd.DataFrame({
        'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        'Average Power (kW)': hydro_stats['monthly_avg_power']
    })
    
    fig_monthly = px.bar(monthly_data, x='Month', y='Average Power (kW)',
                         color='Average Power (kW)', color_continuous_scale='Blues')
    fig_monthly.update_layout(height=300)
    st.plotly_chart(fig_monthly, use_container_width=True)

with col2:
    st.header("💰 Mining Economics")
    
    # Load BTC data
    @st.cache_data
    def get_btc_data():
        return load_btc_data('btc_price.csv', 'btc_difficulty.csv')
    
    btc_data = get_btc_data()
    
    # Current mining metrics
    st.subheader("Current Mining Metrics")
    
    current_btc_price = btc_data['current_price']
    current_difficulty = btc_data['current_difficulty']
    current_revenue_per_th = btc_data['current_revenue_per_th']
    
    metrics_col1, metrics_col2 = st.columns(2)
    with metrics_col1:
        st.metric("BTC Price", f"${current_btc_price:,.0f}")
        st.metric("Revenue/TH/day", f"${current_revenue_per_th:.4f}")
    with metrics_col2:
        st.metric("Network Difficulty", f"{current_difficulty/1e12:.0f}T")
        st.metric("Revenue/ASIC/day", f"${current_revenue_per_th * asic_hashrate:.2f}")
    
    # Historical trends
    st.subheader("Historical Trends")
    
    # Create subplot with price and difficulty
    fig_trends = go.Figure()
    
    # Add BTC price on primary y-axis
    fig_trends.add_trace(go.Scatter(
        x=btc_data['historical_data']['date'],
        y=btc_data['historical_data']['price'],
        name='BTC Price',
        yaxis='y'
    ))
    
    # Add difficulty on secondary y-axis
    fig_trends.add_trace(go.Scatter(
        x=btc_data['historical_data']['date'],
        y=btc_data['historical_data']['difficulty'],
        name='Network Difficulty',
        yaxis='y2',
        line=dict(color='red')
    ))
    
    fig_trends.update_layout(
        yaxis=dict(title='BTC Price ($)', side='left'),
        yaxis2=dict(title='Network Difficulty', side='right', overlaying='y'),
        height=300,
        hovermode='x unified'
    )
    st.plotly_chart(fig_trends, use_container_width=True)

# Simulation results section
if run_simulation:
    st.header("🎯 Optimization Results")
    
    with st.spinner("Running Monte Carlo simulation..."):
        # Prepare ASIC specs
        asic_specs = {
            'model': asic_model,
            'hash_rate_th': asic_hashrate,
            'watts_per_th': asic_power,
            'price_usd_per_th': asic_price,
            'power_consumption_kw': asic_hashrate * asic_power / 1000,
            'unit_price': asic_hashrate * asic_price
        }
        
        # Run simulation
        simulation_results = run_monte_carlo_simulation(
            hydro_stats=hydro_stats,
            btc_data=btc_data,
            asic_specs=asic_specs,
            annual_opex=annual_opex,
            n_simulations=n_simulations,
            fleet_step=fleet_step,
            scenario_params=selected_scenario,
            projection_years=projection_years,
            pool_fee=pool_fee
        )
        
        st.session_state.simulation_results = simulation_results
    
    # Display results
    results = st.session_state.simulation_results
    
    # Optimal fleet recommendation
    st.subheader("🏆 Optimal Fleet Size")
    
    optimal = calculate_optimal_fleet(results, hydro_stats, asic_specs)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Recommended ASICs", f"{optimal['n_asics']}")
    with col2:
        st.metric("Expected NPV", f"${optimal['expected_npv']:,.0f}")
    with col3:
        st.metric("IRR", f"{optimal['irr']:.1f}%")
    with col4:
        st.metric("Payback Period", f"{optimal['payback_months']:.1f} months")
    
    # NPV by fleet size
    st.subheader("NPV Analysis by Fleet Size")
    
    fig_npv = go.Figure()
    
    # Add percentile bands
    fig_npv.add_trace(go.Scatter(
        x=results['fleet_sizes'],
        y=results['npv_p10'],
        fill=None,
        mode='lines',
        line_color='rgba(0,100,80,0)',
        showlegend=False
    ))
    
    fig_npv.add_trace(go.Scatter(
        x=results['fleet_sizes'],
        y=results['npv_p90'],
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,100,80,0)',
        name='P10-P90 Range'
    ))
    
    fig_npv.add_trace(go.Scatter(
        x=results['fleet_sizes'],
        y=results['npv_expected'],
        mode='lines+markers',
        name='Expected NPV',
        line=dict(color='blue', width=3)
    ))
    
    # Mark optimal point
    fig_npv.add_trace(go.Scatter(
        x=[optimal['n_asics']],
        y=[optimal['expected_npv']],
        mode='markers',
        marker=dict(size=15, color='red', symbol='star'),
        name='Optimal'
    ))
    
    fig_npv.update_layout(
        xaxis_title="Number of ASICs",
        yaxis_title="Net Present Value ($)",
        height=400
    )
    st.plotly_chart(fig_npv, use_container_width=True)
    
    # Risk metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Metrics")

        risk_df = pd.DataFrame({
            'Fleet Size': results['fleet_sizes'],
            'Probability of Loss (%)': results['prob_loss'] * 100,
            'Value at Risk (95%)': results['var_95'],
            'Sharpe Ratio': results['sharpe_ratio']
        })

        # Format for display
        display_df = risk_df.copy()
        display_df['Probability of Loss (%)'] = display_df['Probability of Loss (%)'].map('{:.1f}%'.format)
        display_df['Value at Risk (95%)'] = display_df['Value at Risk (95%)'].map('${:,.0f}'.format)
        display_df['Sharpe Ratio'] = display_df['Sharpe Ratio'].map('{:.2f}'.format)

        st.dataframe(display_df.set_index('Fleet Size'))

        # Add VaR chart
        fig_var = px.line(risk_df, x='Fleet Size', y='Value at Risk (95%)',
                          title='Value at Risk (95%) by Fleet Size')
        fig_var.update_layout(height=250, yaxis_title="VaR ($)")
        st.plotly_chart(fig_var, use_container_width=True)

        # Add Sharpe Ratio chart
        fig_sharpe = px.line(risk_df, x='Fleet Size', y='Sharpe Ratio',
                             title='Sharpe Ratio by Fleet Size', color_discrete_sequence=['green'])
        fig_sharpe.update_layout(height=250, yaxis_title="Ratio")
        st.plotly_chart(fig_sharpe, use_container_width=True)
    
    with col2:
        st.subheader("Utilization Analysis")
        
        util_df = pd.DataFrame({
            'Fleet Size': results['fleet_sizes'],
            'Average Utilization (%)': results['avg_utilization'],
            'Capacity Factor (%)': results['capacity_factor']
        })
        
        fig_util = px.line(util_df, x='Fleet Size', y=['Average Utilization (%)', 'Capacity Factor (%)'],
                          title='Fleet Utilization Metrics', 
                          color_discrete_map={
                              'Average Utilization (%)': 'dodgerblue',
                              'Capacity Factor (%)': 'mediumorchid'
                          })
        fig_util.update_layout(height=300, yaxis_title="Percentage (%)")
        st.plotly_chart(fig_util, use_container_width=True)
    
    # Detailed projections
    st.subheader("📈 Detailed Projections")
    
    projections = project_mining_economics(
        n_asics=optimal['n_asics'],
        hydro_stats=hydro_stats,
        btc_data=btc_data,
        asic_specs=asic_specs,
        scenario_params=selected_scenario,
        years=projection_years,
        annual_opex=annual_opex,
        pool_fee=pool_fee
    )
    
    # Create projection table
    proj_df = pd.DataFrame(projections['yearly_data'])
    proj_df['Year'] = range(1, len(proj_df) + 1)
    
    # Format columns
    currency_cols = ['Revenue', 'Operating Costs', 'Net Income', 'Cumulative Cash Flow']
    for col in currency_cols:
        proj_df[col] = proj_df[col].map('${:,.0f}'.format)
    
    proj_df['BTC Mined'] = proj_df['BTC Mined'].map('{:.4f}'.format)
    proj_df['Avg BTC Price'] = proj_df['Avg BTC Price'].map('${:,.0f}'.format)
    proj_df['Avg Difficulty'] = (proj_df['Avg Difficulty'] / 1e12).map('{:.0f}T'.format)
    
    st.dataframe(proj_df[['Year', 'BTC Mined', 'Revenue', 'Operating Costs', 
                         'Net Income', 'Cumulative Cash Flow', 'Avg BTC Price', 'Avg Difficulty']])
    
    # Summary report
    with st.expander("📋 Executive Summary", expanded=True):
        st.markdown(f"""
        ### Optimization Results for {selected_scenario['name']} Scenario
        
        **Recommended Configuration:**
        - Optimal Fleet Size: **{optimal['n_asics']} ASICs**
        - Total Investment: **${optimal['n_asics'] * asic_specs['unit_price']:,.0f}**
        - Expected NPV: **${optimal['expected_npv']:,.0f}**
        - Internal Rate of Return: **{optimal['irr']:.1f}%**
        - Payback Period: **{optimal['payback_months']:.1f} months**
        
        **Key Insights:**
        - Power constraint allows maximum {int(hydro_stats['avg_power_kw'] / asic_specs['power_consumption_kw'])} ASICs at average power
        - Optimal size balances utilization ({optimal['utilization']:.1f}%) with revenue potential
        - {optimal['risk_assessment']}
        
        **Power Utilization:**
        - Fleet will operate at 100% capacity {optimal['full_power_percent']:.1f}% of the time
        - Average throttling level: {optimal['avg_throttle']:.1f}%
        - Zero production expected {optimal['zero_production_days']:.1f}% of days
        
        **Financial Projections ({projection_years} years):**
        - Total BTC Mined: {projections['total_btc_mined']:.4f} BTC
        - Total Revenue: ${projections['total_revenue']:,.0f}
        - Net Profit: ${projections['total_profit']:,.0f}
        
        **Recommendation:** {optimal['recommendation']}
        """)

# Footer
st.markdown("---")
st.caption("Bitcoin Mining Optimizer | Hydroelectric Power Constraints with Dynamic Throttling")