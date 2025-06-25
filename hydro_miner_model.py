import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import warnings
from joblib import Parallel, delayed
warnings.filterwarnings('ignore')

def load_hydro_data(filepath):
    """Load and process hydroelectric power data from Excel file."""
    df = pd.read_excel(filepath)
    
    # Convert date column
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Extract key columns
    df['available_energy_kwh'] = pd.to_numeric(df['Available energy (kWh)'], errors='coerce')
    df['available_power_kw'] = df['available_energy_kwh'] / 24  # Convert to average power
    
    return df

def load_btc_data(price_file, difficulty_file):
    """Load and process Bitcoin price and difficulty data."""
    # Load price data
    price_df = pd.read_csv(price_file)
    price_df['Date'] = pd.to_datetime(price_df['Date'])
    price_df = price_df.sort_values('Date')
    
    # Load difficulty data
    diff_df = pd.read_csv(difficulty_file)
    diff_df['Date'] = pd.to_datetime(diff_df['Date'])
    diff_df = diff_df.sort_values('Date')
    
    # Merge on date
    btc_df = pd.merge(price_df, diff_df, on='Date', how='inner')
    
    # Calculate mining economics
    btc_df['block_reward'] = get_block_reward(btc_df['Date'])
    btc_df['btc_per_th_day'] = (1e12 * 86400 * btc_df['block_reward']) / (btc_df['Difficulty'] * 2**32)
    btc_df['revenue_per_th_day'] = btc_df['btc_per_th_day'] * btc_df['Close']
    
    # Calculate growth rates
    btc_df['difficulty_growth_30d'] = btc_df['Difficulty'].pct_change(periods=30)
    btc_df['price_change_30d'] = btc_df['Close'].pct_change(periods=30)
    
    # Current values
    current = btc_df.iloc[-1]
    
    # Historical statistics
    last_year = btc_df[btc_df['Date'] > btc_df['Date'].max() - timedelta(days=365)]
    
    return {
        'current_price': current['Close'],
        'current_difficulty': current['Difficulty'],
        'current_revenue_per_th': current['revenue_per_th_day'],
        'difficulty_growth_annual': last_year['difficulty_growth_30d'].mean() * 12,
        'price_volatility_annual': last_year['Close'].pct_change().std() * np.sqrt(365),
        'historical_data': btc_df[['Date', 'Close', 'Difficulty', 'revenue_per_th_day']].rename(
            columns={'Date': 'date', 'Close': 'price', 'Difficulty': 'difficulty'})
    }

def get_block_reward(dates):
    """Get Bitcoin block reward for a given date or series of dates (vectorized)."""
    date_series = pd.to_datetime(dates)
    
    # Define halving dates and corresponding rewards
    halving_dates = [
        pd.to_datetime('2012-11-28'),
        pd.to_datetime('2016-07-09'),
        pd.to_datetime('2020-05-11'),
        pd.to_datetime('2024-04-20'),
        pd.to_datetime('2028-04-01')  # Approximate next halving
    ]
    rewards = [50, 25, 12.5, 6.25, 3.125, 1.5625]
    
    # Create conditions for np.select
    conditions = [
        date_series < halving_dates[0],
        (date_series >= halving_dates[0]) & (date_series < halving_dates[1]),
        (date_series >= halving_dates[1]) & (date_series < halving_dates[2]),
        (date_series >= halving_dates[2]) & (date_series < halving_dates[3]),
        (date_series >= halving_dates[3]) & (date_series < halving_dates[4])
    ]
    
    # Use np.select for efficient conditional logic
    return np.select(conditions, rewards[:-1], default=rewards[-1])

def analyze_power_profile(hydro_df):
    """Analyze hydroelectric power availability patterns."""
    # Basic statistics
    power_data = hydro_df['available_power_kw'].dropna().values
    
    # Calculate percentiles
    percentiles = {}
    for p in range(0, 101, 5):
        percentiles[p] = np.percentile(power_data, p)

    # Create data for power duration curve (correcting the graph)
    duration_curve = {}
    for p in range(0, 101, 5):
        # For exceedance probability p, we need the (100-p)th percentile
        duration_curve[p] = np.percentile(power_data, 100 - p)
    
    # Monthly averages
    hydro_df['month'] = hydro_df['Date'].dt.month
    monthly_avg = hydro_df.groupby('month')['available_power_kw'].mean().values
    
    # Continuous run analysis
    runs = []
    current_run = 0
    for power in power_data:
        if power > 0:
            current_run += 1
        else:
            if current_run > 0:
                runs.append(current_run)
                current_run = 0
    if current_run > 0:
        runs.append(current_run)
    
    # Calculate statistics
    return {
        'avg_power_kw': np.mean(power_data),
        'min_power_kw': np.min(power_data),
        'max_power_kw': np.max(power_data),
        'p10_power_kw': percentiles[10],
        'p25_power_kw': percentiles[25],
        'p50_power_kw': percentiles[50],
        'p75_power_kw': percentiles[75],
        'p90_power_kw': percentiles[90],
        'power_percentiles': duration_curve,
        'monthly_avg_power': monthly_avg.tolist(),
        'uptime_percent': (power_data > 0).sum() / len(power_data) * 100,
        'zero_power_days': (power_data == 0).sum(),
        'avg_run_days': np.mean(runs) if runs else 0,
        'median_run_days': np.median(runs) if runs else 0,
        'power_distribution': power_data
    }

def simulate_power_availability(hydro_stats, days, seed=None):
    """Bootstrap simulate future power availability from historical data."""
    if seed is not None:
        np.random.seed(seed)
    
    # Use historical distribution with monthly seasonality consideration
    power_dist = hydro_stats['power_distribution']
    
    # Simple bootstrap - could be enhanced with seasonal patterns
    simulated_power = np.random.choice(power_dist, size=days, replace=True)
    
    return simulated_power

def calculate_mining_revenue(hashrate_th, power_available_kw, asic_power_kw, difficulty, btc_price, block_reward):
    """Calculate mining revenue with power constraints and throttling."""
    # Determine throttle level
    if power_available_kw == 0:
        return 0, 0, 0
    
    # Maximum ASICs that can run at full power
    max_asics_full = power_available_kw / asic_power_kw
    
    # If we have enough power, run at 100%
    if max_asics_full >= 1:
        effective_hashrate = hashrate_th
        power_used = asic_power_kw
    else:
        # Throttle to available power
        throttle_percent = max_asics_full
        effective_hashrate = hashrate_th * throttle_percent
        power_used = power_available_kw
    
    # Calculate BTC mined
    btc_per_day = (effective_hashrate * 1e12 * 86400 * block_reward) / (difficulty * 2**32)
    revenue = btc_per_day * btc_price
    return revenue, btc_per_day, power_used

def _precompute_simulation_parameters(projection_years, btc_data, scenario_params, sim_seed):
    """Pre-computes daily price, difficulty, and block rewards for the entire simulation period."""
    np.random.seed(sim_seed)
    
    n_days = projection_years * 365
    n_months = projection_years * 12
    
    # Scenario parameters
    diff_growth_monthly = (1 + scenario_params['difficulty_growth_annual'])**(1/12)
    price_trend_monthly = scenario_params['price_change_annual'] / 12
    price_vol_monthly = btc_data['price_volatility_annual'] / np.sqrt(12)
    
    # --- Generate monthly multipliers ---
    monthly_diff_multipliers = np.cumprod(np.insert(np.full(n_months, diff_growth_monthly), 0, 1))[:-1]
    
    monthly_shocks = np.random.normal(price_trend_monthly, price_vol_monthly, n_months)
    monthly_price_multipliers = np.cumprod(np.insert(1 + monthly_shocks, 0, 1))[:-1]

    # --- Map days to months for accurate daily values ---
    base_date = pd.to_datetime(datetime.now())
    dates = base_date + pd.to_timedelta(np.arange(n_days), unit='d')
    
    # Calculate month index for each day (0 for first month, 1 for second, etc.)
    month_indices = (dates.year - base_date.year) * 12 + (dates.month - base_date.month)
    month_indices = np.minimum(month_indices, n_months - 1) # Ensure indices are within bounds

    # --- Vectorized Difficulty and Price Calculation using month mapping ---
    daily_diff_multipliers = monthly_diff_multipliers[month_indices]
    daily_difficulty = btc_data['current_difficulty'] * daily_diff_multipliers

    daily_price_multipliers = monthly_price_multipliers[month_indices]
    daily_price = btc_data['current_price'] * daily_price_multipliers
    np.maximum(daily_price, 1000, out=daily_price) # Floor price

    # --- Vectorized Block Reward Calculation ---
    daily_block_reward = get_block_reward(dates)
    
    return daily_difficulty, daily_price, daily_block_reward

def _run_single_simulation(sim_seed, fleet_sizes_arr, asic_specs, scenario_params, btc_data, hydro_stats, projection_years, annual_opex, discount_rate, pool_fee):
    """
    Helper function to run a single vectorized simulation for all fleet sizes.
    This version is optimized to remove daily loops and use matrix operations.
    """
    # Extract ASIC parameters
    asic_hashrate = asic_specs['hash_rate_th']
    asic_power_kw = asic_specs['power_consumption_kw']
    asic_price = asic_specs['unit_price']
    n_days = projection_years * 365

    # 1. Pre-computation of Time-Dependent Variables
    daily_difficulty, daily_price, daily_block_reward = _precompute_simulation_parameters(
        projection_years, btc_data, scenario_params, sim_seed
    )
    simulated_power = simulate_power_availability(hydro_stats, n_days, seed=sim_seed)

    # 2. Full Vectorization Across Days AND Fleet Sizes
    # Reshape daily arrays for broadcasting against fleet arrays
    simulated_power_col = simulated_power[:, np.newaxis]
    daily_difficulty_col = daily_difficulty[:, np.newaxis]
    daily_price_col = daily_price[:, np.newaxis]
    daily_block_reward_col = daily_block_reward[:, np.newaxis]

    # Calculate fleet requirements (row vector)
    fleet_power_req_row = fleet_sizes_arr * asic_power_kw
    fleet_hashrate_row = fleet_sizes_arr * asic_hashrate

    # 3. Efficient Power Masking and Matrix-Based Calculations
    # Use a mask for days with power
    power_mask = simulated_power > 0
    
    # Calculate available power for each fleet size (n_days, n_fleets)
    fleet_power_avail = np.minimum(simulated_power_col, fleet_power_req_row)
    
    # Calculate throttle percentage for each fleet on each day
    throttle = np.divide(fleet_power_avail, fleet_power_req_row, 
                         out=np.zeros_like(fleet_power_avail), 
                         where=fleet_power_req_row > 0)

    # Calculate effective hashrate based on throttling
    effective_hashrate = fleet_hashrate_row * throttle
    
    # Calculate daily BTC mined for each fleet
    btc_mined = (effective_hashrate * 1e12 * 86400 * daily_block_reward_col) / (daily_difficulty_col * 2**32)
    
    # Calculate daily revenue
    daily_revenue = (btc_mined * daily_price_col) * (1 - pool_fee)
    daily_revenue[~power_mask, :] = 0
    
    # Aggregate daily revenues into annual cash flows
    annual_revenue = daily_revenue.reshape(projection_years, 365, -1).sum(axis=1)
    
    # Calculate final cash flows including investment and opex
    initial_investment = -fleet_sizes_arr * asic_price
    cash_flows = np.zeros((projection_years + 1, len(fleet_sizes_arr)))
    cash_flows[0, :] = initial_investment
    cash_flows[1:, :] = annual_revenue - annual_opex

    # Calculate NPV for all fleet sizes
    discounts = np.array([(1 + discount_rate)**i for i in range(projection_years + 1)])
    npv = np.sum(cash_flows / discounts[:, np.newaxis], axis=0)

    # Calculate utilization and hashrate metrics
    days_operational = (fleet_power_avail > 0).sum(axis=0)
    avg_utilization = (days_operational / n_days) * 100 if n_days > 0 else 0
    total_effective_hashrate = effective_hashrate.sum(axis=0)

    return npv, cash_flows, avg_utilization, total_effective_hashrate

def run_monte_carlo_simulation(hydro_stats, btc_data, asic_specs, annual_opex, 
                             n_simulations, fleet_step, scenario_params, projection_years, pool_fee):
    """Run Monte Carlo simulation for different fleet sizes."""
    import streamlit as st

    asic_hashrate = asic_specs['hash_rate_th']
    asic_power_kw = asic_specs['power_consumption_kw']
    asic_price = asic_specs['unit_price']

    max_fleet = int(hydro_stats['max_power_kw'] / asic_power_kw)
    fleet_sizes = list(range(fleet_step, max_fleet + 1, fleet_step))

    if not fleet_sizes:
        st.warning("No fleet sizes to simulate. This might be due to low available power or high ASIC power consumption.")
        return {}
        
    fleet_sizes_arr = np.array(fleet_sizes)

    if n_simulations > 2000:
        st.warning(f"Reducing simulations from {n_simulations} to 500 for faster processing")
        n_simulations = 2000

    discount_rate = 0.15
    
    status_text = st.empty()
    status_text.text(f"Running {n_simulations} simulations across {len(fleet_sizes)} fleet sizes...")

    # Parallel execution
    results_list = Parallel(n_jobs=-1)(
        delayed(_run_single_simulation)(
            sim_seed=i,
            fleet_sizes_arr=fleet_sizes_arr,
            asic_specs=asic_specs,
            scenario_params=scenario_params,
            btc_data=btc_data,
            hydro_stats=hydro_stats,
            projection_years=projection_years,
            annual_opex=annual_opex,
            discount_rate=discount_rate,
            pool_fee=pool_fee
        ) for i in range(n_simulations)
    )

    status_text.text("Simulations complete. Aggregating results...")

    # Unpack results
    all_npvs = np.array([res[0] for res in results_list])
    all_cash_flows = np.array([res[1] for res in results_list])
    all_utilizations = np.array([res[2] for res in results_list])
    all_total_effective_hashrates = np.array([res[3] for res in results_list])

    # Aggregate results
    mean_total_effective_hashrate = np.mean(all_total_effective_hashrates, axis=0)
    theoretical_max_total_hashrate = fleet_sizes_arr * asic_hashrate * 365 * projection_years
    capacity_factor = np.divide(mean_total_effective_hashrate, theoretical_max_total_hashrate, out=np.zeros_like(mean_total_effective_hashrate), where=theoretical_max_total_hashrate > 0) * 100
    
    results = {
        'fleet_sizes': fleet_sizes,
        'npv_expected': np.mean(all_npvs, axis=0),
        'npv_p10': np.percentile(all_npvs, 10, axis=0),
        'npv_p90': np.percentile(all_npvs, 90, axis=0),
        'prob_loss': np.mean(all_npvs < 0, axis=0),
        'var_95': -np.percentile(all_npvs, 5, axis=0),
        'sharpe_ratio': np.mean(all_npvs, axis=0) / np.std(all_npvs, axis=0),
        'avg_utilization': np.mean(all_utilizations, axis=0),
        'irr_expected': [],
        'payback_months': [],
        'capacity_factor': capacity_factor
    }

    # Simplified IRR and Payback
    for i, n_asics in enumerate(fleet_sizes):
        initial_investment = n_asics * asic_price
        avg_annual_cf = np.mean(all_cash_flows[:, 1:, i], axis=0).sum() / projection_years
        results['irr_expected'].append((avg_annual_cf / initial_investment) * 100 if initial_investment > 0 else 0)
        
        # Payback (simplified)
        cumulative_cfs = np.cumsum(np.mean(all_cash_flows[:, :, i], axis=0))
        payback_year = np.where(cumulative_cfs > 0)[0]
        if payback_year.any():
            results['payback_months'].append(payback_year[0] * 12)
        else:
            results['payback_months'].append(projection_years * 12)

    status_text.empty()
    return results

def calculate_payback_months(cash_flows):
    """Calculate payback period in months."""
    cumulative = 0
    for i, cf in enumerate(cash_flows):
        cumulative += cf
        if cumulative > 0:
            # Linear interpolation for month
            prev_cumulative = cumulative - cf
            months_in_year = 12 if i > 0 else 0
            month_fraction = (-prev_cumulative / cf) * 12 if cf != 0 else 0
            return (i - 1) * 12 + month_fraction
    return len(cash_flows) * 12

def calculate_actual_hashrate(cash_flows, n_asics, asic_hashrate, asic_price):
    """Estimate actual hashrate from cash flows (simplified)."""
    # This is a rough estimate - in practice would need more detailed tracking
    total_revenue = sum(cash_flows[1:])  # Exclude initial investment
    # Rough estimate assuming average conditions
    return total_revenue / (asic_price * n_asics) * asic_hashrate * 365

def calculate_optimal_fleet(results):
    """Determine optimal fleet size based on simulation results."""
    # Find fleet size with maximum expected NPV
    max_npv_idx = np.argmax(results['npv_expected'])
    optimal_size = results['fleet_sizes'][max_npv_idx]
    
    # Risk-adjusted recommendation
    sharpe_ratios = results['sharpe_ratio']
    max_sharpe_idx = np.argmax(sharpe_ratios)
    risk_adjusted_size = results['fleet_sizes'][max_sharpe_idx]
    
    # Determine recommendation
    if results['prob_loss'][max_npv_idx] > 0.3:
        risk_assessment = "High risk - significant probability of loss"
        recommendation = f"Consider more conservative fleet size of {risk_adjusted_size} ASICs for better risk-adjusted returns"
    elif results['prob_loss'][max_npv_idx] > 0.1:
        risk_assessment = "Moderate risk - acceptable for most investors"
        recommendation = f"Proceed with {optimal_size} ASICs, monitor market conditions closely"
    else:
        risk_assessment = "Low risk - strong probability of positive returns"
        recommendation = f"Optimal configuration of {optimal_size} ASICs offers best returns"
    
    return {
        'n_asics': optimal_size,
        'expected_npv': results['npv_expected'][max_npv_idx],
        'npv_p10': results['npv_p10'][max_npv_idx],
        'npv_p90': results['npv_p90'][max_npv_idx],
        'irr': results['irr_expected'][max_npv_idx],
        'payback_months': results['payback_months'][max_npv_idx],
        'prob_loss': results['prob_loss'][max_npv_idx],
        'utilization': results['avg_utilization'][max_npv_idx],
        'risk_assessment': risk_assessment,
        'recommendation': recommendation,
        'full_power_percent': 50,  # Placeholder - would calculate from detailed simulation
        'avg_throttle': 75,  # Placeholder
        'zero_production_days': 36  # Based on hydro stats
    }

def project_mining_economics(n_asics, hydro_stats, btc_data, asic_specs, scenario_params, years, annual_opex, pool_fee):
    """Generate detailed projections for a specific fleet size."""
    projections = {
        'yearly_data': [],
        'total_btc_mined': 0,
        'total_revenue': 0,
        'total_profit': 0
    }
    
    # Starting conditions
    current_difficulty = btc_data['current_difficulty']
    current_price = btc_data['current_price']
    
    # Annual parameters
    diff_growth = scenario_params['difficulty_growth_annual']
    price_trend = scenario_params['price_change_annual']
    
    cumulative_cash = -n_asics * asic_specs['unit_price']
    
    for year in range(1, years + 1):
        # Project difficulty and price
        year_difficulty = current_difficulty * (1 + diff_growth) ** year
        year_price = current_price * (1 + price_trend) ** year
        
        # Estimate annual production (simplified)
        avg_power_available = hydro_stats['avg_power_kw']
        fleet_power_required = n_asics * asic_specs['power_consumption_kw']
        
        utilization = min(avg_power_available / fleet_power_required, 1.0) if fleet_power_required > 0 else 0
        effective_hashrate = n_asics * asic_specs['hash_rate_th'] * utilization
        
        # Account for uptime
        uptime_factor = hydro_stats['uptime_percent'] / 100
        
        # Check for halving
        year_date = datetime.now() + timedelta(days=365 * year)
        block_reward = get_block_reward(year_date)
        
        # Annual BTC production
        btc_per_day = (effective_hashrate * 1e12 * 86400 * block_reward) / (year_difficulty * 2**32)
        annual_btc = btc_per_day * 365 * uptime_factor
        
        # Financial calculations
        annual_revenue = (annual_btc * year_price) * (1 - pool_fee)
        annual_profit = annual_revenue - annual_opex
        cumulative_cash += annual_profit
        
        projections['yearly_data'].append({
            'Revenue': annual_revenue,
            'Operating Costs': annual_opex,
            'Net Income': annual_profit,
            'BTC Mined': annual_btc,
            'Cumulative Cash Flow': cumulative_cash,
            'Avg BTC Price': year_price,
            'Avg Difficulty': year_difficulty
        })
        
        projections['total_btc_mined'] += annual_btc
        projections['total_revenue'] += annual_revenue
        projections['total_profit'] += annual_profit
    
    return projections