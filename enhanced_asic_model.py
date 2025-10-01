"""
Enhanced ASIC Model with Overclocking Support
This module extends the original model to support dynamic overclocking
based on available hydro power.
"""

import numpy as np
import pandas as pd

class EnhancedASICModel:
    """ASIC model with overclocking capabilities"""
    
    def __init__(self, base_hashrate, base_power_per_th, overclock_hashrate, 
                 overclock_power_per_th, price_per_th):
        """
        Initialize ASIC with both standard and overclocked specifications
        
        Args:
            base_hashrate: Standard hashrate in TH/s
            base_power_per_th: Standard power consumption in W/TH
            overclock_hashrate: Overclocked hashrate in TH/s
            overclock_power_per_th: Overclocked power consumption in W/TH
            price_per_th: Cost per TH (same regardless of mode)
        """
        self.base_hashrate = base_hashrate
        self.base_power_per_th = base_power_per_th
        self.overclock_hashrate = overclock_hashrate
        self.overclock_power_per_th = overclock_power_per_th
        self.price_per_th = price_per_th
        
        # Calculate power consumption for each mode
        self.base_power_kw = base_hashrate * base_power_per_th / 1000
        self.overclock_power_kw = overclock_hashrate * overclock_power_per_th / 1000
        
        # Calculate efficiency metrics
        self.base_efficiency = base_hashrate / self.base_power_kw  # TH/kW
        self.overclock_efficiency = overclock_hashrate / self.overclock_power_kw  # TH/kW
        
    def calculate_optimal_operation(self, available_power_kw, n_asics):
        """
        Determine optimal operating mode for given available power
        
        Returns:
            dict with effective_hashrate, power_used, mode_distribution
        """
        base_power_required = n_asics * self.base_power_kw
        overclock_power_required = n_asics * self.overclock_power_kw
        
        if available_power_kw >= overclock_power_required:
            # Full overclocking possible
            return {
                'effective_hashrate': n_asics * self.overclock_hashrate,
                'power_used': overclock_power_required,
                'throttle_factor': 1.0,
                'mode': 'full_overclock',
                'asics_overclocked': n_asics,
                'asics_standard': 0,
                'asics_throttled': 0
            }
            
        elif available_power_kw >= base_power_required:
            # Partial overclocking strategy
            excess_power = available_power_kw - base_power_required
            additional_power_per_asic = self.overclock_power_kw - self.base_power_kw
            
            # How many ASICs can be overclocked with excess power?
            asics_overclock = min(n_asics, int(excess_power / additional_power_per_asic))
            asics_standard = n_asics - asics_overclock
            
            effective_hashrate = (asics_overclock * self.overclock_hashrate + 
                                asics_standard * self.base_hashrate)
            power_used = (asics_overclock * self.overclock_power_kw + 
                         asics_standard * self.base_power_kw)
            
            return {
                'effective_hashrate': effective_hashrate,
                'power_used': power_used,
                'throttle_factor': available_power_kw / base_power_required,
                'mode': 'mixed_operation',
                'asics_overclocked': asics_overclock,
                'asics_standard': asics_standard,
                'asics_throttled': 0
            }
            
        else:
            # Power insufficient for full standard operation - throttle down
            throttle = available_power_kw / base_power_required
            
            return {
                'effective_hashrate': n_asics * self.base_hashrate * throttle,
                'power_used': available_power_kw,
                'throttle_factor': throttle,
                'mode': 'throttled_standard',
                'asics_overclocked': 0,
                'asics_standard': 0,
                'asics_throttled': n_asics
            }

def enhanced_fleet_optimization(hydro_stats, asic_model, max_investment_budget=None):
    """
    Find optimal fleet size considering overclocking capabilities
    
    This function tests different fleet sizes and determines the best
    configuration considering both CAPEX savings and operational flexibility
    """
    results = []
    
    # Test fleet sizes from 1 to maximum possible
    max_fleet_standard = int(hydro_stats['max_power_kw'] / asic_model.base_power_kw)
    max_fleet_overclock = int(hydro_stats['max_power_kw'] / asic_model.overclock_power_kw)
    
    # Test range should go up to the standard power limit since we can mix modes
    for n_asics in range(1, max_fleet_standard + 1):
        
        # Calculate power utilization scenarios
        scenarios = []
        
        # Test at different power levels (percentiles of historical data)
        power_levels = [
            hydro_stats['p10_power_kw'],    # Low power scenario
            hydro_stats['p50_power_kw'],    # Median power
            hydro_stats['p90_power_kw'],    # High power scenario
            hydro_stats['max_power_kw']     # Peak power
        ]
        
        for power_kw in power_levels:
            operation = asic_model.calculate_optimal_operation(power_kw, n_asics)
            scenarios.append(operation)
        
        # Calculate investment metrics
        total_investment = n_asics * asic_model.base_hashrate * asic_model.price_per_th
        
        # Calculate average effective hashrate across scenarios
        avg_hashrate = np.mean([s['effective_hashrate'] for s in scenarios])
        avg_power_utilization = np.mean([s['power_used'] for s in scenarios])
        
        results.append({
            'n_asics': n_asics,
            'total_investment': total_investment,
            'avg_effective_hashrate': avg_hashrate,
            'avg_power_utilization': avg_power_utilization,
            'hashrate_per_dollar': avg_hashrate / total_investment,
            'power_efficiency': avg_hashrate / avg_power_utilization,  # TH/kW
            'scenarios': scenarios
        })
    
    return pd.DataFrame(results)

# Example usage and comparison
def compare_standard_vs_enhanced():
    """Compare traditional fixed-mode vs enhanced overclocking approach"""
    
    # Standard Whatsminer M63s++ specifications
    standard_asic = {
        'hashrate': 460,      # TH/s
        'power_per_th': 15.5, # W/TH
        'price_per_th': 14.90 # $/TH
    }
    
    # Enhanced model with overclocking
    enhanced_asic = EnhancedASICModel(
        base_hashrate=460,
        base_power_per_th=15.5,
        overclock_hashrate=562.5,
        overclock_power_per_th=18.7,
        price_per_th=14.90
    )
    
    # Example power scenario: 1000 kW available (realistic hydro capacity)
    available_power = 1000   # kW
    
    # Standard approach: How many ASICs can we fit?
    standard_power_per_asic = standard_asic['hashrate'] * standard_asic['power_per_th'] / 1000
    standard_max_asics = int(available_power / standard_power_per_asic)
    standard_total_hashrate = standard_max_asics * standard_asic['hashrate']
    standard_investment = standard_max_asics * standard_asic['hashrate'] * standard_asic['price_per_th']
    
    # Enhanced approach: Overclock mode
    enhanced_max_asics = int(available_power / enhanced_asic.overclock_power_kw)
    enhanced_operation = enhanced_asic.calculate_optimal_operation(available_power, enhanced_max_asics)
    enhanced_investment = enhanced_max_asics * enhanced_asic.base_hashrate * enhanced_asic.price_per_th
    
    print("=== ASIC Fleet Comparison for 1000 kW Hydro Facility ===")
    print()
    print("Standard Fixed-Mode Approach:")
    print(f"  Power per ASIC: {standard_power_per_asic:.2f} kW")
    print(f"  Max ASICs: {standard_max_asics}")
    print(f"  Total hashrate: {standard_total_hashrate:,.0f} TH/s")
    print(f"  Investment: ${standard_investment:,.0f}")
    print()
    print("Enhanced Overclocking Approach:")
    print(f"  Power per ASIC (overclock): {enhanced_asic.overclock_power_kw:.2f} kW")
    print(f"  Max ASICs: {enhanced_max_asics} ← This should be ~95!")
    print(f"  Total hashrate: {enhanced_operation['effective_hashrate']:,.0f} TH/s")
    print(f"  Investment: ${enhanced_investment:,.0f}")
    print(f"  Mode: {enhanced_operation['mode']}")
    print()
    print("Key Benefits:")
    print(f"  CAPEX Savings: ${standard_investment - enhanced_investment:,.0f}")
    print(f"  Hashrate Gain: +{enhanced_operation['effective_hashrate'] - standard_total_hashrate:,.0f} TH/s")
    print(f"  Efficiency Gain: {((enhanced_operation['effective_hashrate'] / enhanced_operation['power_used']) / (standard_total_hashrate / (standard_max_asics * standard_power_per_asic)) - 1) * 100:.1f}%")

def demonstrate_mixed_operation():
    """Show how mixed operation works with varying power"""
    
    enhanced_asic = EnhancedASICModel(
        base_hashrate=460,
        base_power_per_th=15.5,
        overclock_hashrate=562.5,
        overclock_power_per_th=18.7,
        price_per_th=14.90
    )
    
    # Install 95 ASICs (max for overclock mode)
    fleet_size = 95
    
    print("=== Mixed Operation Demonstration ===")
    print(f"Installed fleet: {fleet_size} ASICs")
    print(f"Total investment: ${fleet_size * enhanced_asic.base_hashrate * enhanced_asic.price_per_th:,.0f}")
    print()
    
    # Test different power scenarios
    power_scenarios = [
        (700, "Low water (70% capacity)"),
        (850, "Medium water (85% capacity)"),
        (1000, "High water (100% capacity)"),
        (1200, "Peak flow (120% capacity)")
    ]
    
    for power_kw, description in power_scenarios:
        operation = enhanced_asic.calculate_optimal_operation(power_kw, fleet_size)
        
        print(f"{description}: {power_kw} kW available")
        print(f"  Mode: {operation['mode']}")
        print(f"  Overclocked ASICs: {operation['asics_overclocked']}")
        print(f"  Standard ASICs: {operation['asics_standard']}")
        print(f"  Throttled ASICs: {operation['asics_throttled']}")
        print(f"  Effective hashrate: {operation['effective_hashrate']:,.0f} TH/s")
        print(f"  Power utilization: {operation['power_used']:.0f} kW ({operation['power_used']/power_kw*100:.1f}%)")
        print()

if __name__ == "__main__":
    compare_standard_vs_enhanced()
    print("\n" + "="*60 + "\n")
    demonstrate_mixed_operation()
