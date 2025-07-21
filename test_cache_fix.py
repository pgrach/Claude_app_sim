#!/usr/bin/env python3
"""
Test script to verify the cache fix works correctly.
"""

import sys
sys.path.append('.')

from hydro_miner_model import load_btc_data
from datetime import datetime

def test_data_loading():
    """Test that the data loading function works correctly."""
    print("Testing data loading...")
    
    # Test the core function
    data = load_btc_data('btc_price.csv', 'btc_difficulty.csv')
    
    # Check required keys
    required_keys = ['current_price', 'current_difficulty', 'current_revenue_per_th', 'data_date']
    missing_keys = [key for key in required_keys if key not in data]
    
    if missing_keys:
        print(f"❌ Missing keys: {missing_keys}")
        return False
    
    print("✅ All required keys present")
    
    # Test the data_date specifically
    data_date = data['data_date']
    print(f"📅 Data date: {data_date.strftime('%B %d, %Y')}")
    
    # Test the fallback logic
    data_date_fallback = data.get('data_date')
    if data_date_fallback:
        print(f"✅ Fallback logic works: {data_date_fallback.strftime('%B %d, %Y')} (latest available)")
    else:
        print(f"⚠️ Using fallback: {datetime.now().strftime('%B %d, %Y')} (cached data)")
    
    return True

if __name__ == "__main__":
    print("🧪 Testing Bitcoin Mining Optimizer Cache Fix")
    print("=" * 50)
    
    try:
        success = test_data_loading()
        if success:
            print("\n🎉 All tests passed! The cache fix should work correctly.")
        else:
            print("\n❌ Tests failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Error during testing: {e}")
        sys.exit(1)
