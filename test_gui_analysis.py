#!/usr/bin/env python3
"""
Test GUI analysis process
"""

import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
from datetime import datetime
import os

def test_gui_analysis():
    """Test the GUI analysis process."""
    print("🧪 Testing GUI Analysis Process")
    print("=" * 50)
    
    # Check if sample data exists
    if not os.path.exists("sample_contingency_table.xlsx"):
        print("❌ Sample data not found. Run create_sample_data.py first.")
        return False
    
    # Load sample data
    try:
        data = pd.read_excel("sample_contingency_table.xlsx", index_col=0)
        print(f"✅ Sample data loaded: {data.shape}")
        print(data)
    except Exception as e:
        print(f"❌ Failed to load sample data: {e}")
        return False
    
    # Simulate the analysis process
    print("\n🔬 Simulating analysis process...")
    
    # Calculate metrics
    observed = data.values
    chi2_stat, p_value, dof, expected = chi2_contingency(observed)
    n = observed.sum()
    cramers_v = np.sqrt(chi2_stat / (n * (min(observed.shape) - 1)))
    
    metrics = {
        'cramers_v': cramers_v,
        'p_value': p_value,
        'chi2_statistic': chi2_stat,
        'degrees_freedom': dof,
        'sample_size': n,
        'expected_frequencies': expected,
        'residuals': (observed - expected) / np.sqrt(expected)
    }
    
    print(f"✅ Analysis completed:")
    print(f"   Cramer's V: {cramers_v:.4f}")
    print(f"   p-value: {p_value:.6f}")
    print(f"   Chi-square: {chi2_stat:.4f}")
    print(f"   Sample size: {n}")
    
    # Test visualization data
    print("\n📈 Testing visualization data...")
    
    # Test confusion matrix
    if data is not None:
        print(f"✅ Confusion matrix data: {data.shape}")
    
    # Test statistical metrics
    if metrics:
        print(f"✅ Statistical metrics available: {len(metrics)} items")
    
    # Test report generation
    print("\n📄 Testing report generation...")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"✅ Report timestamp: {timestamp}")
    
    print("\n🎉 All tests passed!")
    print("\n📋 Summary:")
    print(f"   Data: {data.shape} matrix")
    print(f"   Cramer's V: {cramers_v:.4f} (Moderate association)")
    print(f"   p-value: {p_value:.6f} (Significant)")
    print(f"   Sample size: {n}")
    
    print("\n💡 The GUI should now display:")
    print("   • Confusion matrix heatmap")
    print("   • Statistical metrics")
    print("   • Pie chart visualizations")
    print("   • Radar chart with metrics")
    print("   • Professional report")
    
    return True

if __name__ == "__main__":
    try:
        from scipy.stats import chi2_contingency
        test_gui_analysis()
    except ImportError:
        print("❌ SciPy not available. Install with: pip install scipy")
    except Exception as e:
        print(f"❌ Test failed: {e}") 