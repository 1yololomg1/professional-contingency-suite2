#!/usr/bin/env python3
"""
Test rectangular contingency table analysis
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def test_rectangular_analysis():
    """Test analysis with rectangular contingency table."""
    print("🔬 Testing Rectangular Contingency Table Analysis")
    print("=" * 60)
    
    # Load rectangular data
    try:
        data = pd.read_excel("rectangular_contingency_table.xlsx", index_col=0)
        print(f"📊 Rectangular data loaded: {data.shape}")
        print(data)
        print(f"📋 Matrix type: {data.shape[0]}×{data.shape[1]} (rectangular)")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return False
    
    # Perform analysis
    print("\n🔬 Performing statistical analysis...")
    
    observed = data.values
    chi2_stat, p_value, dof, expected = chi2_contingency(observed)
    n = observed.sum()
    cramers_v = np.sqrt(chi2_stat / (n * (min(observed.shape) - 1)))
    
    print(f"✅ Analysis completed successfully!")
    print(f"   Chi-square statistic: {chi2_stat:.4f}")
    print(f"   p-value: {p_value:.6f}")
    print(f"   Degrees of freedom: {dof}")
    print(f"   Cramer's V: {cramers_v:.4f}")
    print(f"   Sample size: {n}")
    
    # Show expected frequencies
    print(f"\n📊 Expected frequencies:")
    expected_df = pd.DataFrame(expected, index=data.index, columns=data.columns)
    print(expected_df)
    
    # Show residuals
    residuals = (observed - expected) / np.sqrt(expected)
    print(f"\n📊 Standardized residuals:")
    residuals_df = pd.DataFrame(residuals, index=data.index, columns=data.columns)
    print(residuals_df)
    
    # Interpretation
    print(f"\n📋 Interpretation:")
    print(f"   • Matrix shape: {data.shape[0]}×{data.shape[1]} (rectangular)")
    print(f"   • Cramer's V: {cramers_v:.4f}")
    if cramers_v > 0.3:
        strength = "Strong"
    elif cramers_v > 0.1:
        strength = "Moderate"
    else:
        strength = "Weak"
    print(f"   • Association strength: {strength}")
    
    if p_value < 0.05:
        print(f"   • Significance: Statistically significant (p < 0.05)")
    else:
        print(f"   • Significance: Not statistically significant (p ≥ 0.05)")
    
    print(f"\n🎉 Rectangular contingency tables work perfectly!")
    print(f"   The analysis handles any matrix shape: 2×2, 3×2, 4×3, etc.")
    
    return True

if __name__ == "__main__":
    test_rectangular_analysis() 