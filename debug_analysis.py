#!/usr/bin/env python3
"""
Debug script to test analysis process
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import os

def test_excel_loading():
    """Test if we can load Excel files."""
    print("🔍 Testing Excel file loading...")
    
    # Check if we have any Excel files in the current directory
    excel_files = []
    for file in os.listdir('.'):
        if file.endswith(('.xlsx', '.xls')):
            excel_files.append(file)
    
    if not excel_files:
        print("❌ No Excel files found in current directory")
        print("Please place an Excel file with contingency table data in this folder")
        return None
    
    print(f"✅ Found Excel files: {excel_files}")
    
    # Try to load the first Excel file
    try:
        file_path = excel_files[0]
        print(f"📊 Loading: {file_path}")
        
        # Try different ways to load
        try:
            data = pd.read_excel(file_path, index_col=0)
            print(f"✅ Successfully loaded with index_col=0")
        except:
            try:
                data = pd.read_excel(file_path)
                print(f"✅ Successfully loaded without index_col")
            except Exception as e:
                print(f"❌ Failed to load: {e}")
                return None
        
        print(f"📋 Data shape: {data.shape}")
        print(f"📋 Data columns: {list(data.columns)}")
        print(f"📋 Data index: {list(data.index)}")
        print(f"📋 Data preview:")
        print(data.head())
        
        return data
        
    except Exception as e:
        print(f"❌ Error loading Excel file: {e}")
        return None

def test_statistical_calculation(data):
    """Test statistical calculations."""
    print("\n🔬 Testing statistical calculations...")
    
    try:
        # Convert to numpy array
        observed = data.values
        print(f"📊 Observed data shape: {observed.shape}")
        print(f"📊 Observed data:\n{observed}")
        
        # Calculate chi-square
        chi2_stat, p_value, dof, expected = chi2_contingency(observed)
        print(f"✅ Chi-square test completed")
        print(f"   Chi-square statistic: {chi2_stat:.4f}")
        print(f"   p-value: {p_value:.6f}")
        print(f"   Degrees of freedom: {dof}")
        
        # Calculate Cramer's V
        n = observed.sum()
        cramers_v = np.sqrt(chi2_stat / (n * (min(observed.shape) - 1)))
        print(f"   Cramer's V: {cramers_v:.4f}")
        
        # Calculate residuals
        residuals = (observed - expected) / np.sqrt(expected)
        print(f"   Residuals range: [{residuals.min():.3f}, {residuals.max():.3f}]")
        
        return {
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'degrees_freedom': dof,
            'cramers_v': cramers_v,
            'expected_frequencies': expected,
            'residuals': residuals,
            'sample_size': n
        }
        
    except Exception as e:
        print(f"❌ Error in statistical calculation: {e}")
        return None

def test_visualization_data(data, metrics):
    """Test if we have the right data for visualizations."""
    print("\n📈 Testing visualization data...")
    
    try:
        # Test confusion matrix data
        print(f"✅ Confusion matrix data available: {data.shape}")
        
        # Test statistical metrics
        if metrics:
            print(f"✅ Statistical metrics available:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"   {key}: {value:.4f}")
                else:
                    print(f"   {key}: {type(value).__name__}")
        else:
            print("❌ No statistical metrics available")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in visualization data: {e}")
        return False

def main():
    """Main debug function."""
    print("🐛 Professional Contingency Analysis Suite - Debug Mode")
    print("=" * 60)
    
    # Test 1: Excel loading
    data = test_excel_loading()
    if data is None:
        print("\n❌ Cannot proceed without data")
        return
    
    # Test 2: Statistical calculations
    metrics = test_statistical_calculation(data)
    if metrics is None:
        print("\n❌ Cannot proceed without statistical calculations")
        return
    
    # Test 3: Visualization data
    viz_ok = test_visualization_data(data, metrics)
    if not viz_ok:
        print("\n❌ Visualization data preparation failed")
        return
    
    print("\n🎉 All tests passed! The analysis should work.")
    print("\n📋 Summary:")
    print(f"   Data loaded: {data.shape}")
    print(f"   Cramer's V: {metrics['cramers_v']:.4f}")
    print(f"   p-value: {metrics['p_value']:.6f}")
    print(f"   Sample size: {metrics['sample_size']}")
    
    print("\n💡 If the GUI is not showing results, the issue is in the GUI update process, not the analysis itself.")

if __name__ == "__main__":
    main() 