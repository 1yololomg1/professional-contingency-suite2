#!/usr/bin/env python3
"""
Create sample contingency table data for testing
"""

import pandas as pd
import numpy as np

def create_sample_contingency_table():
    """Create a sample 2x2 contingency table."""
    
    # Create sample data
    data = {
        'Category_A': [25, 15],
        'Category_B': [10, 30]
    }
    
    # Create DataFrame with row labels
    df = pd.DataFrame(data, index=['Group_1', 'Group_2'])
    
    print("📊 Sample Contingency Table Created:")
    print(df)
    print(f"\n📋 Shape: {df.shape}")
    print(f"📋 Total observations: {df.values.sum()}")
    
    # Save to Excel
    filename = "sample_contingency_table.xlsx"
    df.to_excel(filename, index=True)
    
    print(f"\n✅ Saved to: {filename}")
    print("You can now use this file to test the analysis!")
    
    return df

if __name__ == "__main__":
    create_sample_contingency_table() 