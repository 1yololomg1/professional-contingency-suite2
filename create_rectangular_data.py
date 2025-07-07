#!/usr/bin/env python3
"""
Create rectangular contingency table data for testing
"""

import pandas as pd
import numpy as np

def create_rectangular_contingency_table():
    """Create a rectangular 3x2 contingency table."""
    
    # Create rectangular data (3 rows x 2 columns)
    data = {
        'Category_A': [30, 20, 15],
        'Category_B': [10, 25, 20]
    }
    
    # Create DataFrame with row labels
    df = pd.DataFrame(data, index=['Group_1', 'Group_2', 'Group_3'])
    
    print("📊 Rectangular Contingency Table Created (3×2):")
    print(df)
    print(f"\n📋 Shape: {df.shape}")
    print(f"📋 Total observations: {df.values.sum()}")
    print(f"📋 Rows: {len(df.index)}")
    print(f"📋 Columns: {len(df.columns)}")
    
    # Save to Excel
    filename = "rectangular_contingency_table.xlsx"
    df.to_excel(filename, index=True)
    
    print(f"\n✅ Saved to: {filename}")
    print("This rectangular matrix will work perfectly for analysis!")
    
    return df

if __name__ == "__main__":
    create_rectangular_contingency_table() 