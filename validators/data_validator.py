#!/usr/bin/env python3
"""
Data Validation Module
Professional Contingency Analysis Suite
Validates input data structure and content
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from datetime import datetime


class DataValidator:
    """
    Validates contingency table data from Excel files.
    Ensures data integrity and proper formatting before analysis.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validation_rules = {
            "min_categories": 2,
            "max_categories": 50,
            "min_observations": 5,
            "max_missing_percentage": 0.05,
            "required_numeric_columns": True,
            "symmetric_tables_only": False
        }
    
    def validate_contingency_data(self, excel_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive validation of contingency table data.
        
        Args:
            excel_data: Dictionary containing Excel sheet data
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "data_summary": {},
            "validation_timestamp": datetime.now().isoformat()
        }
        
        try:
            # Validate data structure
            structure_validation = self._validate_data_structure(excel_data)
            validation_result["errors"].extend(structure_validation["errors"])
            validation_result["warnings"].extend(structure_validation["warnings"])
            
            # Validate data content
            content_validation = self._validate_data_content(excel_data)
            validation_result["errors"].extend(content_validation["errors"])
            validation_result["warnings"].extend(content_validation["warnings"])
            
            # Validate contingency table properties
            contingency_validation = self._validate_contingency_properties(excel_data)
            validation_result["errors"].extend(contingency_validation["errors"])
            validation_result["warnings"].extend(contingency_validation["warnings"])
            
            # Statistical validation
            statistical_validation = self._validate_statistical_assumptions(excel_data)
            validation_result["errors"].extend(statistical_validation["errors"])
            validation_result["warnings"].extend(statistical_validation["warnings"])
            
            # Generate data summary
            validation_result["data_summary"] = self._generate_data_summary(excel_data)
            
            # Set final validation status
            validation_result["valid"] = len(validation_result["errors"]) == 0
            
            if validation_result["valid"]:
                self.logger.info("Data validation passed successfully")
            else:
                self.logger.error(f"Data validation failed with {len(validation_result['errors'])} errors")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Data validation error: {str(e)}")
            validation_result["valid"] = False
            validation_result["errors"].append(f"Validation process failed: {str(e)}")
            return validation_result
    
    def _validate_data_structure(self, excel_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate the basic structure of Excel data."""
        errors = []
        warnings = []
        
        try:
            # Check if data exists
            if not excel_data or not isinstance(excel_data, dict):
                errors.append("Excel data is empty or not in expected format")
                return {"errors": errors, "warnings": warnings}
            
            # Check for required sheets
            if "sheets" not in excel_data:
                errors.append("No sheet information found in Excel data")
                return {"errors": errors, "warnings": warnings}
            
            sheets = excel_data["sheets"]
            if not sheets:
                errors.append("No sheets found in Excel file")
                return {"errors": errors, "warnings": warnings}
            
            # Validate each sheet
            for sheet_name, sheet_data in sheets.items():
                if not isinstance(sheet_data, pd.DataFrame):
                    errors.append(f"Sheet '{sheet_name}' is not in DataFrame format")
                    continue
                
                # Check for empty sheets
                if sheet_data.empty:
                    warnings.append(f"Sheet '{sheet_name}' is empty")
                    continue
                
                # Check minimum dimensions
                if sheet_data.shape[0] < 2 or sheet_data.shape[1] < 2:
                    errors.append(f"Sheet '{sheet_name}' has insufficient dimensions for contingency analysis")
                
                # Check for proper column names
                if sheet_data.columns.duplicated().any():
                    errors.append(f"Sheet '{sheet_name}' has duplicate column names")
                
                # Check for proper index
                if sheet_data.index.duplicated().any():
                    errors.append(f"Sheet '{sheet_name}' has duplicate row indices")
            
        except Exception as e:
            errors.append(f"Structure validation error: {str(e)}")
        
        return {"errors": errors, "warnings": warnings}
    
    def _validate_data_content(self, excel_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate the content of contingency table data."""
        errors = []
        warnings = []
        
        try:
            sheets = excel_data.get("sheets", {})
            
            for sheet_name, sheet_data in sheets.items():
                if not isinstance(sheet_data, pd.DataFrame):
                    continue
                
                # Check for numeric data
                numeric_columns = sheet_data.select_dtypes(include=[np.number]).columns
                if len(numeric_columns) == 0:
                    errors.append(f"Sheet '{sheet_name}' contains no numeric data")
                    continue
                
                # Check for negative values
                numeric_data = sheet_data[numeric_columns]
                if (numeric_data < 0).any().any():
                    errors.append(f"Sheet '{sheet_name}' contains negative values")
                
                # Check for missing values
                missing_percentage = (sheet_data.isnull().sum().sum() / sheet_data.size) * 100
                if missing_percentage > self.validation_rules["max_missing_percentage"] * 100:
                    errors.append(f"Sheet '{sheet_name}' has {missing_percentage:.2f}% missing values (exceeds {self.validation_rules['max_missing_percentage']*100}% threshold)")
                elif missing_percentage > 0:
                    warnings.append(f"Sheet '{sheet_name}' has {missing_percentage:.2f}% missing values")
                
                # Check for extremely large values
                if (numeric_data > 1e6).any().any():
                    warnings.append(f"Sheet '{sheet_name}' contains very large values (>1,000,000)")
                
                # Check for zero-only rows/columns
                zero_rows = (numeric_data == 0).all(axis=1).sum()
                zero_cols = (numeric_data == 0).all(axis=0).sum()
                
                if zero_rows > 0:
                    warnings.append(f"Sheet '{sheet_name}' has {zero_rows} rows with all zeros")
                if zero_cols > 0:
                    warnings.append(f"Sheet '{sheet_name}' has {zero_cols} columns with all zeros")
        
        except Exception as e:
            errors.append(f"Content validation error: {str(e)}")
        
        return {"errors": errors, "warnings": warnings}
    
    def _validate_contingency_properties(self, excel_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate specific properties required for contingency analysis."""
        errors = []
        warnings = []
        
        try:
            sheets = excel_data.get("sheets", {})
            
            for sheet_name, sheet_data in sheets.items():
                if not isinstance(sheet_data, pd.DataFrame):
                    continue
                
                numeric_columns = sheet_data.select_dtypes(include=[np.number]).columns
                if len(numeric_columns) == 0:
                    continue
                
                numeric_data = sheet_data[numeric_columns]
                
                # Check category count
                if len(numeric_data.columns) < self.validation_rules["min_categories"]:
                    errors.append(f"Sheet '{sheet_name}' has too few categories ({len(numeric_data.columns)} < {self.validation_rules['min_categories']})")
                
                if len(numeric_data.columns) > self.validation_rules["max_categories"]:
                    warnings.append(f"Sheet '{sheet_name}' has many categories ({len(numeric_data.columns)} > {self.validation_rules['max_categories']})")
                
                # Check observation count
                total_observations = numeric_data.sum().sum()
                if total_observations < self.validation_rules["min_observations"]:
                    errors.append(f"Sheet '{sheet_name}' has too few observations ({total_observations} < {self.validation_rules['min_observations']})")
                
                # Check for square contingency table if required
                if self.validation_rules["symmetric_tables_only"]:
                    if numeric_data.shape[0] != numeric_data.shape[1]:
                        errors.append(f"Sheet '{sheet_name}' is not a square contingency table")
                
                # Check for expected frequency assumptions
                expected_freq_violations = 0
                for col in numeric_data.columns:
                    col_total = numeric_data[col].sum()
                    row_totals = numeric_data.sum(axis=1)
                    grand_total = numeric_data.sum().sum()
                    
                    for idx in numeric_data.index:
                        if grand_total > 0:
                            expected_freq = (row_totals[idx] * col_total) / grand_total
                            if expected_freq < 5:
                                expected_freq_violations += 1
                
                if expected_freq_violations > 0:
                    violation_percentage = (expected_freq_violations / (numeric_data.shape[0] * numeric_data.shape[1])) * 100
                    if violation_percentage > 20:
                        errors.append(f"Sheet '{sheet_name}' has {violation_percentage:.1f}% cells with expected frequency < 5")
                    else:
                        warnings.append(f"Sheet '{sheet_name}' has {violation_percentage:.1f}% cells with expected frequency < 5")
        
        except Exception as e:
            errors.append(f"Contingency properties validation error: {str(e)}")
        
        return {"errors": errors, "warnings": warnings}
    
    def _validate_statistical_assumptions(self, excel_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate statistical assumptions for contingency analysis."""
        errors = []
        warnings = []
        
        try:
            sheets = excel_data.get("sheets", {})
            
            for sheet_name, sheet_data in sheets.items():
                if not isinstance(sheet_data, pd.DataFrame):
                    continue
                
                numeric_columns = sheet_data.select_dtypes(include=[np.number]).columns
                if len(numeric_columns) == 0:
                    continue
                
                numeric_data = sheet_data[numeric_columns]
                
                # Check for independence assumption
                # This is a basic check - actual independence testing would require more context
                total_observations = numeric_data.sum().sum()
                if total_observations > 0:
                    # Check for extreme skewness in marginal distributions
                    row_totals = numeric_data.sum(axis=1)
                    col_totals = numeric_data.sum(axis=0)
                    
                    # Check row distribution
                    if len(row_totals) > 1:
                        row_cv = row_totals.std() / row_totals.mean()
                        if row_cv > 2.0:
                            warnings.append(f"Sheet '{sheet_name}' has highly skewed row marginals (CV={row_cv:.2f})")
                    
                    # Check column distribution
                    if len(col_totals) > 1:
                        col_cv = col_totals.std() / col_totals.mean()
                        if col_cv > 2.0:
                            warnings.append(f"Sheet '{sheet_name}' has highly skewed column marginals (CV={col_cv:.2f})")
                
                # Check for sparse data
                zero_percentage = (numeric_data == 0).sum().sum() / numeric_data.size * 100
                if zero_percentage > 50:
                    warnings.append(f"Sheet '{sheet_name}' is sparse with {zero_percentage:.1f}% zeros")
        
        except Exception as e:
            errors.append(f"Statistical assumptions validation error: {str(e)}")
        
        return {"errors": errors, "warnings": warnings}
    
    def _generate_data_summary(self, excel_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of the data for reporting."""
        summary = {
            "total_sheets": 0,
            "valid_sheets": 0,
            "sheet_details": {}
        }
        
        try:
            sheets = excel_data.get("sheets", {})
            summary["total_sheets"] = len(sheets)
            
            for sheet_name, sheet_data in sheets.items():
                if not isinstance(sheet_data, pd.DataFrame):
                    continue
                
                numeric_columns = sheet_data.select_dtypes(include=[np.number]).columns
                if len(numeric_columns) == 0:
                    continue
                
                summary["valid_sheets"] += 1
                numeric_data = sheet_data[numeric_columns]
                
                sheet_summary = {
                    "dimensions": numeric_data.shape,
                    "total_observations": numeric_data.sum().sum(),
                    "missing_values": numeric_data.isnull().sum().sum(),
                    "zero_values": (numeric_data == 0).sum().sum(),
                    "min_value": numeric_data.min().min(),
                    "max_value": numeric_data.max().max(),
                    "mean_value": numeric_data.mean().mean(),
                    "row_totals": numeric_data.sum(axis=1).tolist(),
                    "column_totals": numeric_data.sum(axis=0).tolist()
                }
                
                summary["sheet_details"][sheet_name] = sheet_summary
        
        except Exception as e:
            summary["error"] = f"Summary generation error: {str(e)}"
        
        return summary
    
    def set_validation_rules(self, rules: Dict[str, Any]) -> None:
        """Update validation rules."""
        self.validation_rules.update(rules)
        self.logger.info(f"Validation rules updated: {rules}")
    
    def get_validation_rules(self) -> Dict[str, Any]:
        """Get current validation rules."""
        return self.validation_rules.copy()
