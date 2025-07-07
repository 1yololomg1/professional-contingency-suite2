#!/usr/bin/env python3
"""
Contingency Processor Module
Professional Contingency Analysis Suite
Processes contingency squares from Excel data for downstream analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from datetime import datetime
from scipy import stats


class ContingencyProcessor:
    """
    Processes contingency squares from Excel data.
    Handles data cleaning, normalization, and preparation for analysis.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.processing_options = {
            "remove_zero_rows": True,
            "remove_zero_columns": True,
            "handle_missing_values": "exclude",
            "minimum_cell_value": 0,
            "normalize_tables": False,
            "enforce_integer_counts": True,
            "merge_small_categories": False,
            "small_category_threshold": 5
        }
    
    def process_contingency_squares(self, excel_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process contingency squares from Excel data.
        
        Args:
            excel_data: Dictionary containing Excel sheet data
            
        Returns:
            Dictionary with processed contingency data
        """
        try:
            self.logger.info("Starting contingency squares processing")
            
            processed_result = {
                "processed_tables": {},
                "processing_summary": {},
                "metadata": {
                    "processing_timestamp": datetime.now().isoformat(),
                    "processing_options": self.processing_options.copy()
                }
            }
            
            # Extract sheets from Excel data
            sheets = excel_data.get("sheets", {})
            if not sheets:
                raise ValueError("No sheets found in Excel data")
            
            # Process each sheet
            for sheet_name, sheet_data in sheets.items():
                self.logger.info(f"Processing sheet: {sheet_name}")
                
                # Process individual contingency table
                processed_table = self._process_individual_table(sheet_data, sheet_name)
                processed_result["processed_tables"][sheet_name] = processed_table
                
                # Generate processing summary for this table
                summary = self._generate_table_summary(processed_table, sheet_name)
                processed_result["processing_summary"][sheet_name] = summary
            
            self.logger.info("Contingency squares processing completed successfully")
            return processed_result
            
        except Exception as e:
            self.logger.error(f"Contingency processing failed: {str(e)}")
            raise
    
    def _process_individual_table(self, sheet_data: pd.DataFrame, sheet_name: str) -> Dict[str, Any]:
        """Process an individual contingency table."""
        try:
            # Initialize processed table structure
            processed_table = {
                "original_data": sheet_data.copy(),
                "processed_data": None,
                "transformations_applied": [],
                "quality_metrics": {},
                "warnings": []
            }
            
            # Start with original data
            current_data = sheet_data.copy()
            
            # Step 1: Extract numeric data
            numeric_columns = current_data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                raise ValueError(f"No numeric columns found in sheet '{sheet_name}'")
            
            # Use only numeric columns
            current_data = current_data[numeric_columns]
            processed_table["transformations_applied"].append("extracted_numeric_columns")
            
            # Step 2: Handle missing values
            if current_data.isnull().any().any():
                if self.processing_options["handle_missing_values"] == "exclude":
                    current_data = current_data.dropna()
                    processed_table["transformations_applied"].append("removed_missing_values")
                elif self.processing_options["handle_missing_values"] == "zero":
                    current_data = current_data.fillna(0)
                    processed_table["transformations_applied"].append("filled_missing_with_zero")
                elif self.processing_options["handle_missing_values"] == "mean":
                    current_data = current_data.fillna(current_data.mean())
                    processed_table["transformations_applied"].append("filled_missing_with_mean")
            
            # Step 3: Enforce minimum cell values
            if self.processing_options["minimum_cell_value"] > 0:
                current_data = current_data.clip(lower=self.processing_options["minimum_cell_value"])
                processed_table["transformations_applied"].append("enforced_minimum_cell_value")
            
            # Step 4: Enforce integer counts (if required)
            if self.processing_options["enforce_integer_counts"]:
                current_data = current_data.round().astype(int)
                processed_table["transformations_applied"].append("enforced_integer_counts")
            
            # Step 5: Remove zero rows/columns (if required)
            if self.processing_options["remove_zero_rows"]:
                zero_rows = (current_data == 0).all(axis=1)
                if zero_rows.any():
                    current_data = current_data.loc[~zero_rows]
                    processed_table["transformations_applied"].append("removed_zero_rows")
                    processed_table["warnings"].append(f"Removed {zero_rows.sum()} zero rows")
            
            if self.processing_options["remove_zero_columns"]:
                zero_cols = (current_data == 0).all(axis=0)
                if zero_cols.any():
                    current_data = current_data.loc[:, ~zero_cols]
                    processed_table["transformations_applied"].append("removed_zero_columns")
                    processed_table["warnings"].append(f"Removed {zero_cols.sum()} zero columns")
            
            # Step 6: Merge small categories (if required)
            if self.processing_options["merge_small_categories"]:
                current_data = self._merge_small_categories(current_data, processed_table)
            
            # Step 7: Normalize tables (if required)
            if self.processing_options["normalize_tables"]:
                current_data = self._normalize_table(current_data, processed_table)
            
            # Step 8: Validate processed data
            self._validate_processed_data(current_data, processed_table)
            
            # Step 9: Calculate quality metrics
            processed_table["quality_metrics"] = self._calculate_quality_metrics(current_data)
            
            # Store final processed data
            processed_table["processed_data"] = current_data
            
            return processed_table
            
        except Exception as e:
            self.logger.error(f"Error processing table '{sheet_name}': {str(e)}")
            raise
    
    def _merge_small_categories(self, data: pd.DataFrame, processed_table: Dict[str, Any]) -> pd.DataFrame:
        """Merge categories with small counts."""
        try:
            threshold = self.processing_options["small_category_threshold"]
            
            # Check row sums
            row_sums = data.sum(axis=1)
            small_rows = row_sums < threshold
            
            if small_rows.any():
                # Merge small rows into "Other" category
                other_row = data.loc[small_rows].sum()
                data_merged = data.loc[~small_rows].copy()
                data_merged.loc["Other"] = other_row
                
                processed_table["transformations_applied"].append("merged_small_row_categories")
                processed_table["warnings"].append(f"Merged {small_rows.sum()} small row categories")
                
                return data_merged
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error merging small categories: {str(e)}")
            return data
    
    def _normalize_table(self, data: pd.DataFrame, processed_table: Dict[str, Any]) -> pd.DataFrame:
        """Normalize contingency table."""
        try:
            # Normalize by total sum
            total_sum = data.sum().sum()
            if total_sum > 0:
                normalized_data = data / total_sum
                processed_table["transformations_applied"].append("normalized_by_total")
                return normalized_data
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error normalizing table: {str(e)}")
            return data
    
    def _validate_processed_data(self, data: pd.DataFrame, processed_table: Dict[str, Any]) -> None:
        """Validate processed contingency data."""
        try:
            # Check for empty data
            if data.empty:
                raise ValueError("Processed data is empty")
            
            # Check for negative values
            if (data < 0).any().any():
                processed_table["warnings"].append("Processed data contains negative values")
            
            # Check for infinite values
            if np.isinf(data.values).any():
                raise ValueError("Processed data contains infinite values")
            
            # Check for NaN values
            if data.isnull().any().any():
                raise ValueError("Processed data contains NaN values")
            
            # Check minimum dimensions
            if data.shape[0] < 2 or data.shape[1] < 2:
                processed_table["warnings"].append("Processed data has very small dimensions")
            
            # Check for zero-only data
            if (data == 0).all().all():
                raise ValueError("Processed data contains only zeros")
            
        except Exception as e:
            self.logger.error(f"Data validation failed: {str(e)}")
            raise
    
    def _calculate_quality_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate quality metrics for processed data."""
        try:
            metrics = {
                "dimensions": data.shape,
                "total_observations": data.sum().sum(),
                "non_zero_cells": (data > 0).sum().sum(),
                "sparsity": (data == 0).sum().sum() / data.size,
                "min_value": data.min().min(),
                "max_value": data.max().max(),
                "mean_value": data.mean().mean(),
                "std_value": data.std().std(),
                "row_totals": data.sum(axis=1).tolist(),
                "column_totals": data.sum(axis=0).tolist()
            }
            
            # Calculate additional metrics
            total_obs = metrics["total_observations"]
            if total_obs > 0:
                # Expected frequencies for independence
                row_totals = data.sum(axis=1)
                col_totals = data.sum(axis=0)
                
                expected_freq = []
                for i in range(data.shape[0]):
                    for j in range(data.shape[1]):
                        expected = (row_totals.iloc[i] * col_totals.iloc[j]) / total_obs
                        expected_freq.append(expected)
                
                metrics["expected_frequencies"] = expected_freq
                metrics["min_expected_frequency"] = min(expected_freq)
                metrics["cells_with_low_expected_freq"] = sum(1 for f in expected_freq if f < 5)
                
                # Chi-square statistic calculation
                chi_square = 0
                for i in range(data.shape[0]):
                    for j in range(data.shape[1]):
                        observed = data.iloc[i, j]
                        expected = (row_totals.iloc[i] * col_totals.iloc[j]) / total_obs
                        if expected > 0:
                            chi_square += ((observed - expected) ** 2) / expected
                
                metrics["chi_square_statistic"] = chi_square
                metrics["degrees_of_freedom"] = (data.shape[0] - 1) * (data.shape[1] - 1)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating quality metrics: {str(e)}")
            return {"error": str(e)}
    
    def _generate_table_summary(self, processed_table: Dict[str, Any], sheet_name: str) -> Dict[str, Any]:
        """Generate summary for processed table."""
        try:
            summary = {
                "sheet_name": sheet_name,
                "processing_successful": processed_table["processed_data"] is not None,
                "transformations_count": len(processed_table["transformations_applied"]),
                "transformations": processed_table["transformations_applied"],
                "warnings_count": len(processed_table["warnings"]),
                "warnings": processed_table["warnings"],
                "quality_metrics": processed_table["quality_metrics"]
            }
            
            if processed_table["processed_data"] is not None:
                data = processed_table["processed_data"]
                summary.update({
                    "final_dimensions": data.shape,
                    "total_observations": data.sum().sum(),
                    "data_density": (data > 0).sum().sum() / data.size
                })
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating table summary: {str(e)}")
            return {"error": str(e)}
    
    def get_processing_options(self) -> Dict[str, Any]:
        """Get current processing options."""
        return self.processing_options.copy()
    
    def set_processing_options(self, options: Dict[str, Any]) -> None:
        """Set processing options."""
        self.processing_options.update(options)
        self.logger.info(f"Processing options updated: {options}")
    
    def get_contingency_statistics(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive statistics for all processed tables."""
        try:
            statistics = {
                "overall_summary": {
                    "total_tables": len(processed_data.get("processed_tables", {})),
                    "successful_processing": 0,
                    "total_observations": 0,
                    "average_dimensions": [0, 0]
                },
                "table_statistics": {}
            }
            
            processed_tables = processed_data.get("processed_tables", {})
            
            dimensions = []
            for table_name, table_data in processed_tables.items():
                if table_data["processed_data"] is not None:
                    statistics["overall_summary"]["successful_processing"] += 1
                    
                    data = table_data["processed_data"]
                    obs = data.sum().sum()
                    statistics["overall_summary"]["total_observations"] += obs
                    dimensions.append(data.shape)
                    
                    # Individual table statistics
                    statistics["table_statistics"][table_name] = {
                        "dimensions": data.shape,
                        "observations": obs,
                        "density": (data > 0).sum().sum() / data.size,
                        "quality_score": self._calculate_quality_score(table_data["quality_metrics"])
                    }
            
            # Calculate average dimensions
            if dimensions:
                avg_rows = sum(d[0] for d in dimensions) / len(dimensions)
                avg_cols = sum(d[1] for d in dimensions) / len(dimensions)
                statistics["overall_summary"]["average_dimensions"] = [avg_rows, avg_cols]
            
            return statistics
            
        except Exception as e:
            self.logger.error(f"Error calculating contingency statistics: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_quality_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate a quality score for the processed data."""
        try:
            score = 0.0
            
            # Dimension score (prefer reasonable sized tables)
            dims = metrics.get("dimensions", [0, 0])
            if 2 <= dims[0] <= 10 and 2 <= dims[1] <= 10:
                score += 0.3
            elif dims[0] >= 2 and dims[1] >= 2:
                score += 0.2
            
            # Observation count score
            obs = metrics.get("total_observations", 0)
            if obs >= 100:
                score += 0.3
            elif obs >= 20:
                score += 0.2
            elif obs >= 10:
                score += 0.1
            
            # Sparsity score (prefer less sparse data)
            sparsity = metrics.get("sparsity", 1.0)
            if sparsity < 0.3:
                score += 0.2
            elif sparsity < 0.5:
                score += 0.1
            
            # Expected frequency score
            min_expected = metrics.get("min_expected_frequency", 0)
            if min_expected >= 5:
                score += 0.2
            elif min_expected >= 1:
                score += 0.1
            
            return min(score, 1.0)
            
        except Exception:
            return 0.0
