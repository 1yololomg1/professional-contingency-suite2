#!/usr/bin/env python3
"""
Confusion Matrix Converter Module
Professional Contingency Analysis Suite
Converts contingency tables to confusion matrices for classification analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from datetime import datetime
from sklearn.metrics import confusion_matrix as sk_confusion_matrix


class ConfusionMatrixConverter:
    """
    Converts contingency tables to confusion matrices.
    Handles various matrix formats and provides classification metrics.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.conversion_options = {
            "matrix_format": "standard",  # "standard", "normalized", "percentage"
            "include_marginals": True,
            "preserve_labels": True,
            "handle_multiclass": True,
            "binary_threshold": 0.5,
            "normalization_method": "none",  # "none", "true", "pred", "all"
            "class_ordering": "default"  # "default", "alphabetical", "frequency"
        }
    
    def convert_to_confusion_matrices(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert processed contingency tables to confusion matrices.
        
        Args:
            processed_data: Dictionary containing processed contingency data
            
        Returns:
            Dictionary with confusion matrices and metadata
        """
        try:
            self.logger.info("Starting confusion matrix conversion")
            
            conversion_result = {
                "matrices": {},
                "matrix_metadata": {},
                "classification_metrics": {},
                "conversion_summary": {},
                "metadata": {
                    "conversion_timestamp": datetime.now().isoformat(),
                    "conversion_options": self.conversion_options.copy()
                }
            }
            
            # Extract processed tables
            processed_tables = processed_data.get("processed_tables", {})
            if not processed_tables:
                raise ValueError("No processed tables found in input data")
            
            # Convert each table
            for table_name, table_data in processed_tables.items():
                self.logger.info(f"Converting table: {table_name}")
                
                if table_data["processed_data"] is None:
                    self.logger.warning(f"Skipping table '{table_name}' - no processed data")
                    continue
                
                # Convert individual table
                matrix_result = self._convert_individual_table(table_data, table_name)
                
                # Store results
                conversion_result["matrices"][table_name] = matrix_result["matrix"]
                conversion_result["matrix_metadata"][table_name] = matrix_result["metadata"]
                conversion_result["classification_metrics"][table_name] = matrix_result["metrics"]
                
                # Generate conversion summary
                summary = self._generate_conversion_summary(matrix_result, table_name)
                conversion_result["conversion_summary"][table_name] = summary
            
            self.logger.info("Confusion matrix conversion completed successfully")
            return conversion_result
            
        except Exception as e:
            self.logger.error(f"Confusion matrix conversion failed: {str(e)}")
            raise
    
    def _convert_individual_table(self, table_data: Dict[str, Any], table_name: str) -> Dict[str, Any]:
        """Convert an individual contingency table to confusion matrix."""
        try:
            processed_data = table_data["processed_data"]
            
            # Initialize result structure
            matrix_result = {
                "matrix": None,
                "metadata": {
                    "original_shape": processed_data.shape,
                    "matrix_type": "confusion_matrix",
                    "class_labels": [],
                    "conversion_method": "direct",
                    "warnings": []
                },
                "metrics": {}
            }
            
            # Step 1: Determine matrix conversion approach
            conversion_approach = self._determine_conversion_approach(processed_data)
            matrix_result["metadata"]["conversion_method"] = conversion_approach
            
            # Step 2: Convert based on approach
            if conversion_approach == "direct_square":
                confusion_matrix = self._convert_direct_square(processed_data, matrix_result)
            elif conversion_approach == "direct_rectangular":
                confusion_matrix = self._convert_direct_rectangular(processed_data, matrix_result)
            elif conversion_approach == "binary_classification":
                confusion_matrix = self._convert_binary_classification(processed_data, matrix_result)
            elif conversion_approach == "multiclass_classification":
                confusion_matrix = self._convert_multiclass_classification(processed_data, matrix_result)
            else:
                raise ValueError(f"Unknown conversion approach: {conversion_approach}")
            
            # Step 3: Apply normalization if requested
            if self.conversion_options["normalization_method"] != "none":
                confusion_matrix = self._apply_normalization(confusion_matrix, matrix_result)
            
            # Step 4: Add marginals if requested
            if self.conversion_options["include_marginals"]:
                confusion_matrix = self._add_marginals(confusion_matrix, matrix_result)
            
            # Step 5: Calculate classification metrics
            metrics = self._calculate_classification_metrics(confusion_matrix, matrix_result)
            matrix_result["metrics"] = metrics
            
            # Step 6: Store final matrix
            matrix_result["matrix"] = confusion_matrix
            
            return matrix_result
            
        except Exception as e:
            self.logger.error(f"Error converting table '{table_name}': {str(e)}")
            raise
    
    def _determine_conversion_approach(self, data: pd.DataFrame) -> str:
        """Determine the best approach for converting to confusion matrix."""
        try:
            rows, cols = data.shape
            
            # Check if it's already a square matrix (likely confusion matrix)
            if rows == cols:
                return "direct_square"
            
            # Check if it's a rectangular contingency table
            elif rows != cols:
                return "direct_rectangular"
            
            # Check if it's suitable for binary classification
            elif rows == 2 and cols == 2:
                return "binary_classification"
            
            # Default to multiclass classification
            else:
                return "multiclass_classification"
                
        except Exception as e:
            self.logger.error(f"Error determining conversion approach: {str(e)}")
            return "direct_square"
    
    def _convert_direct_square(self, data: pd.DataFrame, matrix_result: Dict[str, Any]) -> pd.DataFrame:
        """Convert square contingency table directly to confusion matrix."""
        try:
            # Assume it's already in confusion matrix format
            confusion_matrix = data.copy()
            
            # Set class labels
            matrix_result["metadata"]["class_labels"] = list(data.index)
            
            # Ensure proper ordering
            if self.conversion_options["class_ordering"] == "alphabetical":
                confusion_matrix = confusion_matrix.sort_index().sort_index(axis=1)
            elif self.conversion_options["class_ordering"] == "frequency":
                # Order by frequency (diagonal elements)
                diagonal_values = np.diag(confusion_matrix.values)
                order = np.argsort(diagonal_values)[::-1]
                confusion_matrix = confusion_matrix.iloc[order, order]
            
            matrix_result["metadata"]["conversion_method"] = "direct_square"
            return confusion_matrix
            
        except Exception as e:
            self.logger.error(f"Error in direct square conversion: {str(e)}")
            raise
    
    def _convert_direct_rectangular(self, data: pd.DataFrame, matrix_result: Dict[str, Any]) -> pd.DataFrame:
        """Convert rectangular contingency table to confusion matrix."""
        try:
            # For rectangular tables, we need to determine predicted vs actual
            # Assume rows are actual classes, columns are predicted classes
            
            confusion_matrix = data.copy()
            
            # Set class labels
            matrix_result["metadata"]["class_labels"] = {
                "actual": list(data.index),
                "predicted": list(data.columns)
            }
            
            # Add warning about rectangular matrix
            matrix_result["metadata"]["warnings"].append(
                "Rectangular matrix converted directly - verify actual vs predicted orientation"
            )
            
            matrix_result["metadata"]["conversion_method"] = "direct_rectangular"
            return confusion_matrix
            
        except Exception as e:
            self.logger.error(f"Error in direct rectangular conversion: {str(e)}")
            raise
    
    def _convert_binary_classification(self, data: pd.DataFrame, matrix_result: Dict[str, Any]) -> pd.DataFrame:
        """Convert 2x2 table to binary confusion matrix."""
        try:
            if data.shape != (2, 2):
                raise ValueError("Binary classification requires 2x2 matrix")
            
            # Standard binary confusion matrix layout:
            # [[TN, FP],
            #  [FN, TP]]
            
            confusion_matrix = data.copy()
            
            # Set standard binary labels
            binary_labels = ["Negative", "Positive"]
            confusion_matrix.index = binary_labels
            confusion_matrix.columns = binary_labels
            
            matrix_result["metadata"]["class_labels"] = binary_labels
            matrix_result["metadata"]["matrix_layout"] = {
                "TN": (0, 0), "FP": (0, 1),
                "FN": (1, 0), "TP": (1, 1)
            }
            
            matrix_result["metadata"]["conversion_method"] = "binary_classification"
            return confusion_matrix
            
        except Exception as e:
            self.logger.error(f"Error in binary classification conversion: {str(e)}")
            raise
    
    def _convert_multiclass_classification(self, data: pd.DataFrame, matrix_result: Dict[str, Any]) -> pd.DataFrame:
        """Convert multiclass contingency table to confusion matrix."""
        try:
            # For multiclass, ensure square matrix
            if data.shape[0] != data.shape[1]:
                # Pad with zeros to make square
                max_size = max(data.shape)
                square_data = pd.DataFrame(
                    np.zeros((max_size, max_size)),
                    index=range(max_size),
                    columns=range(max_size)
                )
                square_data.iloc[:data.shape[0], :data.shape[1]] = data.values
                confusion_matrix = square_data
                
                matrix_result["metadata"]["warnings"].append(
                    "Non-square matrix padded with zeros for multiclass conversion"
                )
            else:
                confusion_matrix = data.copy()
            
            # Set class labels
            matrix_result["metadata"]["class_labels"] = list(confusion_matrix.index)
            
            matrix_result["metadata"]["conversion_method"] = "multiclass_classification"
            return confusion_matrix
            
        except Exception as e:
            self.logger.error(f"Error in multiclass classification conversion: {str(e)}")
            raise
    
    def _apply_normalization(self, matrix: pd.DataFrame, matrix_result: Dict[str, Any]) -> pd.DataFrame:
        """Apply normalization to confusion matrix."""
        try:
            normalization_method = self.conversion_options["normalization_method"]
            
            if normalization_method == "true":
                # Normalize by true labels (rows)
                row_sums = matrix.sum(axis=1)
                normalized_matrix = matrix.div(row_sums, axis=0)
            elif normalization_method == "pred":
                # Normalize by predicted labels (columns)
                col_sums = matrix.sum(axis=0)
                normalized_matrix = matrix.div(col_sums, axis=1)
            elif normalization_method == "all":
                # Normalize by total sum
                total_sum = matrix.sum().sum()
                normalized_matrix = matrix / total_sum
            else:
                return matrix
            
            # Handle division by zero
            normalized_matrix = normalized_matrix.fillna(0)
            
            matrix_result["metadata"]["normalization_applied"] = normalization_method
            matrix_result["metadata"]["warnings"].append(
                f"Matrix normalized using method: {normalization_method}"
            )
            
            return normalized_matrix
            
        except Exception as e:
            self.logger.error(f"Error applying normalization: {str(e)}")
            return matrix
    
    def _add_marginals(self, matrix: pd.DataFrame, matrix_result: Dict[str, Any]) -> pd.DataFrame:
        """Add marginal totals to confusion matrix."""
        try:
            # Add row and column totals
            matrix_with_marginals = matrix.copy()
            
            # Add row totals
            matrix_with_marginals["Row_Total"] = matrix_with_marginals.sum(axis=1)
            
            # Add column totals
            col_totals = matrix_with_marginals.sum(axis=0)
            matrix_with_marginals.loc["Col_Total"] = col_totals
            
            matrix_result["metadata"]["marginals_added"] = True
            return matrix_with_marginals
            
        except Exception as e:
            self.logger.error(f"Error adding marginals: {str(e)}")
            return matrix
    
    def _calculate_classification_metrics(self, matrix: pd.DataFrame, matrix_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate classification metrics from confusion matrix."""
        try:
            metrics = {
                "basic_metrics": {},
                "advanced_metrics": {},
                "per_class_metrics": {}
            }
            
            # Remove marginals if present for calculations
            calc_matrix = matrix.copy()
            if "Row_Total" in calc_matrix.columns:
                calc_matrix = calc_matrix.drop("Row_Total", axis=1)
            if "Col_Total" in calc_matrix.index:
                calc_matrix = calc_matrix.drop("Col_Total", axis=0)
            
            # Convert to numpy array for calculations
            matrix_array = calc_matrix.values
            
            # Basic metrics
            total_samples = np.sum(matrix_array)
            correct_predictions = np.trace(matrix_array)
            
            metrics["basic_metrics"]["total_samples"] = total_samples
            metrics["basic_metrics"]["correct_predictions"] = correct_predictions
            metrics["basic_metrics"]["accuracy"] = correct_predictions / total_samples if total_samples > 0 else 0
            
            # Binary classification metrics (if 2x2 matrix)
            if matrix_array.shape == (2, 2):
                metrics["binary_metrics"] = self._calculate_binary_metrics(matrix_array)
            
            # Multiclass metrics - only for matrices with more than 2 classes
            if matrix_array.shape[0] > 2 and matrix_array.shape[1] > 2:
                metrics["multiclass_metrics"] = self._calculate_multiclass_metrics(matrix_array)
            
            # Per-class metrics - only for classes that exist in the matrix
            num_classes = matrix_array.shape[0]
            for i, class_label in enumerate(calc_matrix.index):
                if i < num_classes:  # Only process classes that exist in the matrix
                    class_metrics = self._calculate_per_class_metrics(matrix_array, i)
                    metrics["per_class_metrics"][class_label] = class_metrics
                else:
                    # Skip classes that don't exist in the matrix
                    self.logger.warning(f"Skipping class index {i} - matrix only has {num_classes} classes")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating classification metrics: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_binary_metrics(self, matrix: np.ndarray) -> Dict[str, float]:
        """Calculate binary classification metrics."""
        try:
            # Standard binary confusion matrix layout:
            # [[TN, FP],
            #  [FN, TP]]
            tn, fp, fn, tp = matrix.ravel()
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            return {
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn,
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "f1_score": f1_score
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating binary metrics: {str(e)}")
            return {}
    
    def _calculate_multiclass_metrics(self, matrix: np.ndarray) -> Dict[str, float]:
        """Calculate multiclass classification metrics."""
        try:
            n_classes = matrix.shape[0]
            
            # Macro-averaged metrics
            precisions = []
            recalls = []
            f1_scores = []
            
            for i in range(n_classes):
                tp = matrix[i, i]
                fp = np.sum(matrix[:, i]) - tp
                fn = np.sum(matrix[i, :]) - tp
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)
            
            return {
                "macro_precision": np.mean(precisions),
                "macro_recall": np.mean(recalls),
                "macro_f1": np.mean(f1_scores),
                "num_classes": n_classes
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating multiclass metrics: {str(e)}")
            return {}
    
    def _calculate_per_class_metrics(self, matrix: np.ndarray, class_index: int) -> Dict[str, float]:
        """Calculate per-class metrics."""
        try:
            tp = matrix[class_index, class_index]
            fp = np.sum(matrix[:, class_index]) - tp
            fn = np.sum(matrix[class_index, :]) - tp
            tn = np.sum(matrix) - tp - fp - fn
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            return {
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "f1_score": f1_score,
                "support": tp + fn
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating per-class metrics: {str(e)}")
            return {}
    
    def _generate_conversion_summary(self, matrix_result: Dict[str, Any], table_name: str) -> Dict[str, Any]:
        """Generate summary for matrix conversion."""
        try:
            summary = {
                "table_name": table_name,
                "conversion_successful": matrix_result["matrix"] is not None,
                "matrix_shape": matrix_result["matrix"].shape if matrix_result["matrix"] is not None else None,
                "conversion_method": matrix_result["metadata"]["conversion_method"],
                "warnings_count": len(matrix_result["metadata"]["warnings"]),
                "warnings": matrix_result["metadata"]["warnings"],
                "has_classification_metrics": bool(matrix_result["metrics"])
            }
            
            if matrix_result["metrics"]:
                basic_metrics = matrix_result["metrics"].get("basic_metrics", {})
                summary.update({
                    "total_samples": basic_metrics.get("total_samples", 0),
                    "accuracy": basic_metrics.get("accuracy", 0),
                    "matrix_type": "binary" if matrix_result["matrix"].shape == (2, 2) else "multiclass"
                })
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating conversion summary: {str(e)}")
            return {"error": str(e)}
    
    def get_conversion_options(self) -> Dict[str, Any]:
        """Get current conversion options."""
        return self.conversion_options.copy()
    
    def set_conversion_options(self, options: Dict[str, Any]) -> None:
        """Set conversion options."""
        self.conversion_options.update(options)
        self.logger.info(f"Conversion options updated: {options}")
    
    def validate_confusion_matrix(self, matrix: pd.DataFrame) -> Dict[str, Any]:
        """Validate a confusion matrix."""
        try:
            validation_result = {
                "valid": True,
                "errors": [],
                "warnings": []
            }
            
            # Check if matrix is square
            if matrix.shape[0] != matrix.shape[1]:
                validation_result["warnings"].append("Matrix is not square")
            
            # Check for negative values
            if (matrix < 0).any().any():
                validation_result["errors"].append("Matrix contains negative values")
                validation_result["valid"] = False
            
            # Check for NaN values
            if matrix.isnull().any().any():
                validation_result["errors"].append("Matrix contains NaN values")
                validation_result["valid"] = False
            
            # Check for zero diagonal
            if matrix.shape[0] == matrix.shape[1]:
                diagonal_sum = np.trace(matrix.values)
                if diagonal_sum == 0:
                    validation_result["warnings"].append("Matrix has zero diagonal sum")
            
            return validation_result
            
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Validation failed: {str(e)}"],
                "warnings": []
            }
