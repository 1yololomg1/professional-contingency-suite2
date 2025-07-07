#!/usr/bin/env python3
"""
Matrix Validation Module
Professional Contingency Analysis Suite
Validates confusion matrices after conversion from contingency tables
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from datetime import datetime


class MatrixValidator:
    """
    Validates confusion matrices for structural integrity and statistical validity.
    Ensures matrices are properly formatted for downstream analysis.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.matrix_rules = {
            "require_square_matrices": True,
            "min_matrix_size": 2,
            "max_matrix_size": 100,
            "allow_zero_diagonal": False,
            "min_diagonal_sum": 1,
            "max_condition_number": 1e12,
            "min_determinant": 1e-10,
            "require_positive_definite": False
        }
    
    def validate_confusion_matrices(self, confusion_matrices: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive validation of confusion matrices.
        
        Args:
            confusion_matrices: Dictionary containing confusion matrix data
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "matrix_summary": {},
            "validation_timestamp": datetime.now().isoformat()
        }
        
        try:
            # Validate matrix structure
            structure_validation = self._validate_matrix_structure(confusion_matrices)
            validation_result["errors"].extend(structure_validation["errors"])
            validation_result["warnings"].extend(structure_validation["warnings"])
            
            # Validate matrix properties
            properties_validation = self._validate_matrix_properties(confusion_matrices)
            validation_result["errors"].extend(properties_validation["errors"])
            validation_result["warnings"].extend(properties_validation["warnings"])
            
            # Validate numerical stability
            numerical_validation = self._validate_numerical_stability(confusion_matrices)
            validation_result["errors"].extend(numerical_validation["errors"])
            validation_result["warnings"].extend(numerical_validation["warnings"])
            
            # Validate classification metrics compatibility
            metrics_validation = self._validate_classification_metrics(confusion_matrices)
            validation_result["errors"].extend(metrics_validation["errors"])
            validation_result["warnings"].extend(metrics_validation["warnings"])
            
            # Generate matrix summary
            validation_result["matrix_summary"] = self._generate_matrix_summary(confusion_matrices)
            
            # Set final validation status
            validation_result["valid"] = len(validation_result["errors"]) == 0
            
            if validation_result["valid"]:
                self.logger.info("Matrix validation passed successfully")
            else:
                self.logger.error(f"Matrix validation failed with {len(validation_result['errors'])} errors")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Matrix validation error: {str(e)}")
            validation_result["valid"] = False
            validation_result["errors"].append(f"Matrix validation process failed: {str(e)}")
            return validation_result
    
    def _validate_matrix_structure(self, confusion_matrices: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate the basic structure of confusion matrices."""
        errors = []
        warnings = []
        
        try:
            # Check if matrices data exists
            if not confusion_matrices or not isinstance(confusion_matrices, dict):
                errors.append("Confusion matrices data is empty or not in expected format")
                return {"errors": errors, "warnings": warnings}
            
            # Check for matrices key
            if "matrices" not in confusion_matrices:
                errors.append("No matrices found in confusion matrices data")
                return {"errors": errors, "warnings": warnings}
            
            matrices = confusion_matrices["matrices"]
            if not matrices:
                errors.append("Matrices dictionary is empty")
                return {"errors": errors, "warnings": warnings}
            
            # Validate each matrix
            for matrix_name, matrix_data in matrices.items():
                if not isinstance(matrix_data, (pd.DataFrame, np.ndarray)):
                    errors.append(f"Matrix '{matrix_name}' is not in DataFrame or ndarray format")
                    continue
                
                # Convert to numpy array for validation
                if isinstance(matrix_data, pd.DataFrame):
                    matrix_array = matrix_data.values
                else:
                    matrix_array = matrix_data
                
                # Check dimensions
                if matrix_array.ndim != 2:
                    errors.append(f"Matrix '{matrix_name}' is not 2-dimensional")
                    continue
                
                # Check minimum size
                if matrix_array.shape[0] < self.matrix_rules["min_matrix_size"] or \
                   matrix_array.shape[1] < self.matrix_rules["min_matrix_size"]:
                    errors.append(f"Matrix '{matrix_name}' is too small: {matrix_array.shape}")
                
                # Check maximum size
                if matrix_array.shape[0] > self.matrix_rules["max_matrix_size"] or \
                   matrix_array.shape[1] > self.matrix_rules["max_matrix_size"]:
                    warnings.append(f"Matrix '{matrix_name}' is very large: {matrix_array.shape}")
                
                # Check if square (if required)
                if self.matrix_rules["require_square_matrices"]:
                    if matrix_array.shape[0] != matrix_array.shape[1]:
                        errors.append(f"Matrix '{matrix_name}' is not square: {matrix_array.shape}")
                
                # Check for empty matrix
                if matrix_array.size == 0:
                    errors.append(f"Matrix '{matrix_name}' is empty")
                
                # Check for proper data types
                if not np.issubdtype(matrix_array.dtype, np.number):
                    errors.append(f"Matrix '{matrix_name}' contains non-numeric data")
        
        except Exception as e:
            errors.append(f"Matrix structure validation error: {str(e)}")
        
        return {"errors": errors, "warnings": warnings}
    
    def _validate_matrix_properties(self, confusion_matrices: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate mathematical properties of confusion matrices."""
        errors = []
        warnings = []
        
        try:
            matrices = confusion_matrices.get("matrices", {})
            
            for matrix_name, matrix_data in matrices.items():
                if not isinstance(matrix_data, (pd.DataFrame, np.ndarray)):
                    continue
                
                # Convert to numpy array
                if isinstance(matrix_data, pd.DataFrame):
                    matrix_array = matrix_data.values
                else:
                    matrix_array = matrix_data
                
                # Check for negative values
                if np.any(matrix_array < 0):
                    errors.append(f"Matrix '{matrix_name}' contains negative values")
                
                # Check for NaN or infinite values
                if np.any(np.isnan(matrix_array)):
                    errors.append(f"Matrix '{matrix_name}' contains NaN values")
                
                if np.any(np.isinf(matrix_array)):
                    errors.append(f"Matrix '{matrix_name}' contains infinite values")
                
                # Check diagonal properties (for square matrices)
                if matrix_array.shape[0] == matrix_array.shape[1]:
                    diagonal = np.diag(matrix_array)
                    
                    # Check for zero diagonal elements
                    if not self.matrix_rules["allow_zero_diagonal"] and np.any(diagonal == 0):
                        warnings.append(f"Matrix '{matrix_name}' has zero diagonal elements")
                    
                    # Check diagonal sum
                    diagonal_sum = np.sum(diagonal)
                    if diagonal_sum < self.matrix_rules["min_diagonal_sum"]:
                        errors.append(f"Matrix '{matrix_name}' has insufficient diagonal sum: {diagonal_sum}")
                
                # Check for all-zero rows or columns
                zero_rows = np.all(matrix_array == 0, axis=1)
                zero_cols = np.all(matrix_array == 0, axis=0)
                
                if np.any(zero_rows):
                    zero_row_indices = np.where(zero_rows)[0]
                    warnings.append(f"Matrix '{matrix_name}' has all-zero rows: {zero_row_indices.tolist()}")
                
                if np.any(zero_cols):
                    zero_col_indices = np.where(zero_cols)[0]
                    warnings.append(f"Matrix '{matrix_name}' has all-zero columns: {zero_col_indices.tolist()}")
                
                # Check for extremely sparse matrices
                zero_percentage = (matrix_array == 0).sum() / matrix_array.size * 100
                if zero_percentage > 80:
                    warnings.append(f"Matrix '{matrix_name}' is very sparse: {zero_percentage:.1f}% zeros")
        
        except Exception as e:
            errors.append(f"Matrix properties validation error: {str(e)}")
        
        return {"errors": errors, "warnings": warnings}
    
    def _validate_numerical_stability(self, confusion_matrices: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate numerical stability of matrices."""
        errors = []
        warnings = []
        
        try:
            matrices = confusion_matrices.get("matrices", {})
            
            for matrix_name, matrix_data in matrices.items():
                if not isinstance(matrix_data, (pd.DataFrame, np.ndarray)):
                    continue
                
                # Convert to numpy array
                if isinstance(matrix_data, pd.DataFrame):
                    matrix_array = matrix_data.values
                else:
                    matrix_array = matrix_data
                
                # Skip non-square matrices for some tests
                if matrix_array.shape[0] != matrix_array.shape[1]:
                    continue
                
                try:
                    # Check condition number
                    cond_num = np.linalg.cond(matrix_array)
                    if cond_num > self.matrix_rules["max_condition_number"]:
                        warnings.append(f"Matrix '{matrix_name}' has high condition number: {cond_num:.2e}")
                    
                    # Check determinant (for square matrices)
                    det = np.linalg.det(matrix_array)
                    if abs(det) < self.matrix_rules["min_determinant"]:
                        warnings.append(f"Matrix '{matrix_name}' has very small determinant: {det:.2e}")
                    
                    # Check for positive definiteness (if required)
                    if self.matrix_rules["require_positive_definite"]:
                        eigenvalues = np.linalg.eigvals(matrix_array)
                        if np.any(eigenvalues <= 0):
                            errors.append(f"Matrix '{matrix_name}' is not positive definite")
                
                except np.linalg.LinAlgError as e:
                    warnings.append(f"Matrix '{matrix_name}' numerical analysis failed: {str(e)}")
                
                # Check for extreme values
                max_val = np.max(matrix_array)
                min_val = np.min(matrix_array)
                
                if max_val > 1e10:
                    warnings.append(f"Matrix '{matrix_name}' has very large values (max: {max_val:.2e})")
                
                if min_val > 0 and min_val < 1e-10:
                    warnings.append(f"Matrix '{matrix_name}' has very small non-zero values (min: {min_val:.2e})")
        
        except Exception as e:
            errors.append(f"Numerical stability validation error: {str(e)}")
        
        return {"errors": errors, "warnings": warnings}
    
    def _validate_classification_metrics(self, confusion_matrices: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate compatibility with classification metrics computation."""
        errors = []
        warnings = []
        
        try:
            matrices = confusion_matrices.get("matrices", {})
            
            for matrix_name, matrix_data in matrices.items():
                if not isinstance(matrix_data, (pd.DataFrame, np.ndarray)):
                    continue
                
                # Convert to numpy array
                if isinstance(matrix_data, pd.DataFrame):
                    matrix_array = matrix_data.values
                else:
                    matrix_array = matrix_data
                
                # Check total observations
                total_obs = np.sum(matrix_array)
                if total_obs == 0:
                    errors.append(f"Matrix '{matrix_name}' has zero total observations")
                    continue
                
                # Check for binary classification compatibility
                if matrix_array.shape == (2, 2):
                    # Standard 2x2 confusion matrix checks
                    tp = matrix_array[0, 0]  # True Positives
                    fp = matrix_array[0, 1]  # False Positives
                    fn = matrix_array[1, 0]  # False Negatives
                    tn = matrix_array[1, 1]  # True Negatives
                    
                    # Check for zero classes
                    if tp + fn == 0:
                        warnings.append(f"Matrix '{matrix_name}' has no positive class examples")
                    if fp + tn == 0:
                        warnings.append(f"Matrix '{matrix_name}' has no negative class examples")
                    
                    # Check for perfect classification
                    if fp == 0 and fn == 0:
                        warnings.append(f"Matrix '{matrix_name}' represents perfect classification")
                    
                    # Check for random classification
                    expected_accuracy = ((tp + fn) * (tp + fp) + (fp + tn) * (fn + tn)) / (total_obs ** 2)
                    actual_accuracy = (tp + tn) / total_obs
                    
                    if abs(actual_accuracy - expected_accuracy) < 0.01:
                        warnings.append(f"Matrix '{matrix_name}' shows near-random classification performance")
                
                # Check for multi-class compatibility
                elif matrix_array.shape[0] == matrix_array.shape[1] and matrix_array.shape[0] > 2:
                    # Multi-class confusion matrix checks
                    diagonal_sum = np.trace(matrix_array)
                    total_sum = np.sum(matrix_array)
                    
                    if diagonal_sum == 0:
                        errors.append(f"Matrix '{matrix_name}' has zero diagonal sum (no correct predictions)")
                    
                    # Check for class imbalance
                    row_sums = np.sum(matrix_array, axis=1)
                    col_sums = np.sum(matrix_array, axis=0)
                    
                    if len(row_sums) > 1:
                        row_cv = np.std(row_sums) / np.mean(row_sums)
                        if row_cv > 2.0:
                            warnings.append(f"Matrix '{matrix_name}' shows severe class imbalance in true labels (CV: {row_cv:.2f})")
                    
                    if len(col_sums) > 1:
                        col_cv = np.std(col_sums) / np.mean(col_sums)
                        if col_cv > 2.0:
                            warnings.append(f"Matrix '{matrix_name}' shows severe class imbalance in predictions (CV: {col_cv:.2f})")
        
        except Exception as e:
            errors.append(f"Classification metrics validation error: {str(e)}")
        
        return {"errors": errors, "warnings": warnings}
    
    def _generate_matrix_summary(self, confusion_matrices: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive summary of the matrices."""
        summary = {
            "total_matrices": 0,
            "valid_matrices": 0,
            "matrix_details": {}
        }
        
        try:
            matrices = confusion_matrices.get("matrices", {})
            summary["total_matrices"] = len(matrices)
            
            for matrix_name, matrix_data in matrices.items():
                if not isinstance(matrix_data, (pd.DataFrame, np.ndarray)):
                    continue
                
                summary["valid_matrices"] += 1
                
                # Convert to numpy array
                if isinstance(matrix_data, pd.DataFrame):
                    matrix_array = matrix_data.values
                else:
                    matrix_array = matrix_data
                
                matrix_summary = {
                    "shape": matrix_array.shape,
                    "total_observations": np.sum(matrix_array),
                    "non_zero_elements": np.count_nonzero(matrix_array),
                    "sparsity": (matrix_array == 0).sum() / matrix_array.size * 100,
                    "min_value": np.min(matrix_array),
                    "max_value": np.max(matrix_array),
                    "mean_value": np.mean(matrix_array),
                    "std_value": np.std(matrix_array)
                }
                
                # Add diagonal statistics for square matrices
                if matrix_array.shape[0] == matrix_array.shape[1]:
                    diagonal = np.diag(matrix_array)
                    matrix_summary.update({
                        "diagonal_sum": np.sum(diagonal),
                        "diagonal_mean": np.mean(diagonal),
                        "trace": np.trace(matrix_array),
                        "is_symmetric": np.allclose(matrix_array, matrix_array.T)
                    })
                    
                    # Add determinant and condition number for small matrices
                    if matrix_array.shape[0] <= 10:
                        try:
                            matrix_summary["determinant"] = np.linalg.det(matrix_array)
                            matrix_summary["condition_number"] = np.linalg.cond(matrix_array)
                        except np.linalg.LinAlgError:
                            matrix_summary["determinant"] = None
                            matrix_summary["condition_number"] = None
                
                # Add classification metrics for 2x2 matrices
                if matrix_array.shape == (2, 2):
                    tp, fp, fn, tn = matrix_array[0, 0], matrix_array[0, 1], matrix_array[1, 0], matrix_array[1, 1]
                    total = tp + fp + fn + tn
                    
                    if total > 0:
                        matrix_summary.update({
                            "accuracy": (tp + tn) / total,
                            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
                            "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
                            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0
                        })
                
                summary["matrix_details"][matrix_name] = matrix_summary
        
        except Exception as e:
            summary["error"] = f"Matrix summary generation error: {str(e)}"
        
        return summary
    
    def set_matrix_rules(self, rules: Dict[str, Any]) -> None:
        """Update matrix validation rules."""
        self.matrix_rules.update(rules)
        self.logger.info(f"Matrix validation rules updated: {rules}")
    
    def get_matrix_rules(self) -> Dict[str, Any]:
        """Get current matrix validation rules."""
        return self.matrix_rules.copy()
