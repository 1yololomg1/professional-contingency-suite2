#!/usr/bin/env python3
"""
Cramér's V Calculator Module
Professional Contingency Analysis Suite
Calculates Cramér's V coefficient for measuring association between categorical variables
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from datetime import datetime
from scipy import stats
from scipy.stats import chi2_contingency
import warnings


class CramersVCalculator:
    """
    Calculates Cramér's V coefficient and related association measures.
    Provides comprehensive analysis of categorical variable associations.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.calculation_options = {
            "bias_correction": True,
            "confidence_intervals": True,
            "bootstrap_samples": 1000,
            "confidence_level": 0.95,
            "include_effect_size_interpretation": True,
            "calculate_phi": True,
            "calculate_contingency_coefficient": True,
            "calculate_tschuprow_t": True,
            "small_sample_correction": True,
            "min_expected_frequency": 5,
            "use_continuity_correction": False
        }
        
        # Effect size interpretation thresholds
        self.effect_size_thresholds = {
            "negligible": 0.1,
            "small": 0.2,
            "medium": 0.4,
            "large": 0.6,
            "very_large": 0.8
        }
    
    def calculate_cramers_v(self, confusion_matrices: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate Cramér's V for all confusion matrices.
        
        Args:
            confusion_matrices: Dictionary containing confusion matrices
            
        Returns:
            Dictionary with Cramér's V results and related statistics
        """
        try:
            self.logger.info("Starting Cramér's V calculation")
            
            calculation_result = {
                "values": [],
                "detailed_results": {},
                "confidence_intervals": [],
                "effect_size_interpretations": [],
                "related_measures": {},
                "statistical_significance": {},
                "metadata": {
                    "calculation_timestamp": datetime.now().isoformat(),
                    "calculation_options": self.calculation_options.copy()
                }
            }
            
            # Extract matrices
            matrices = confusion_matrices.get("matrices", {})
            if not matrices:
                raise ValueError("No matrices found in confusion matrices data")
            
            # Calculate Cramér's V for each matrix
            for matrix_name, matrix_data in matrices.items():
                self.logger.info(f"Calculating Cramér's V for matrix: {matrix_name}")
                
                # Calculate for individual matrix
                matrix_result = self._calculate_individual_cramers_v(matrix_data, matrix_name)
                
                # Store results
                calculation_result["values"].append(matrix_result["cramers_v"])
                calculation_result["detailed_results"][matrix_name] = matrix_result
                
                if matrix_result["confidence_interval"]:
                    calculation_result["confidence_intervals"].append(matrix_result["confidence_interval"])
                
                calculation_result["effect_size_interpretations"].append(matrix_result["effect_size_interpretation"])
                calculation_result["related_measures"][matrix_name] = matrix_result["related_measures"]
                calculation_result["statistical_significance"][matrix_name] = matrix_result["statistical_significance"]
            
            # Generate summary statistics
            calculation_result["summary_statistics"] = self._generate_summary_statistics(calculation_result)
            
            self.logger.info("Cramér's V calculation completed successfully")
            return calculation_result
            
        except Exception as e:
            self.logger.error(f"Cramér's V calculation failed: {str(e)}")
            raise
    
    def _calculate_individual_cramers_v(self, matrix_data: pd.DataFrame, matrix_name: str) -> Dict[str, Any]:
        """Calculate Cramér's V for an individual matrix."""
        try:
            # Initialize result structure
            result = {
                "cramers_v": 0.0,
                "cramers_v_corrected": 0.0,
                "confidence_interval": None,
                "effect_size_interpretation": "negligible",
                "related_measures": {},
                "statistical_significance": {},
                "calculation_details": {},
                "warnings": []
            }
            
            # Prepare matrix for calculation
            calc_matrix = self._prepare_matrix_for_calculation(matrix_data)
            
            # Validate matrix
            validation_result = self._validate_matrix_for_calculation(calc_matrix)
            if not validation_result["valid"]:
                result["warnings"].extend(validation_result["warnings"])
                if validation_result["errors"]:
                    result["calculation_details"]["errors"] = validation_result["errors"]
                    return result
            
            # Calculate basic Cramér's V
            cramers_v = self._calculate_basic_cramers_v(calc_matrix)
            result["cramers_v"] = cramers_v
            
            # Apply bias correction if requested
            if self.calculation_options["bias_correction"]:
                cramers_v_corrected = self._apply_bias_correction(cramers_v, calc_matrix)
                result["cramers_v_corrected"] = cramers_v_corrected
            else:
                result["cramers_v_corrected"] = cramers_v
            
            # Calculate confidence intervals
            if self.calculation_options["confidence_intervals"]:
                result["confidence_interval"] = self._calculate_confidence_interval(calc_matrix)
            
            # Determine effect size interpretation
            if self.calculation_options["include_effect_size_interpretation"]:
                result["effect_size_interpretation"] = self._interpret_effect_size(result["cramers_v_corrected"])
            
            # Calculate related measures
            result["related_measures"] = self._calculate_related_measures(calc_matrix)
            
            # Calculate statistical significance
            result["statistical_significance"] = self._calculate_statistical_significance(calc_matrix)
            
            # Store calculation details
            result["calculation_details"] = self._get_calculation_details(calc_matrix)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating Cramér's V for matrix '{matrix_name}': {str(e)}")
            raise
    
    def _prepare_matrix_for_calculation(self, matrix_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare matrix for Cramér's V calculation."""
        try:
            # Remove marginals if present
            calc_matrix = matrix_data.copy()
            
            if "Row_Total" in calc_matrix.columns:
                calc_matrix = calc_matrix.drop("Row_Total", axis=1)
            if "Col_Total" in calc_matrix.index:
                calc_matrix = calc_matrix.drop("Col_Total", axis=0)
            
            # Ensure all values are numeric and non-negative
            calc_matrix = calc_matrix.apply(pd.to_numeric, errors='coerce')
            calc_matrix = calc_matrix.fillna(0)
            calc_matrix = calc_matrix.clip(lower=0)
            
            # Convert to integers if they're close to integers
            if np.allclose(calc_matrix.values, calc_matrix.values.astype(int)):
                calc_matrix = calc_matrix.astype(int)
            
            return calc_matrix
            
        except Exception as e:
            self.logger.error(f"Error preparing matrix for calculation: {str(e)}")
            raise
    
    def _validate_matrix_for_calculation(self, matrix: pd.DataFrame) -> Dict[str, Any]:
        """Validate matrix for Cramér's V calculation."""
        try:
            validation_result = {
                "valid": True,
                "errors": [],
                "warnings": []
            }
            
            # Check minimum dimensions
            if matrix.shape[0] < 2 or matrix.shape[1] < 2:
                validation_result["errors"].append("Matrix must be at least 2x2 for Cramér's V calculation")
                validation_result["valid"] = False
            
            # Check for zero total
            total_sum = matrix.sum().sum()
            if total_sum == 0:
                validation_result["errors"].append("Matrix has zero total sum")
                validation_result["valid"] = False
            
            # Check for zero rows or columns
            zero_rows = (matrix.sum(axis=1) == 0).sum()
            zero_cols = (matrix.sum(axis=0) == 0).sum()
            
            if zero_rows > 0:
                validation_result["warnings"].append(f"Matrix has {zero_rows} zero rows")
            if zero_cols > 0:
                validation_result["warnings"].append(f"Matrix has {zero_cols} zero columns")
            
            # Check expected frequencies
            if total_sum > 0:
                row_totals = matrix.sum(axis=1)
                col_totals = matrix.sum(axis=0)
                
                low_expected_count = 0
                for i in range(matrix.shape[0]):
                    for j in range(matrix.shape[1]):
                        if row_totals.iloc[i] > 0 and col_totals.iloc[j] > 0:
                            expected = (row_totals.iloc[i] * col_totals.iloc[j]) / total_sum
                            if expected < self.calculation_options["min_expected_frequency"]:
                                low_expected_count += 1
                
                if low_expected_count > 0:
                    violation_rate = low_expected_count / (matrix.shape[0] * matrix.shape[1])
                    validation_result["warnings"].append(
                        f"Some cells have expected frequency < {self.calculation_options['min_expected_frequency']}: {violation_rate:.2%}"
                    )
            
            return validation_result
            
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Validation failed: {str(e)}"],
                "warnings": []
            }
    
    def _calculate_basic_cramers_v(self, matrix: pd.DataFrame) -> float:
        """Calculate basic Cramér's V coefficient."""
        try:
            matrix_array = matrix.values
            
            # Perform chi-square test
            chi2_stat, p_value, dof, expected = chi2_contingency(matrix_array)
            
            # Calculate Cramér's V
            n = matrix_array.sum()
            r = matrix.shape[0]  # number of rows
            c = matrix.shape[1]  # number of columns
            
            # Cramér's V formula
            cramers_v = np.sqrt(chi2_stat / (n * (min(r, c) - 1)))
            
            # Ensure the result is between 0 and 1
            cramers_v = np.clip(cramers_v, 0, 1)
            
            return cramers_v
            
        except Exception as e:
            self.logger.error(f"Error calculating basic Cramér's V: {str(e)}")
            return 0.0
    
    def _apply_bias_correction(self, cramers_v: float, matrix: pd.DataFrame) -> float:
        """Apply bias correction to Cramér's V."""
        try:
            # Bergsma-Wicher correction for small samples
            n = matrix.sum().sum()
            r = matrix.shape[0]
            c = matrix.shape[1]
            
            if n <= 0:
                return cramers_v
            
            # Apply small sample correction
            if self.calculation_options["small_sample_correction"]:
                # Bergsma-Wicher bias correction
                k = min(r, c)
                
                # Calculate corrected Cramér's V
                cramers_v_squared = cramers_v ** 2
                
                # Bias correction formula
                correction_factor = (k - 1) / (n - 1)
                cramers_v_corrected_squared = max(0, cramers_v_squared - correction_factor)
                
                cramers_v_corrected = np.sqrt(cramers_v_corrected_squared)
                
                return cramers_v_corrected
            
            return cramers_v
            
        except Exception as e:
            self.logger.error(f"Error applying bias correction: {str(e)}")
            return cramers_v
    
    def _calculate_confidence_interval(self, matrix: pd.DataFrame) -> Tuple[float, float]:
        """Calculate confidence interval for Cramér's V using bootstrap."""
        try:
            if not self.calculation_options["confidence_intervals"]:
                return None
            
            matrix_array = matrix.values
            n_samples = self.calculation_options["bootstrap_samples"]
            confidence_level = self.calculation_options["confidence_level"]
            
            # Generate bootstrap samples
            bootstrap_cramers_v = []
            
            for _ in range(n_samples):
                # Resample the contingency table
                # Create multinomial sample based on original proportions
                total_count = matrix_array.sum()
                probs = matrix_array.flatten() / total_count
                
                # Generate multinomial sample
                sample_counts = np.random.multinomial(total_count, probs)
                sample_matrix = sample_counts.reshape(matrix_array.shape)
                
                # Calculate Cramér's V for this sample
                try:
                    chi2_stat, _, _, _ = chi2_contingency(sample_matrix)
                    r, c = sample_matrix.shape
                    cramers_v_sample = np.sqrt(chi2_stat / (total_count * (min(r, c) - 1)))
                    cramers_v_sample = np.clip(cramers_v_sample, 0, 1)
                    bootstrap_cramers_v.append(cramers_v_sample)
                except:
                    # Skip invalid samples
                    continue
            
            if len(bootstrap_cramers_v) < 10:
                return None
            
            # Calculate confidence interval
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            lower_bound = np.percentile(bootstrap_cramers_v, lower_percentile)
            upper_bound = np.percentile(bootstrap_cramers_v, upper_percentile)
            
            return (lower_bound, upper_bound)
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence interval: {str(e)}")
            return None
    
    def _interpret_effect_size(self, cramers_v: float) -> str:
        """Interpret the effect size of Cramér's V."""
        try:
            if cramers_v < self.effect_size_thresholds["negligible"]:
                return "negligible"
            elif cramers_v < self.effect_size_thresholds["small"]:
                return "small"
            elif cramers_v < self.effect_size_thresholds["medium"]:
                return "medium"
            elif cramers_v < self.effect_size_thresholds["large"]:
                return "large"
            elif cramers_v < self.effect_size_thresholds["very_large"]:
                return "very_large"
            else:
                return "extremely_large"
                
        except Exception as e:
            self.logger.error(f"Error interpreting effect size: {str(e)}")
            return "unknown"
    
    def _calculate_related_measures(self, matrix: pd.DataFrame) -> Dict[str, float]:
        """Calculate related association measures."""
        try:
            related_measures = {}
            matrix_array = matrix.values
            
            # Chi-square statistic
            chi2_stat, p_value, dof, expected = chi2_contingency(matrix_array)
            
            n = matrix_array.sum()
            r = matrix.shape[0]
            c = matrix.shape[1]
            
            # Phi coefficient (for 2x2 tables)
            if self.calculation_options["calculate_phi"] and matrix.shape == (2, 2):
                phi = np.sqrt(chi2_stat / n)
                related_measures["phi"] = phi
            
            # Contingency coefficient
            if self.calculation_options["calculate_contingency_coefficient"]:
                contingency_coeff = np.sqrt(chi2_stat / (chi2_stat + n))
                related_measures["contingency_coefficient"] = contingency_coeff
            
            # Tschuprow's T
            if self.calculation_options["calculate_tschuprow_t"]:
                tschuprow_t = np.sqrt(chi2_stat / (n * np.sqrt((r - 1) * (c - 1))))
                related_measures["tschuprow_t"] = tschuprow_t
            
            # Goodman and Kruskal's lambda
            max_row = np.max(matrix_array, axis=1)
            max_col = np.max(matrix_array, axis=0)
            max_total = np.max(matrix_array.sum(axis=1))
            
            lambda_r = (np.sum(max_row) - max_total) / (n - max_total) if (n - max_total) > 0 else 0
            related_measures["lambda_r"] = lambda_r
            
            # Uncertainty coefficient
            p_matrix = matrix_array / n
            
            # Marginal probabilities
            p_row = matrix_array.sum(axis=1) / n
            p_col = matrix_array.sum(axis=0) / n
            
            # Entropy calculations
            h_row = -np.sum(p_row * np.log2(p_row + 1e-10))
            h_col = -np.sum(p_col * np.log2(p_col + 1e-10))
            h_joint = -np.sum(p_matrix * np.log2(p_matrix + 1e-10))
            
            mutual_info = h_row + h_col - h_joint
            uncertainty_coeff = 2 * mutual_info / (h_row + h_col) if (h_row + h_col) > 0 else 0
            related_measures["uncertainty_coefficient"] = uncertainty_coeff
            
            return related_measures
            
        except Exception as e:
            self.logger.error(f"Error calculating related measures: {str(e)}")
            return {}
    
    def _calculate_statistical_significance(self, matrix: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistical significance tests."""
        try:
            significance_results = {}
            matrix_array = matrix.values
            
            # Chi-square test
            chi2_stat, p_value, dof, expected = chi2_contingency(matrix_array)
            
            significance_results["chi_square"] = {
                "statistic": chi2_stat,
                "p_value": p_value,
                "degrees_of_freedom": dof,
                "significant": p_value < 0.05
            }
            
            # Likelihood ratio test
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    from scipy.stats import power_divergence
                    g_stat, g_p_value, g_dof, g_expected = power_divergence(matrix_array, lambda_="log-likelihood")
                    significance_results["likelihood_ratio"] = {
                        "statistic": g_stat,
                        "p_value": g_p_value,
                        "degrees_of_freedom": g_dof,
                        "significant": g_p_value < 0.05
                    }
                except:
                    pass
            
            # Effect size significance
            n = matrix_array.sum()
            cramers_v = self._calculate_basic_cramers_v(pd.DataFrame(matrix_array))
            
            # Critical value for Cramér's V (approximate)
            critical_cramers_v = np.sqrt(stats.chi2.ppf(0.95, dof) / (n * (min(matrix.shape) - 1)))
            
            significance_results["effect_size"] = {
                "cramers_v": cramers_v,
                "critical_value": critical_cramers_v,
                "significant": cramers_v > critical_cramers_v
            }
            
            return significance_results
            
        except Exception as e:
            self.logger.error(f"Error calculating statistical significance: {str(e)}")
            return {}
    
    def _get_calculation_details(self, matrix: pd.DataFrame) -> Dict[str, Any]:
        """Get detailed calculation information."""
        try:
            details = {
                "matrix_dimensions": matrix.shape,
                "total_observations": matrix.sum().sum(),
                "degrees_of_freedom": (matrix.shape[0] - 1) * (matrix.shape[1] - 1),
                "min_dimension": min(matrix.shape),
                "max_dimension": max(matrix.shape)
            }
            
            # Calculate expected frequencies
            row_totals = matrix.sum(axis=1)
            col_totals = matrix.sum(axis=0)
            total_sum = matrix.sum().sum()
            
            if total_sum > 0:
                expected_freq = []
                for i in range(matrix.shape[0]):
                    for j in range(matrix.shape[1]):
                        expected = (row_totals.iloc[i] * col_totals.iloc[j]) / total_sum
                        expected_freq.append(expected)
                
                details["expected_frequencies"] = {
                    "min": min(expected_freq),
                    "max": max(expected_freq),
                    "mean": np.mean(expected_freq),
                    "cells_below_5": sum(1 for f in expected_freq if f < 5)
                }
            
            return details
            
        except Exception as e:
            self.logger.error(f"Error getting calculation details: {str(e)}")
            return {}
    
    def _generate_summary_statistics(self, calculation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics for all Cramér's V calculations."""
        try:
            values = calculation_result["values"]
            
            if not values:
                return {"error": "No valid Cramér's V values calculated"}
            
            summary = {
                "count": len(values),
                "mean": np.mean(values),
                "median": np.median(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "range": np.max(values) - np.min(values)
            }
            
            # Effect size distribution
            interpretations = calculation_result["effect_size_interpretations"]
            interpretation_counts = {}
            for interp in interpretations:
                interpretation_counts[interp] = interpretation_counts.get(interp, 0) + 1
            
            summary["effect_size_distribution"] = interpretation_counts
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating summary statistics: {str(e)}")
            return {"error": str(e)}
    
    def get_calculation_options(self) -> Dict[str, Any]:
        """Get current calculation options."""
        return self.calculation_options.copy()
    
    def set_calculation_options(self, options: Dict[str, Any]) -> None:
        """Set calculation options."""
        self.calculation_options.update(options)
        self.logger.info(f"Calculation options updated: {options}")
    
    def set_effect_size_thresholds(self, thresholds: Dict[str, float]) -> None:
        """Set custom effect size interpretation thresholds."""
        self.effect_size_thresholds.update(thresholds)
        self.logger.info(f"Effect size thresholds updated: {thresholds}")
    
    def get_effect_size_thresholds(self) -> Dict[str, float]:
        """Get current effect size interpretation thresholds."""
        return self.effect_size_thresholds.copy()
