#!/usr/bin/env python3
"""
Global Fit Analyzer Module
Professional Contingency Analysis Suite
Performs global fit analysis including chi-square tests, likelihood ratios, and goodness-of-fit
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from datetime import datetime
from scipy import stats
from scipy.stats import chi2_contingency, power_divergence


class GlobalFitAnalyzer:
    """
    Performs comprehensive global fit analysis for contingency tables.
    Includes chi-square tests, likelihood ratios, and goodness-of-fit measures.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.analysis_options = {
            "chi_square_test": True,
            "likelihood_ratio_test": True,
            "freeman_tukey_test": True,
            "cressie_read_test": True,
            "modified_log_likelihood_test": True,
            "continuity_correction": True,
            "significance_level": 0.05,
            "min_expected_frequency": 5,
            "bootstrap_samples": 1000,
            "confidence_level": 0.95,
            "include_residuals": True,
            "standardized_residuals": True,
            "adjusted_residuals": True
        }
    
    def analyze_global_fit(self, confusion_matrices: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive global fit analysis.
        
        Args:
            confusion_matrices: Dictionary containing confusion matrices
            
        Returns:
            Dictionary with global fit analysis results
        """
        try:
            self.logger.info("Starting global fit analysis")
            
            analysis_result = {
                "test_results": {},
                "goodness_of_fit": {},
                "residual_analysis": {},
                "power_analysis": {},
                "overall_summary": {},
                "metadata": {
                    "analysis_timestamp": datetime.now().isoformat(),
                    "analysis_options": self.analysis_options.copy()
                }
            }
            
            # Extract matrices
            matrices = confusion_matrices.get("matrices", {})
            if not matrices:
                raise ValueError("No matrices found in confusion matrices data")
            
            # Analyze each matrix
            for matrix_name, matrix_data in matrices.items():
                self.logger.info(f"Analyzing matrix: {matrix_name}")
                
                # Analyze individual matrix
                matrix_analysis = self._analyze_individual_matrix(matrix_data, matrix_name)
                
                # Store results
                analysis_result["test_results"][matrix_name] = matrix_analysis["tests"]
                analysis_result["goodness_of_fit"][matrix_name] = matrix_analysis["goodness_of_fit"]
                analysis_result["residual_analysis"][matrix_name] = matrix_analysis["residuals"]
                analysis_result["power_analysis"][matrix_name] = matrix_analysis["power_analysis"]
            
            # Generate overall summary
            analysis_result["overall_summary"] = self._generate_overall_summary(analysis_result)
            
            # Extract key statistics for main interface
            analysis_result.update(self._extract_key_statistics(analysis_result))
            
            self.logger.info("Global fit analysis completed successfully")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Global fit analysis failed: {str(e)}")
            raise
    
    def _analyze_individual_matrix(self, matrix_data: pd.DataFrame, matrix_name: str) -> Dict[str, Any]:
        """Analyze an individual confusion matrix."""
        try:
            # Initialize analysis structure
            matrix_analysis = {
                "tests": {},
                "goodness_of_fit": {},
                "residuals": {},
                "power_analysis": {}
            }
            
            # Remove marginals if present
            analysis_matrix = self._prepare_matrix_for_analysis(matrix_data)
            
            # Validate matrix for analysis
            validation_result = self._validate_matrix_for_analysis(analysis_matrix)
            if not validation_result["valid"]:
                matrix_analysis["tests"]["validation_errors"] = validation_result["errors"]
                return matrix_analysis
            
            # Perform statistical tests
            matrix_analysis["tests"] = self._perform_statistical_tests(analysis_matrix)
            
            # Calculate goodness-of-fit measures
            matrix_analysis["goodness_of_fit"] = self._calculate_goodness_of_fit(analysis_matrix)
            
            # Perform residual analysis
            if self.analysis_options["include_residuals"]:
                matrix_analysis["residuals"] = self._perform_residual_analysis(analysis_matrix)
            
            # Perform power analysis
            matrix_analysis["power_analysis"] = self._perform_power_analysis(analysis_matrix)
            
            return matrix_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing matrix '{matrix_name}': {str(e)}")
            raise
    
    def _prepare_matrix_for_analysis(self, matrix_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare matrix for statistical analysis."""
        try:
            # Remove marginals if present
            analysis_matrix = matrix_data.copy()
            
            if "Row_Total" in analysis_matrix.columns:
                analysis_matrix = analysis_matrix.drop("Row_Total", axis=1)
            if "Col_Total" in analysis_matrix.index:
                analysis_matrix = analysis_matrix.drop("Col_Total", axis=0)
            
            # Ensure all values are numeric and non-negative
            analysis_matrix = analysis_matrix.apply(pd.to_numeric, errors='coerce')
            analysis_matrix = analysis_matrix.fillna(0)
            analysis_matrix = analysis_matrix.clip(lower=0)
            
            return analysis_matrix
            
        except Exception as e:
            self.logger.error(f"Error preparing matrix for analysis: {str(e)}")
            raise
    
    def _validate_matrix_for_analysis(self, matrix: pd.DataFrame) -> Dict[str, Any]:
        """Validate matrix for statistical analysis."""
        try:
            validation_result = {
                "valid": True,
                "errors": [],
                "warnings": []
            }
            
            # Check minimum dimensions
            if matrix.shape[0] < 2 or matrix.shape[1] < 2:
                validation_result["errors"].append("Matrix must be at least 2x2")
                validation_result["valid"] = False
            
            # Check for zero total
            total_sum = matrix.sum().sum()
            if total_sum == 0:
                validation_result["errors"].append("Matrix has zero total sum")
                validation_result["valid"] = False
            
            # Check expected frequencies
            if total_sum > 0:
                row_totals = matrix.sum(axis=1)
                col_totals = matrix.sum(axis=0)
                
                low_expected_count = 0
                for i in range(matrix.shape[0]):
                    for j in range(matrix.shape[1]):
                        expected = (row_totals.iloc[i] * col_totals.iloc[j]) / total_sum
                        if expected < self.analysis_options["min_expected_frequency"]:
                            low_expected_count += 1
                
                if low_expected_count > 0:
                    violation_rate = low_expected_count / (matrix.shape[0] * matrix.shape[1])
                    if violation_rate > 0.2:
                        validation_result["errors"].append(
                            f"Too many cells with expected frequency < {self.analysis_options['min_expected_frequency']}: {violation_rate:.2%}"
                        )
                        validation_result["valid"] = False
                    else:
                        validation_result["warnings"].append(
                            f"Some cells with expected frequency < {self.analysis_options['min_expected_frequency']}: {violation_rate:.2%}"
                        )
            
            return validation_result
            
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Validation failed: {str(e)}"],
                "warnings": []
            }
    
    def _perform_statistical_tests(self, matrix: pd.DataFrame) -> Dict[str, Any]:
        """Perform various statistical tests."""
        try:
            test_results = {}
            matrix_array = matrix.values
            
            # Chi-square test of independence
            if self.analysis_options["chi_square_test"]:
                try:
                    chi2, p_value, dof, expected = chi2_contingency(matrix_array)
                    test_results["chi_square"] = {
                        "statistic": chi2,
                        "p_value": p_value,
                        "degrees_of_freedom": dof,
                        "expected_frequencies": expected,
                        "critical_value": stats.chi2.ppf(
                            1 - self.analysis_options["significance_level"], dof
                        ),
                        "significant": p_value < self.analysis_options["significance_level"]
                    }
                except Exception as e:
                    test_results["chi_square"] = {"error": str(e)}
            
            # Likelihood ratio test (G-test)
            if self.analysis_options["likelihood_ratio_test"]:
                try:
                    g_stat, p_value, dof, expected = power_divergence(
                        matrix_array, lambda_="log-likelihood"
                    )
                    test_results["likelihood_ratio"] = {
                        "statistic": g_stat,
                        "p_value": p_value,
                        "degrees_of_freedom": dof,
                        "expected_frequencies": expected,
                        "significant": p_value < self.analysis_options["significance_level"]
                    }
                except Exception as e:
                    test_results["likelihood_ratio"] = {"error": str(e)}
            
            # Freeman-Tukey test
            if self.analysis_options["freeman_tukey_test"]:
                try:
                    ft_stat, p_value, dof, expected = power_divergence(
                        matrix_array, lambda_="freeman-tukey"
                    )
                    test_results["freeman_tukey"] = {
                        "statistic": ft_stat,
                        "p_value": p_value,
                        "degrees_of_freedom": dof,
                        "significant": p_value < self.analysis_options["significance_level"]
                    }
                except Exception as e:
                    test_results["freeman_tukey"] = {"error": str(e)}
            
            # Cressie-Read test
            if self.analysis_options["cressie_read_test"]:
                try:
                    cr_stat, p_value, dof, expected = power_divergence(
                        matrix_array, lambda_="cressie-read"
                    )
                    test_results["cressie_read"] = {
                        "statistic": cr_stat,
                        "p_value": p_value,
                        "degrees_of_freedom": dof,
                        "significant": p_value < self.analysis_options["significance_level"]
                    }
                except Exception as e:
                    test_results["cressie_read"] = {"error": str(e)}
            
            # Modified log-likelihood test
            if self.analysis_options["modified_log_likelihood_test"]:
                try:
                    mll_stat, p_value, dof, expected = power_divergence(
                        matrix_array, lambda_="mod-log-likelihood"
                    )
                    test_results["modified_log_likelihood"] = {
                        "statistic": mll_stat,
                        "p_value": p_value,
                        "degrees_of_freedom": dof,
                        "significant": p_value < self.analysis_options["significance_level"]
                    }
                except Exception as e:
                    test_results["modified_log_likelihood"] = {"error": str(e)}
            
            return test_results
            
        except Exception as e:
            self.logger.error(f"Error performing statistical tests: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_goodness_of_fit(self, matrix: pd.DataFrame) -> Dict[str, Any]:
        """Calculate goodness-of-fit measures."""
        try:
            goodness_measures = {}
            matrix_array = matrix.values
            
            # Calculate expected frequencies under independence
            row_totals = matrix.sum(axis=1)
            col_totals = matrix.sum(axis=0)
            total_sum = matrix.sum().sum()
            
            expected = np.outer(row_totals, col_totals) / total_sum
            
            # Chi-square based measures
            chi_square = np.sum((matrix_array - expected) ** 2 / expected)
            n = total_sum
            r = matrix.shape[0]
            c = matrix.shape[1]
            
            # Phi coefficient (for 2x2 tables)
            if matrix.shape == (2, 2):
                phi = np.sqrt(chi_square / n)
                goodness_measures["phi"] = phi
            
            # Contingency coefficient
            contingency_coeff = np.sqrt(chi_square / (chi_square + n))
            goodness_measures["contingency_coefficient"] = contingency_coeff
            
            # Cramer's V
            cramers_v = np.sqrt(chi_square / (n * (min(r, c) - 1)))
            goodness_measures["cramers_v"] = cramers_v
            
            # Tschuprow's T
            tschuprow_t = np.sqrt(chi_square / (n * np.sqrt((r - 1) * (c - 1))))
            goodness_measures["tschuprow_t"] = tschuprow_t
            
            # Uncertainty coefficient (symmetric)
            # H(X|Y) and H(Y|X) calculations
            p_x = row_totals / n
            p_y = col_totals / n
            p_xy = matrix_array / n
            
            h_x = -np.sum(p_x * np.log2(p_x + 1e-10))
            h_y = -np.sum(p_y * np.log2(p_y + 1e-10))
            h_xy = -np.sum(p_xy * np.log2(p_xy + 1e-10))
            
            uncertainty_coeff = 2 * (h_x + h_y - h_xy) / (h_x + h_y) if (h_x + h_y) > 0 else 0
            goodness_measures["uncertainty_coefficient"] = uncertainty_coeff
            
            # Lambda (Goodman and Kruskal's lambda)
            # Proportional reduction in error
            max_row = np.max(matrix_array, axis=1)
            max_col = np.max(matrix_array, axis=0)
            
            lambda_symmetric = (np.sum(max_row) + np.sum(max_col) - np.max(row_totals) - np.max(col_totals)) / (2 * n - np.max(row_totals) - np.max(col_totals))
            goodness_measures["lambda_symmetric"] = lambda_symmetric
            
            # Goodman and Kruskal's tau
            # Based on proportional reduction in error
            e1 = n - np.max(row_totals)
            e2 = np.sum(row_totals) - np.sum(np.max(matrix_array, axis=1))
            
            tau = (e1 - e2) / e1 if e1 > 0 else 0
            goodness_measures["tau"] = tau
            
            return goodness_measures
            
        except Exception as e:
            self.logger.error(f"Error calculating goodness-of-fit measures: {str(e)}")
            return {"error": str(e)}
    
    def _perform_residual_analysis(self, matrix: pd.DataFrame) -> Dict[str, Any]:
        """Perform residual analysis."""
        try:
            residual_analysis = {}
            matrix_array = matrix.values
            
            # Calculate expected frequencies
            row_totals = matrix.sum(axis=1)
            col_totals = matrix.sum(axis=0)
            total_sum = matrix.sum().sum()
            
            expected = np.outer(row_totals, col_totals) / total_sum
            
            # Raw residuals
            raw_residuals = matrix_array - expected
            residual_analysis["raw_residuals"] = raw_residuals
            
            # Standardized residuals
            if self.analysis_options["standardized_residuals"]:
                std_residuals = raw_residuals / np.sqrt(expected)
                residual_analysis["standardized_residuals"] = std_residuals
            
            # Adjusted residuals
            if self.analysis_options["adjusted_residuals"]:
                n = total_sum
                adjusted_residuals = np.zeros_like(raw_residuals)
                
                for i in range(matrix.shape[0]):
                    for j in range(matrix.shape[1]):
                        variance = expected[i, j] * (1 - row_totals.iloc[i]/n) * (1 - col_totals.iloc[j]/n)
                        if variance > 0:
                            adjusted_residuals[i, j] = raw_residuals[i, j] / np.sqrt(variance)
                
                residual_analysis["adjusted_residuals"] = adjusted_residuals
            
            # Residual statistics
            residual_analysis["residual_statistics"] = {
                "max_absolute_residual": np.max(np.abs(raw_residuals)),
                "sum_squared_residuals": np.sum(raw_residuals ** 2),
                "mean_absolute_residual": np.mean(np.abs(raw_residuals))
            }
            
            return residual_analysis
            
        except Exception as e:
            self.logger.error(f"Error performing residual analysis: {str(e)}")
            return {"error": str(e)}
    
    def _perform_power_analysis(self, matrix: pd.DataFrame) -> Dict[str, Any]:
        """Perform power analysis."""
        try:
            power_analysis = {}
            matrix_array = matrix.values
            
            # Calculate effect size (w)
            row_totals = matrix.sum(axis=1)
            col_totals = matrix.sum(axis=0)
            total_sum = matrix.sum().sum()
            
            expected = np.outer(row_totals, col_totals) / total_sum
            
            # Effect size (Cohen's w)
            w = np.sqrt(np.sum((matrix_array - expected) ** 2 / expected) / total_sum)
            power_analysis["effect_size_w"] = w
            
            # Effect size interpretation
            if w < 0.1:
                effect_size_interpretation = "negligible"
            elif w < 0.3:
                effect_size_interpretation = "small"
            elif w < 0.5:
                effect_size_interpretation = "medium"
            else:
                effect_size_interpretation = "large"
            
            power_analysis["effect_size_interpretation"] = effect_size_interpretation
            
            # Degrees of freedom
            dof = (matrix.shape[0] - 1) * (matrix.shape[1] - 1)
            power_analysis["degrees_of_freedom"] = dof
            
            # Critical value
            alpha = self.analysis_options["significance_level"]
            critical_value = stats.chi2.ppf(1 - alpha, dof)
            power_analysis["critical_value"] = critical_value
            
            # Non-centrality parameter
            ncp = total_sum * (w ** 2)
            power_analysis["noncentrality_parameter"] = ncp
            
            # Power calculation (approximate)
            # Using non-central chi-square distribution
            try:
                from scipy.stats import ncx2
                power = 1 - ncx2.cdf(critical_value, dof, ncp)
                power_analysis["statistical_power"] = power
            except ImportError:
                # Fallback approximation
                power_analysis["statistical_power"] = "not_calculated"
            
            return power_analysis
            
        except Exception as e:
            self.logger.error(f"Error performing power analysis: {str(e)}")
            return {"error": str(e)}
    
    def _generate_overall_summary(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall summary of global fit analysis."""
        try:
            summary = {
                "total_matrices": len(analysis_result["test_results"]),
                "successful_analyses": 0,
                "significant_results": 0,
                "average_p_value": None,
                "average_effect_size": None,
                "common_violations": []
            }
            
            p_values = []
            effect_sizes = []
            
            for matrix_name, test_results in analysis_result["test_results"].items():
                if "chi_square" in test_results and "error" not in test_results["chi_square"]:
                    summary["successful_analyses"] += 1
                    
                    p_val = test_results["chi_square"]["p_value"]
                    p_values.append(p_val)
                    
                    if p_val < self.analysis_options["significance_level"]:
                        summary["significant_results"] += 1
                
                # Collect effect sizes
                if matrix_name in analysis_result["power_analysis"]:
                    power_data = analysis_result["power_analysis"][matrix_name]
                    if "effect_size_w" in power_data:
                        effect_sizes.append(power_data["effect_size_w"])
            
            # Calculate averages
            if p_values:
                summary["average_p_value"] = np.mean(p_values)
            
            if effect_sizes:
                summary["average_effect_size"] = np.mean(effect_sizes)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating overall summary: {str(e)}")
            return {"error": str(e)}
    
    def _extract_key_statistics(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key statistics for main interface compatibility."""
        try:
            # Extract first matrix results for main interface
            test_results = analysis_result.get("test_results", {})
            
            if test_results:
                first_matrix = list(test_results.keys())[0]
                first_result = test_results[first_matrix]
                
                if "chi_square" in first_result and "error" not in first_result["chi_square"]:
                    chi_square_data = first_result["chi_square"]
                    return {
                        "chi_square": chi_square_data["statistic"],
                        "p_value": chi_square_data["p_value"],
                        "degrees_of_freedom": chi_square_data["degrees_of_freedom"]
                    }
            
            # Fallback values
            return {
                "chi_square": 0,
                "p_value": 1.0,
                "degrees_of_freedom": 1
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting key statistics: {str(e)}")
            return {
                "chi_square": 0,
                "p_value": 1.0,
                "degrees_of_freedom": 1
            }
    
    def get_analysis_options(self) -> Dict[str, Any]:
        """Get current analysis options."""
        return self.analysis_options.copy()
    
    def set_analysis_options(self, options: Dict[str, Any]) -> None:
        """Set analysis options."""
        self.analysis_options.update(options)
        self.logger.info(f"Analysis options updated: {options}")
