#!/usr/bin/env python3
"""
Statistics Validation Module
Professional Contingency Analysis Suite
Validates statistical results including global fit analysis and Cramér's V calculations
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from datetime import datetime
from scipy import stats


class StatisticsValidator:
    """
    Validates statistical results for accuracy and reliability.
    Ensures computed statistics meet quality and significance standards.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.stats_rules = {
            "min_p_value": 0.0,
            "max_p_value": 1.0,
            "min_degrees_of_freedom": 1,
            "max_chi_square": 1e6,
            "min_cramers_v": 0.0,
            "max_cramers_v": 1.0,
            "significance_level": 0.05,
            "min_sample_size": 10,
            "max_expected_freq_violations": 0.2,
            "min_effect_size": 0.1
        }
    
    def validate_statistics(self, global_fit_results: Dict[str, Any], 
                          cramers_v_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive validation of statistical results.
        
        Args:
            global_fit_results: Results from global fit analysis
            cramers_v_results: Results from Cramér's V calculations
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "statistics_summary": {},
            "validation_timestamp": datetime.now().isoformat()
        }
        
        try:
            # Validate global fit results
            fit_validation = self._validate_global_fit_results(global_fit_results)
            validation_result["errors"].extend(fit_validation["errors"])
            validation_result["warnings"].extend(fit_validation["warnings"])
            
            # Validate Cramér's V results
            cramers_validation = self._validate_cramers_v_results(cramers_v_results)
            validation_result["errors"].extend(cramers_validation["errors"])
            validation_result["warnings"].extend(cramers_validation["warnings"])
            
            # Cross-validate statistical consistency
            consistency_validation = self._validate_statistical_consistency(
                global_fit_results, cramers_v_results
            )
            validation_result["errors"].extend(consistency_validation["errors"])
            validation_result["warnings"].extend(consistency_validation["warnings"])
            
            # Validate statistical significance and power
            significance_validation = self._validate_statistical_significance(
                global_fit_results, cramers_v_results
            )
            validation_result["errors"].extend(significance_validation["errors"])
            validation_result["warnings"].extend(significance_validation["warnings"])
            
            # Generate statistics summary
            validation_result["statistics_summary"] = self._generate_statistics_summary(
                global_fit_results, cramers_v_results
            )
            
            # Set final validation status
            validation_result["valid"] = len(validation_result["errors"]) == 0
            
            if validation_result["valid"]:
                self.logger.info("Statistics validation passed successfully")
            else:
                self.logger.error(f"Statistics validation failed with {len(validation_result['errors'])} errors")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Statistics validation error: {str(e)}")
            validation_result["valid"] = False
            validation_result["errors"].append(f"Statistics validation process failed: {str(e)}")
            return validation_result
    
    def _validate_global_fit_results(self, global_fit_results: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate global fit analysis results."""
        errors = []
        warnings = []
        
        try:
            # Check if results exist
            if not global_fit_results or not isinstance(global_fit_results, dict):
                errors.append("Global fit results are empty or not in expected format")
                return {"errors": errors, "warnings": warnings}
            
            # Check for required fields
            required_fields = ["chi_square", "p_value", "degrees_of_freedom"]
            for field in required_fields:
                if field not in global_fit_results:
                    errors.append(f"Missing required field in global fit results: {field}")
            
            # Validate chi-square values
            if "chi_square" in global_fit_results:
                chi_square = global_fit_results["chi_square"]
                
                if isinstance(chi_square, (list, np.ndarray)):
                    # Multiple chi-square values
                    for i, chi_val in enumerate(chi_square):
                        if not isinstance(chi_val, (int, float)) or np.isnan(chi_val):
                            errors.append(f"Invalid chi-square value at index {i}: {chi_val}")
                        elif chi_val < 0:
                            errors.append(f"Negative chi-square value at index {i}: {chi_val}")
                        elif chi_val > self.stats_rules["max_chi_square"]:
                            warnings.append(f"Very large chi-square value at index {i}: {chi_val}")
                else:
                    # Single chi-square value
                    if not isinstance(chi_square, (int, float)) or np.isnan(chi_square):
                        errors.append(f"Invalid chi-square value: {chi_square}")
                    elif chi_square < 0:
                        errors.append(f"Negative chi-square value: {chi_square}")
                    elif chi_square > self.stats_rules["max_chi_square"]:
                        warnings.append(f"Very large chi-square value: {chi_square}")
            
            # Validate p-values
            if "p_value" in global_fit_results:
                p_value = global_fit_results["p_value"]
                
                if isinstance(p_value, (list, np.ndarray)):
                    # Multiple p-values
                    for i, p_val in enumerate(p_value):
                        if not isinstance(p_val, (int, float)) or np.isnan(p_val):
                            errors.append(f"Invalid p-value at index {i}: {p_val}")
                        elif p_val < self.stats_rules["min_p_value"] or p_val > self.stats_rules["max_p_value"]:
                            errors.append(f"P-value out of range at index {i}: {p_val}")
                else:
                    # Single p-value
                    if not isinstance(p_value, (int, float)) or np.isnan(p_value):
                        errors.append(f"Invalid p-value: {p_value}")
                    elif p_value < self.stats_rules["min_p_value"] or p_value > self.stats_rules["max_p_value"]:
                        errors.append(f"P-value out of range: {p_value}")
            
            # Validate degrees of freedom
            if "degrees_of_freedom" in global_fit_results:
                dof = global_fit_results["degrees_of_freedom"]
                
                if isinstance(dof, (list, np.ndarray)):
                    # Multiple degrees of freedom
                    for i, dof_val in enumerate(dof):
                        if not isinstance(dof_val, (int, float)) or np.isnan(dof_val):
                            errors.append(f"Invalid degrees of freedom at index {i}: {dof_val}")
                        elif dof_val < self.stats_rules["min_degrees_of_freedom"]:
                            errors.append(f"Insufficient degrees of freedom at index {i}: {dof_val}")
                else:
                    # Single degrees of freedom
                    if not isinstance(dof, (int, float)) or np.isnan(dof):
                        errors.append(f"Invalid degrees of freedom: {dof}")
                    elif dof < self.stats_rules["min_degrees_of_freedom"]:
                        errors.append(f"Insufficient degrees of freedom: {dof}")
            
            # Validate expected frequencies (if available)
            if "expected_frequencies" in global_fit_results:
                expected_freq = global_fit_results["expected_frequencies"]
                
                if isinstance(expected_freq, (list, np.ndarray)):
                    violations = sum(1 for freq in expected_freq if freq < 5)
                    violation_rate = violations / len(expected_freq)
                    
                    if violation_rate > self.stats_rules["max_expected_freq_violations"]:
                        errors.append(f"Too many expected frequencies < 5: {violation_rate:.2%}")
                    elif violation_rate > 0.1:
                        warnings.append(f"Some expected frequencies < 5: {violation_rate:.2%}")
        
        except Exception as e:
            errors.append(f"Global fit validation error: {str(e)}")
        
        return {"errors": errors, "warnings": warnings}
    
    def _validate_cramers_v_results(self, cramers_v_results: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate Cramér's V calculation results."""
        errors = []
        warnings = []
        
        try:
            # Check if results exist
            if not cramers_v_results or not isinstance(cramers_v_results, dict):
                errors.append("Cramér's V results are empty or not in expected format")
                return {"errors": errors, "warnings": warnings}
            
            # Check for required fields
            required_fields = ["values"]
            for field in required_fields:
                if field not in cramers_v_results:
                    errors.append(f"Missing required field in Cramér's V results: {field}")
            
            # Validate Cramér's V values
            if "values" in cramers_v_results:
                values = cramers_v_results["values"]
                
                if isinstance(values, (list, np.ndarray)):
                    # Multiple Cramér's V values
                    for i, v_val in enumerate(values):
                        if not isinstance(v_val, (int, float)) or np.isnan(v_val):
                            errors.append(f"Invalid Cramér's V value at index {i}: {v_val}")
                        elif v_val < self.stats_rules["min_cramers_v"] or v_val > self.stats_rules["max_cramers_v"]:
                            errors.append(f"Cramér's V out of range at index {i}: {v_val}")
                        elif v_val < self.stats_rules["min_effect_size"]:
                            warnings.append(f"Very small effect size (Cramér's V) at index {i}: {v_val}")
                else:
                    # Single Cramér's V value
                    if not isinstance(values, (int, float)) or np.isnan(values):
                        errors.append(f"Invalid Cramér's V value: {values}")
                    elif values < self.stats_rules["min_cramers_v"] or values > self.stats_rules["max_cramers_v"]:
                        errors.append(f"Cramér's V out of range: {values}")
                    elif values < self.stats_rules["min_effect_size"]:
                        warnings.append(f"Very small effect size (Cramér's V): {values}")
            
            # Validate confidence intervals (if available)
            if "confidence_intervals" in cramers_v_results:
                ci = cramers_v_results["confidence_intervals"]
                
                if isinstance(ci, list):
                    for i, interval in enumerate(ci):
                        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                            errors.append(f"Invalid confidence interval format at index {i}: {interval}")
                        else:
                            lower, upper = interval
                            if lower > upper:
                                errors.append(f"Invalid confidence interval bounds at index {i}: [{lower}, {upper}]")
                            if lower < 0 or upper > 1:
                                errors.append(f"Confidence interval out of valid range at index {i}: [{lower}, {upper}]")
            
            # Validate sample sizes (if available)
            if "sample_sizes" in cramers_v_results:
                sample_sizes = cramers_v_results["sample_sizes"]
                
                if isinstance(sample_sizes, (list, np.ndarray)):
                    for i, size in enumerate(sample_sizes):
                        if not isinstance(size, (int, float)) or size < self.stats_rules["min_sample_size"]:
                            warnings.append(f"Small sample size at index {i}: {size}")
                else:
                    if not isinstance(sample_sizes, (int, float)) or sample_sizes < self.stats_rules["min_sample_size"]:
                        warnings.append(f"Small sample size: {sample_sizes}")
        
        except Exception as e:
            errors.append(f"Cramér's V validation error: {str(e)}")
        
        return {"errors": errors, "warnings": warnings}
    
    def _validate_statistical_consistency(self, global_fit_results: Dict[str, Any], 
                                        cramers_v_results: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate consistency between different statistical measures."""
        errors = []
        warnings = []
        
        try:
            # Check if both results are available
            if not global_fit_results or not cramers_v_results:
                warnings.append("Cannot validate statistical consistency - missing results")
                return {"errors": errors, "warnings": warnings}
            
            # Extract values for comparison
            chi_square = global_fit_results.get("chi_square")
            p_value = global_fit_results.get("p_value")
            dof = global_fit_results.get("degrees_of_freedom")
            cramers_v = cramers_v_results.get("values")
            
            # Check if all required values are present
            if not all([chi_square is not None, p_value is not None, dof is not None, cramers_v is not None]):
                warnings.append("Cannot validate statistical consistency - missing statistical values")
                return {"errors": errors, "warnings": warnings}
            
            # Convert to arrays for consistent handling
            if not isinstance(chi_square, (list, np.ndarray)):
                chi_square = [chi_square]
            if not isinstance(p_value, (list, np.ndarray)):
                p_value = [p_value]
            if not isinstance(dof, (list, np.ndarray)):
                dof = [dof]
            if not isinstance(cramers_v, (list, np.ndarray)):
                cramers_v = [cramers_v]
            
            # Check consistency between chi-square and p-value
            for i, (chi_val, p_val, dof_val) in enumerate(zip(chi_square, p_value, dof)):
                if isinstance(chi_val, (int, float)) and isinstance(p_val, (int, float)) and isinstance(dof_val, (int, float)):
                    # Calculate expected p-value from chi-square
                    try:
                        expected_p = 1 - stats.chi2.cdf(chi_val, dof_val)
                        
                        # Allow for some numerical precision differences
                        if abs(expected_p - p_val) > 0.001:
                            warnings.append(f"Inconsistent chi-square and p-value at index {i}: chi²={chi_val:.4f}, p={p_val:.4f}, expected_p={expected_p:.4f}")
                    except Exception:
                        warnings.append(f"Could not verify chi-square/p-value consistency at index {i}")
            
            # Check consistency between chi-square and Cramér's V
            # For a 2x2 table, Cramér's V = sqrt(chi²/N)
            if "sample_sizes" in cramers_v_results:
                sample_sizes = cramers_v_results["sample_sizes"]
                if not isinstance(sample_sizes, (list, np.ndarray)):
                    sample_sizes = [sample_sizes]
                
                for i, (chi_val, v_val, n) in enumerate(zip(chi_square, cramers_v, sample_sizes)):
                    if isinstance(chi_val, (int, float)) and isinstance(v_val, (int, float)) and isinstance(n, (int, float)):
                        if n > 0:
                            expected_v = np.sqrt(chi_val / n)
                            
                            # Allow for some numerical precision differences
                            if abs(expected_v - v_val) > 0.01:
                                warnings.append(f"Inconsistent chi-square and Cramér's V at index {i}: chi²={chi_val:.4f}, V={v_val:.4f}, expected_V={expected_v:.4f}")
            
            # Check for contradictory significance interpretations
            for i, (p_val, v_val) in enumerate(zip(p_value, cramers_v)):
                if isinstance(p_val, (int, float)) and isinstance(v_val, (int, float)):
                    significant = p_val < self.stats_rules["significance_level"]
                    large_effect = v_val > 0.3  # Common threshold for large effect
                    
                    if significant and v_val < 0.1:
                        warnings.append(f"Statistically significant but very small effect size at index {i}: p={p_val:.4f}, V={v_val:.4f}")
                    elif not significant and large_effect:
                        warnings.append(f"Large effect size but not statistically significant at index {i}: p={p_val:.4f}, V={v_val:.4f}")
        
        except Exception as e:
            errors.append(f"Statistical consistency validation error: {str(e)}")
        
        return {"errors": errors, "warnings": warnings}
    
    def _validate_statistical_significance(self, global_fit_results: Dict[str, Any], 
                                         cramers_v_results: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate statistical significance and power considerations."""
        errors = []
        warnings = []
        
        try:
            # Check statistical power considerations
            if "sample_sizes" in cramers_v_results:
                sample_sizes = cramers_v_results["sample_sizes"]
                if not isinstance(sample_sizes, (list, np.ndarray)):
                    sample_sizes = [sample_sizes]
                
                for i, n in enumerate(sample_sizes):
                    if isinstance(n, (int, float)):
                        if n < 30:
                            warnings.append(f"Small sample size may affect statistical power at index {i}: n={n}")
                        elif n > 10000:
                            warnings.append(f"Very large sample size may lead to trivial significance at index {i}: n={n}")
            
            # Check multiple testing considerations
            p_values = global_fit_results.get("p_value")
            if isinstance(p_values, (list, np.ndarray)) and len(p_values) > 1:
                # Multiple comparisons detected
                significant_count = sum(1 for p in p_values if isinstance(p, (int, float)) and p < self.stats_rules["significance_level"])
                
                if significant_count > 0:
                    warnings.append(f"Multiple testing detected ({len(p_values)} tests, {significant_count} significant). Consider correction for multiple comparisons.")
            
            # Check for effect size interpretation
            cramers_v = cramers_v_results.get("values")
            if isinstance(cramers_v, (list, np.ndarray)):
                for i, v in enumerate(cramers_v):
                    if isinstance(v, (int, float)):
                        if v < 0.1:
                            warnings.append(f"Negligible effect size at index {i}: V={v:.4f}")
                        elif v > 0.5:
                            warnings.append(f"Very large effect size at index {i}: V={v:.4f} - verify data quality")
            
            # Check for statistical assumptions
            if "expected_frequencies" in global_fit_results:
                expected_freq = global_fit_results["expected_frequencies"]
                if isinstance(expected_freq, (list, np.ndarray)):
                    low_freq_count = sum(1 for freq in expected_freq if freq < 5)
                    if low_freq_count > 0:
                        warnings.append(f"Chi-square test assumption violated: {low_freq_count} cells with expected frequency < 5")
        
        except Exception as e:
            errors.append(f"Statistical significance validation error: {str(e)}")
        
        return {"errors": errors, "warnings": warnings}
    
    def _generate_statistics_summary(self, global_fit_results: Dict[str, Any], 
                                   cramers_v_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive summary of statistical results."""
        summary = {
            "global_fit_summary": {},
            "cramers_v_summary": {},
            "overall_assessment": {}
        }
        
        try:
            # Summarize global fit results
            if global_fit_results:
                chi_square = global_fit_results.get("chi_square")
                p_value = global_fit_results.get("p_value")
                dof = global_fit_results.get("degrees_of_freedom")
                
                if chi_square is not None:
                    if isinstance(chi_square, (list, np.ndarray)):
                        summary["global_fit_summary"]["chi_square"] = {
                            "count": len(chi_square),
                            "min": min(chi_square),
                            "max": max(chi_square),
                            "mean": np.mean(chi_square),
                            "std": np.std(chi_square)
                        }
                    else:
                        summary["global_fit_summary"]["chi_square"] = {
                            "value": chi_square,
                            "count": 1
                        }
                
                if p_value is not None:
                    if isinstance(p_value, (list, np.ndarray)):
                        significant_count = sum(1 for p in p_value if p < self.stats_rules["significance_level"])
                        summary["global_fit_summary"]["p_value"] = {
                            "count": len(p_value),
                            "min": min(p_value),
                            "max": max(p_value),
                            "mean": np.mean(p_value),
                            "significant_count": significant_count,
                            "significant_percentage": (significant_count / len(p_value)) * 100
                        }
                    else:
                        summary["global_fit_summary"]["p_value"] = {
                            "value": p_value,
                            "significant": p_value < self.stats_rules["significance_level"]
                        }
            
            # Summarize Cramér's V results
            if cramers_v_results:
                values = cramers_v_results.get("values")
                
                if values is not None:
                    if isinstance(values, (list, np.ndarray)):
                        summary["cramers_v_summary"]["values"] = {
                            "count": len(values),
                            "min": min(values),
                            "max": max(values),
                            "mean": np.mean(values),
                            "std": np.std(values)
                        }
                        
                        # Effect size categorization
                        small_effect = sum(1 for v in values if v < 0.1)
                        medium_effect = sum(1 for v in values if 0.1 <= v < 0.3)
                        large_effect = sum(1 for v in values if v >= 0.3)
                        
                        summary["cramers_v_summary"]["effect_sizes"] = {
                            "small": small_effect,
                            "medium": medium_effect,
                            "large": large_effect
                        }
                    else:
                        summary["cramers_v_summary"]["values"] = {
                            "value": values,
                            "count": 1
                        }
                        
                        # Effect size categorization
                        if values < 0.1:
                            effect_size = "small"
                        elif values < 0.3:
                            effect_size = "medium"
                        else:
                            effect_size = "large"
                        
                        summary["cramers_v_summary"]["effect_size"] = effect_size
            
            # Overall assessment
            summary["overall_assessment"] = {
                "statistics_computed": bool(global_fit_results or cramers_v_results),
                "has_global_fit": bool(global_fit_results),
                "has_cramers_v": bool(cramers_v_results),
                "assessment_timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            summary["error"] = f"Statistics summary generation error: {str(e)}"
        
        return summary
    
    def set_stats_rules(self, rules: Dict[str, Any]) -> None:
        """Update statistics validation rules."""
        self.stats_rules.update(rules)
        self.logger.info(f"Statistics validation rules updated: {rules}")
    
    def get_stats_rules(self) -> Dict[str, Any]:
        """Get current statistics validation rules."""
        return self.stats_rules.copy()
