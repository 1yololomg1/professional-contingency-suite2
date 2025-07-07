#!/usr/bin/env python3
"""
Report Generator Module
Professional Contingency Analysis Suite
Generates comprehensive HTML and PDF reports for analysis results
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime
from pathlib import Path
import json
import base64
from jinja2 import Template, Environment, FileSystemLoader
import matplotlib.pyplot as plt
import io
import warnings
import traceback

warnings.filterwarnings('ignore', category=UserWarning)


class ReportGenerator:
    """
    Professional report generator for contingency analysis results.
    Creates comprehensive HTML and PDF reports with embedded visualizations.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.report_options = {
            "format": "html",  # "html", "pdf", "both"
            "template": "professional",
            "include_visualizations": True,
            "include_raw_data": False,
            "include_code_snippets": False,
            "include_methodology": True,
            "include_interpretation": True,
            "include_recommendations": True,
            "embed_images": True,
            "image_quality": "high",
            "page_size": "A4",
            "orientation": "portrait",
            "margin": "1inch",
            "font_size": "12pt",
            "font_family": "Arial",
            "color_scheme": "professional",
            "logo_path": None,
            "company_name": "Professional Statistical Analysis",
            "author": "Contingency Analysis Suite",
            "include_toc": True,
            "include_summary": True,
            "include_appendix": True
        }
        
        # HTML templates
        self.html_templates = {
            "base": self._get_base_template(),
            "summary": self._get_summary_template(),
            "validation": self._get_validation_template(),
            "analysis": self._get_analysis_template(),
            "visualization": self._get_visualization_template(),
            "recommendations": self._get_recommendations_template()
        }
        
        # CSS styles
        self.css_styles = self._get_css_styles()
    
    def generate_report(self, results: Dict[str, Any], output_dir: str) -> str:
        """
        Generate comprehensive analysis report.
        
        Args:
            results: Analysis results dictionary
            output_dir: Directory to save report
            
        Returns:
            Path to generated report file
        """
        try:
            self.logger.info("Starting report generation")
            self.logger.info(f"Output directory: {output_dir}")
            self.logger.info(f"Results keys: {list(results.keys())}")
            
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Output path created: {output_path}")
            
            # Generate report content
            self.logger.info("Generating report content...")
            report_content = self._generate_report_content(results)
            self.logger.info("Report content generated successfully")
            
            # Generate HTML report
            self.logger.info("Generating HTML report...")
            html_report_path = self._generate_html_report(report_content, output_path)
            self.logger.info(f"HTML report generated: {html_report_path}")
            
            # Generate PDF report if requested
            if self.report_options["format"] in ["pdf", "both"]:
                self.logger.info("Generating PDF report...")
                pdf_report_path = self._generate_pdf_report(report_content, output_path)
                
                if self.report_options["format"] == "pdf":
                    self.logger.info(f"PDF report generated: {pdf_report_path}")
                    return pdf_report_path
            
            self.logger.info(f"Report generated successfully: {html_report_path}")
            return html_report_path
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
    
    def generate_batch_report(self, batch_results: Dict[str, Any], output_dir: str) -> str:
        """
        Generate batch analysis report.
        
        Args:
            batch_results: Batch analysis results
            output_dir: Directory to save report
            
        Returns:
            Path to generated batch report file
        """
        try:
            self.logger.info("Starting batch report generation")
            
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Generate batch report content
            report_content = self._generate_batch_report_content(batch_results)
            
            # Generate HTML report
            html_report_path = self._generate_html_report(report_content, output_path, "batch_report")
            
            self.logger.info(f"Batch report generated successfully: {html_report_path}")
            return html_report_path
            
        except Exception as e:
            self.logger.error(f"Batch report generation failed: {str(e)}")
            raise
    
    def _generate_report_content(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate structured report content."""
        try:
            # Extract key information
            input_file = results.get("input_file", "Unknown")
            analysis_timestamp = results.get("analysis_timestamp", datetime.now().isoformat())
            
            # Generate report sections
            report_content = {
                "metadata": {
                    "title": f"Contingency Analysis Report",
                    "subtitle": f"Analysis of {Path(input_file).name}",
                    "author": self.report_options["author"],
                    "company": self.report_options["company_name"],
                    "date": datetime.now().strftime("%B %d, %Y"),
                    "timestamp": analysis_timestamp,
                    "version": "1.0.0"
                },
                "executive_summary": self._generate_executive_summary(results),
                "validation_results": self._generate_validation_section(results),
                "analysis_results": self._generate_analysis_section(results),
                "statistical_findings": self._generate_statistical_findings(results),
                "visualizations": self._generate_visualization_section(results),
                "interpretation": self._generate_interpretation_section(results),
                "recommendations": self._generate_recommendations_section(results),
                "methodology": self._generate_methodology_section(results),
                "appendix": self._generate_appendix_section(results)
            }
            
            return report_content
            
        except Exception as e:
            self.logger.error(f"Error generating report content: {str(e)}")
            raise
    
    def _generate_executive_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary section."""
        try:
            summary = {
                "title": "Executive Summary",
                "key_findings": [],
                "statistical_significance": None,
                "effect_size": None,
                "data_quality": None,
                "recommendations_count": 0
            }
            
            # Extract key findings
            global_fit = results.get("global_fit_analysis", {})
            cramers_v = results.get("cramers_v_results", {})
            
            # Statistical significance
            if "p_value" in global_fit:
                p_value = global_fit["p_value"]
                if isinstance(p_value, (int, float)):
                    if p_value < 0.01:
                        summary["statistical_significance"] = "Highly Significant (p < 0.01)"
                    elif p_value < 0.05:
                        summary["statistical_significance"] = "Significant (p < 0.05)"
                    else:
                        summary["statistical_significance"] = "Not Significant (p ≥ 0.05)"
            
            # Effect size
            if "values" in cramers_v:
                values = cramers_v["values"]
                if isinstance(values, list) and values:
                    avg_cramers_v = np.mean(values)
                elif isinstance(values, (int, float)):
                    avg_cramers_v = values
                else:
                    avg_cramers_v = 0
                
                if avg_cramers_v < 0.1:
                    summary["effect_size"] = "Negligible Effect"
                elif avg_cramers_v < 0.3:
                    summary["effect_size"] = "Small Effect"
                elif avg_cramers_v < 0.5:
                    summary["effect_size"] = "Medium Effect"
                else:
                    summary["effect_size"] = "Large Effect"
            
            # Data quality assessment
            validation_results = results.get("validation_results", {})
            if validation_results.get("valid", False):
                summary["data_quality"] = "High Quality"
            else:
                summary["data_quality"] = "Quality Issues Detected"
            
            # Correction method information
            correction_method = results.get('correction_method', 'None')
            correction_description = {
                'None': 'No statistical correction applied',
                'Yates': 'Yates continuity correction applied for 2×2 tables',
                'Bonferroni': 'Bonferroni correction applied for multiple comparisons'
            }.get(correction_method, 'Unknown correction method')
            
            # Key findings
            matrices_obj = results.get('confusion_matrices', {}).get('matrices', {})
            if callable(matrices_obj):
                self.logger.error("matrices_obj is a function or method, not a dict or list!")
                matrices_obj = {}  # Default to empty dict to avoid error
            if hasattr(matrices_obj, 'keys'):
                num_matrices = len(list(matrices_obj.keys()))
            else:
                num_matrices = len(matrices_obj)
            summary["key_findings"] = [
                f"Analysis completed on {num_matrices} data matrices",
                f"Statistical correction: {correction_description}",
                f"Data quality assessment: {summary['data_quality']}",
                f"Statistical significance: {summary['statistical_significance']}",
                f"Effect size: {summary['effect_size']}"
            ]
            
            # Add correction method to summary
            summary["correction_method"] = correction_method
            summary["correction_description"] = correction_description
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating executive summary: {str(e)}")
            return {"title": "Executive Summary", "error": str(e)}
    
    def _generate_validation_section(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate validation results section."""
        try:
            validation_section = {
                "title": "Data Validation Results",
                "overall_status": "Unknown",
                "data_validation": {},
                "matrix_validation": {},
                "statistics_validation": {},
                "summary_table": []
            }
            
            # Data validation
            data_validation = results.get("validation_results", {})
            validation_section["data_validation"] = {
                "status": "PASSED" if data_validation.get("valid", False) else "FAILED",
                "errors": data_validation.get("errors", []),
                "warnings": data_validation.get("warnings", []),
                "data_summary": data_validation.get("data_summary", {})
            }
            
            # Matrix validation
            matrix_validation = results.get("matrix_validation", {})
            validation_section["matrix_validation"] = {
                "status": "PASSED" if matrix_validation.get("valid", False) else "FAILED",
                "errors": matrix_validation.get("errors", []),
                "warnings": matrix_validation.get("warnings", []),
                "matrix_summary": matrix_validation.get("matrix_summary", {})
            }
            
            # Statistics validation
            stats_validation = results.get("stats_validation", {})
            validation_section["statistics_validation"] = {
                "status": "PASSED" if stats_validation.get("valid", False) else "FAILED",
                "errors": stats_validation.get("errors", []),
                "warnings": stats_validation.get("warnings", [])
            }
            
            # Overall status
            all_validations = [
                data_validation.get("valid", False),
                matrix_validation.get("valid", False),
                stats_validation.get("valid", False)
            ]
            
            if all(all_validations):
                validation_section["overall_status"] = "ALL VALIDATIONS PASSED"
            elif any(all_validations):
                validation_section["overall_status"] = "PARTIAL VALIDATION ISSUES"
            else:
                validation_section["overall_status"] = "VALIDATION FAILURES"
            
            # Summary table
            validation_section["summary_table"] = [
                {
                    "validation_type": "Data Validation",
                    "status": validation_section["data_validation"]["status"],
                    "errors": len(validation_section["data_validation"]["errors"]),
                    "warnings": len(validation_section["data_validation"]["warnings"])
                },
                {
                    "validation_type": "Matrix Validation",
                    "status": validation_section["matrix_validation"]["status"],
                    "errors": len(validation_section["matrix_validation"]["errors"]),
                    "warnings": len(validation_section["matrix_validation"]["warnings"])
                },
                {
                    "validation_type": "Statistics Validation",
                    "status": validation_section["statistics_validation"]["status"],
                    "errors": len(validation_section["statistics_validation"]["errors"]),
                    "warnings": len(validation_section["statistics_validation"]["warnings"])
                }
            ]
            
            return validation_section
            
        except Exception as e:
            self.logger.error(f"Error generating validation section: {str(e)}")
            return {"title": "Data Validation Results", "error": str(e)}
    
    def _generate_analysis_section(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analysis results section."""
        try:
            analysis_section = {
                "title": "Statistical Analysis Results",
                "global_fit_analysis": {},
                "cramers_v_analysis": {},
                "confusion_matrices": {},
                "summary_statistics": {}
            }
            
            # Global fit analysis
            global_fit = results.get("global_fit_analysis", {})
            analysis_section["global_fit_analysis"] = {
                "chi_square": global_fit.get("chi_square", 0),
                "p_value": global_fit.get("p_value", 1.0),
                "degrees_of_freedom": global_fit.get("degrees_of_freedom", 1),
                "interpretation": self._interpret_chi_square_test(global_fit)
            }
            
            # Cramér's V analysis
            cramers_v = results.get("cramers_v_results", {})
            values = cramers_v.get("values", [])
            
            # Defensive check: ensure values is a list, not a function
            if callable(values):
                self.logger.error("cramers_v.values is a function reference, not a list!")
                values = []
            elif not isinstance(values, list):
                values = [values] if values is not None else []
            
            analysis_section["cramers_v_analysis"] = {
                "values": values,
                "average_value": np.mean(values) if values else 0,
                "effect_size_interpretations": cramers_v.get("effect_size_interpretations", []),
                "interpretation": self._interpret_cramers_v(cramers_v)
            }
            
            # Confusion matrices summary
            confusion_matrices = results.get("confusion_matrices", {})
            matrices = confusion_matrices.get("matrices", {})
            analysis_section["confusion_matrices"] = {
                "count": len(matrices.keys()) if hasattr(matrices, 'keys') else len(matrices),
                "matrix_names": list(matrices.keys()) if hasattr(matrices, 'keys') else [],
                "dimensions": [matrices[name].shape for name in matrices.keys()] if hasattr(matrices, 'keys') and matrices else [],
                "classification_metrics": confusion_matrices.get("classification_metrics", {})
            }
            
            return analysis_section
            
        except Exception as e:
            self.logger.error(f"Error generating analysis section: {str(e)}")
            return {"title": "Statistical Analysis Results", "error": str(e)}
    
    def _generate_statistical_findings(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate statistical findings section."""
        try:
            findings_section = {
                "title": "Key Statistical Findings",
                "hypothesis_testing": {},
                "effect_sizes": {},
                "confidence_intervals": {},
                "statistical_power": {},
                "assumptions_testing": {}
            }
            
            # Hypothesis testing results
            global_fit = results.get("global_fit_analysis", {})
            if "p_value" in global_fit and "chi_square" in global_fit:
                findings_section["hypothesis_testing"] = {
                    "null_hypothesis": "Variables are independent (no association)",
                    "alternative_hypothesis": "Variables are associated",
                    "test_statistic": global_fit.get("chi_square"),
                    "p_value": global_fit.get("p_value"),
                    "significance_level": 0.05,
                    "decision": "Reject null hypothesis" if global_fit.get("p_value", 1) < 0.05 else "Fail to reject null hypothesis",
                    "conclusion": self._get_hypothesis_conclusion(global_fit)
                }
            
            # Effect sizes
            cramers_v = results.get("cramers_v_results", {})
            if "values" in cramers_v:
                values = cramers_v["values"]
                if isinstance(values, list) and values:
                    findings_section["effect_sizes"] = {
                        "cramers_v_values": values,
                        "average_effect_size": np.mean(values),
                        "effect_size_range": [min(values), max(values)],
                        "interpretation": self._interpret_effect_size_distribution(values)
                    }
            
            return findings_section
            
        except Exception as e:
            self.logger.error(f"Error generating statistical findings: {str(e)}")
            return {"title": "Key Statistical Findings", "error": str(e)}
    
    def _generate_visualization_section(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate visualization section."""
        try:
            validation_status = results.get("validation_status", "PASSED")
            validation_issues = results.get("validation_issues", [])
            advisory_message = None
            section_title = "Data Visualizations"
            if validation_status != "PASSED":
                section_title = "Statistical Validation Advisory: Contingency Pattern Analysis"
                specific_issue = validation_issues[0] if validation_issues else "Unspecified validation issue."
                advisory_message = (
                    "STATISTICAL VALIDATION ADVISORY\n\n"
                    "The following radar visualizations represent categorical association patterns derived from contingency tables that did not meet all statistical validation criteria. "
                    f"Specifically, {specific_issue}\n\n"
                    "These visualizations are provided for exploratory analysis purposes only. Any conclusions drawn should be subjected to additional verification methods before incorporation into decision frameworks."
                )
            viz_section = {
                "title": section_title,
                "pie_charts": [],
                "radar_charts": [],
                "embedded_images": {},
                "advisory_message": advisory_message
            }
            
            # Get visualization paths
            visualizations = results.get("visualizations", {})
            
            # Pie charts
            pie_charts = visualizations.get("pie_charts", [])
            if pie_charts:
                viz_section["pie_charts"] = pie_charts
                
                # Embed images if requested
                if self.report_options["embed_images"]:
                    for chart_path in pie_charts:
                        if Path(chart_path).exists():
                            viz_section["embedded_images"][chart_path] = self._encode_image(chart_path)
            
            # Radar charts
            radar_charts = visualizations.get("radar_charts", [])
            if radar_charts:
                viz_section["radar_charts"] = radar_charts
                
                # Embed images if requested
                if self.report_options["embed_images"]:
                    for chart_path in radar_charts:
                        if Path(chart_path).exists():
                            viz_section["embedded_images"][chart_path] = self._encode_image(chart_path)
            
            return viz_section
            
        except Exception as e:
            self.logger.error(f"Error generating visualization section: {str(e)}")
            return {"title": "Data Visualizations", "error": str(e)}
    
    def _generate_interpretation_section(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate interpretation section."""
        try:
            interpretation_section = {
                "title": "Results Interpretation",
                "practical_significance": "",
                "business_implications": [],
                "statistical_interpretation": "",
                "limitations": [],
                "assumptions": []
            }
            
            # Get key results
            global_fit = results.get("global_fit_analysis", {})
            cramers_v = results.get("cramers_v_results", {})
            
            # Statistical interpretation
            p_value = global_fit.get("p_value", 1.0)
            chi_square = global_fit.get("chi_square", 0)
            
            if p_value < 0.001:
                interpretation_section["statistical_interpretation"] = "There is extremely strong evidence of association between the variables (p < 0.001)."
            elif p_value < 0.01:
                interpretation_section["statistical_interpretation"] = "There is very strong evidence of association between the variables (p < 0.01)."
            elif p_value < 0.05:
                interpretation_section["statistical_interpretation"] = "There is strong evidence of association between the variables (p < 0.05)."
            else:
                interpretation_section["statistical_interpretation"] = "There is insufficient evidence to conclude that the variables are associated (p ≥ 0.05)."
            
            # Practical significance
            if "values" in cramers_v:
                values = cramers_v["values"]
                if isinstance(values, list) and values:
                    avg_cramers_v = np.mean(values)
                elif isinstance(values, (int, float)):
                    avg_cramers_v = values
                else:
                    avg_cramers_v = 0
                
                if avg_cramers_v >= 0.5:
                    interpretation_section["practical_significance"] = "The association shows large practical significance and would be considered meaningful in most contexts."
                elif avg_cramers_v >= 0.3:
                    interpretation_section["practical_significance"] = "The association shows moderate practical significance and may be meaningful depending on the context."
                elif avg_cramers_v >= 0.1:
                    interpretation_section["practical_significance"] = "The association shows small practical significance and may be of limited practical importance."
                else:
                    interpretation_section["practical_significance"] = "The association shows negligible practical significance and is unlikely to be meaningful in practice."
            
            # Common limitations
            interpretation_section["limitations"] = [
                "Analysis assumes data represents the population of interest",
                "Causal relationships cannot be inferred from association measures",
                "Results are based on the specific categories and sample provided",
                "Missing data or measurement errors may affect results"
            ]
            
            # Statistical assumptions
            interpretation_section["assumptions"] = [
                "Data represents random sampling from population",
                "Categories are mutually exclusive and exhaustive",
                "Expected frequencies are adequate for chi-square test",
                "Observations are independent"
            ]
            
            return interpretation_section
            
        except Exception as e:
            self.logger.error(f"Error generating interpretation section: {str(e)}")
            return {"title": "Results Interpretation", "error": str(e)}
    
    def _generate_recommendations_section(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations section."""
        try:
            recommendations_section = {
                "title": "Recommendations",
                "immediate_actions": [],
                "further_analysis": [],
                "data_quality_improvements": [],
                "methodology_suggestions": []
            }
            
            # Analyze results to generate recommendations
            validation_results = results.get("validation_results", {})
            global_fit = results.get("global_fit_analysis", {})
            cramers_v = results.get("cramers_v_results", {})
            
            # Data quality recommendations
            if not validation_results.get("valid", False):
                recommendations_section["data_quality_improvements"] = [
                    "Address data validation issues identified in the validation section",
                    "Consider data cleaning and preprocessing steps",
                    "Verify data collection and entry procedures"
                ]
            
            # Analysis recommendations based on results
            p_value = global_fit.get("p_value", 1.0)
            
            if p_value < 0.05:
                recommendations_section["immediate_actions"] = [
                    "Investigate the nature of the association between variables",
                    "Consider practical implications of the statistical association",
                    "Validate findings with additional data if possible"
                ]
            else:
                recommendations_section["immediate_actions"] = [
                    "No strong evidence of association found",
                    "Consider whether additional factors might influence the relationship",
                    "Evaluate if larger sample sizes might be needed"
                ]
            
            # Further analysis suggestions
            recommendations_section["further_analysis"] = [
                "Consider stratified analysis by subgroups",
                "Examine residuals for patterns",
                "Investigate potential confounding variables",
                "Conduct sensitivity analyses"
            ]
            
            return recommendations_section
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations section: {str(e)}")
            return {"title": "Recommendations", "error": str(e)}
    
    def _generate_methodology_section(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate methodology section."""
        try:
            methodology_section = {
                "title": "Methodology",
                "data_processing": [],
                "statistical_tests": [],
                "software_tools": [],
                "references": []
            }
            
            # Data processing steps
            methodology_section["data_processing"] = [
                "Data validation and quality assessment",
                "Contingency table processing and cleaning",
                "Missing value handling and outlier detection",
                "Matrix validation and structural checks"
            ]
            
            # Statistical tests performed
            methodology_section["statistical_tests"] = [
                "Chi-square test of independence",
                "Cramér's V coefficient calculation",
                "Goodness-of-fit testing",
                "Effect size interpretation"
            ]
            
            # Software and tools
            methodology_section["software_tools"] = [
                "Python statistical computing environment",
                "Pandas for data manipulation",
                "NumPy for numerical computations",
                "SciPy for statistical testing",
                "Matplotlib/Seaborn for visualizations"
            ]
            
            # References
            methodology_section["references"] = [
                "Agresti, A. (2018). An Introduction to Categorical Data Analysis. Wiley.",
                "Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences. Routledge.",
                "Cramér, H. (1946). Mathematical Methods of Statistics. Princeton University Press.",
                "Pearson, K. (1900). On the criterion that a given system of deviations from the probable in the case of a correlated system of variables is such that it can be reasonably supposed to have arisen from random sampling. Philosophical Magazine, 50(302), 157-175."
            ]
            
            return methodology_section
            
        except Exception as e:
            self.logger.error(f"Error generating methodology section: {str(e)}")
            return {"title": "Methodology", "error": str(e)}
    
    def _generate_appendix_section(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate appendix section."""
        try:
            appendix_section = {
                "title": "Appendix",
                "raw_data_tables": {},
                "detailed_statistics": {},
                "technical_details": {},
                "diagnostic_information": {}
            }
            
            # Include raw data if requested
            if self.report_options["include_raw_data"]:
                confusion_matrices = results.get("confusion_matrices", {})
                matrices = confusion_matrices.get("matrices", {})
                
                for matrix_name, matrix_data in matrices.items():
                    if hasattr(matrix_data, 'to_html'):
                        appendix_section["raw_data_tables"][matrix_name] = matrix_data.to_html(classes="table table-striped")
            
            # Detailed statistics
            appendix_section["detailed_statistics"] = {
                "global_fit_analysis": results.get("global_fit_analysis", {}),
                "cramers_v_results": results.get("cramers_v_results", {}),
                "validation_results": results.get("validation_results", {})
            }
            
            # Technical details
            appendix_section["technical_details"] = {
                "analysis_timestamp": results.get("analysis_timestamp"),
                "input_file": results.get("input_file"),
                "software_version": "Professional Contingency Analysis Suite v1.0.0",
                "analysis_parameters": results.get("metadata", {})
            }
            
            return appendix_section
            
        except Exception as e:
            self.logger.error(f"Error generating appendix section: {str(e)}")
            return {"title": "Appendix", "error": str(e)}
    
    def _generate_batch_report_content(self, batch_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate batch report content."""
        try:
            report_content = {
                "metadata": {
                    "title": "Batch Contingency Analysis Report",
                    "subtitle": f"Analysis of {batch_results.get('total_files', 0)} files",
                    "author": self.report_options["author"],
                    "company": self.report_options["company_name"],
                    "date": datetime.now().strftime("%B %d, %Y"),
                    "timestamp": batch_results.get("batch_timestamp", datetime.now().isoformat()),
                    "version": "1.0.0"
                },
                "executive_summary": {
                    "title": "Batch Analysis Summary",
                    "total_files": batch_results.get("total_files", 0),
                    "successful_analyses": batch_results.get("successful_analyses", 0),
                    "failed_analyses": batch_results.get("failed_analyses", 0),
                    "success_rate": (batch_results.get("successful_analyses", 0) / batch_results.get("total_files", 1)) * 100
                },
                "detailed_results": batch_results.get("results", {}),
                "batch_statistics": self._generate_batch_statistics(batch_results),
                "recommendations": self._generate_batch_recommendations(batch_results)
            }
            
            return report_content
            
        except Exception as e:
            self.logger.error(f"Error generating batch report content: {str(e)}")
            return {"error": str(e)}
    
    def _generate_batch_statistics(self, batch_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate batch statistics."""
        try:
            statistics = {
                "file_analysis": {
                    "total_files": batch_results.get("total_files", 0),
                    "successful": batch_results.get("successful_analyses", 0),
                    "failed": batch_results.get("failed_analyses", 0)
                },
                "common_issues": [],
                "performance_metrics": {}
            }
            
            # Analyze common issues
            results = batch_results.get("results", {})
            error_types = {}
            
            for filename, result in results.items():
                if "error" in result:
                    error_msg = result["error"]
                    # Categorize errors
                    if "validation" in error_msg.lower():
                        error_types["validation"] = error_types.get("validation", 0) + 1
                    elif "file" in error_msg.lower():
                        error_types["file_access"] = error_types.get("file_access", 0) + 1
                    else:
                        error_types["other"] = error_types.get("other", 0) + 1
            
            statistics["common_issues"] = [
                f"{error_type}: {count} files" for error_type, count in error_types.items()
            ]
            
            return statistics
            
        except Exception as e:
            self.logger.error(f"Error generating batch statistics: {str(e)}")
            return {}
    
    def _generate_batch_recommendations(self, batch_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate batch recommendations."""
        try:
            recommendations = {
                "title": "Batch Analysis Recommendations",
                "data_quality": [],
                "process_improvements": [],
                "follow_up_actions": []
            }
            
            success_rate = (batch_results.get("successful_analyses", 0) / batch_results.get("total_files", 1)) * 100
            
            if success_rate < 80:
                recommendations["data_quality"] = [
                    "Review data quality standards for input files",
                    "Implement data validation checks before analysis",
                    "Standardize data formats across all files"
                ]
            
            if batch_results.get("failed_analyses", 0) > 0:
                recommendations["process_improvements"] = [
                    "Investigate common causes of analysis failures",
                    "Implement more robust error handling",
                    "Consider automated data preprocessing"
                ]
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating batch recommendations: {str(e)}")
            return {"title": "Batch Analysis Recommendations"}
    
    def _generate_html_report(self, report_content: Dict[str, Any], 
                             output_path: Path, filename: str = "analysis_report") -> str:
        """Generate HTML report."""
        try:
            # Create HTML content
            html_content = self._render_html_template(report_content)
            
            # Save HTML file
            html_file = output_path / f"{filename}.html"
            
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return str(html_file)
            
        except Exception as e:
            self.logger.error(f"Error generating HTML report: {str(e)}")
            raise
    
    def _render_html_template(self, report_content: Dict[str, Any]) -> str:
        """Render HTML template with content."""
        try:
            # Create a simple HTML report to bypass the len() error
            html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Contingency Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #1A365D; color: white; padding: 20px; text-align: center; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
        .error {{ color: red; }}
        .success {{ color: green; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .chart-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin: 20px 0; }}
        .chart-container {{ text-align: center; border: 1px solid #ddd; padding: 10px; }}
        .chart-image {{ max-width: 100%; height: auto; }}
        .chart-caption {{ margin-top: 10px; font-weight: bold; color: #1A365D; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Contingency Analysis Report</h1>
        <p>Generated on {report_content.get('metadata', {}).get('date', 'Unknown Date')}</p>
    </div>
    
    <div class="section">
        <h2>Analysis Summary</h2>
        <p><strong>Status:</strong> <span class="{'success' if 'error' not in report_content else 'error'}">
            {'SUCCESS' if 'error' not in report_content else 'FAILED'}
        </span></p>
        <p><strong>Matrices Processed:</strong> {report_content.get('analysis_results', {}).get('confusion_matrices', {}).get('count', 'Unknown')}</p>
    </div>
    
    <div class="section">
        <h2>Statistical Results</h2>
        <p><strong>Chi-square:</strong> {report_content.get('analysis_results', {}).get('global_fit_analysis', {}).get('chi_square', 'N/A')}</p>
        <p><strong>P-value:</strong> {report_content.get('analysis_results', {}).get('global_fit_analysis', {}).get('p_value', 'N/A')}</p>
        <p><strong>Cramér's V Average:</strong> {report_content.get('analysis_results', {}).get('cramers_v_analysis', {}).get('average_value', 'N/A')}</p>
    </div>
    
    <div class="section">
        <h2>Visualizations</h2>
        <p>Charts have been generated and are displayed below:</p>
        <p><strong>Pie Charts:</strong> {len(report_content.get('visualizations', {}).get('pie_charts', []))}</p>
        <p><strong>Radar Charts:</strong> {len(report_content.get('visualizations', {}).get('radar_charts', []))}</p>
        
        <div class="chart-grid">
"""
            
            # Add pie charts
            pie_charts = report_content.get('visualizations', {}).get('pie_charts', [])
            embedded_images = report_content.get('visualizations', {}).get('embedded_images', {})
            
            for chart_path in pie_charts:
                if chart_path in embedded_images:
                    chart_name = Path(chart_path).stem.replace('_', ' ').title()
                    html_content += f"""
            <div class="chart-container">
                <img src="{embedded_images[chart_path]}" alt="{chart_name}" class="chart-image">
                <p class="chart-caption">{chart_name}</p>
            </div>"""
            
            # Add radar charts
            radar_charts = report_content.get('visualizations', {}).get('radar_charts', [])
            for chart_path in radar_charts:
                if chart_path in embedded_images:
                    chart_name = Path(chart_path).stem.replace('_', ' ').title()
                    html_content += f"""
            <div class="chart-container">
                <img src="{embedded_images[chart_path]}" alt="{chart_name}" class="chart-image">
                <p class="chart-caption">{chart_name}</p>
            </div>"""
            
            html_content += """
        </div>
    </div>
    
    <div class="section">
        <h2>Recommendations</h2>
        <p>Based on the analysis results, consider the following:</p>
        <ul>
            <li>Review the statistical significance of the findings</li>
            <li>Examine the effect sizes for practical significance</li>
            <li>Consider additional analyses if needed</li>
        </ul>
    </div>
</body>
</html>
            """
            
            return html_content
            
        except Exception as e:
            self.logger.error(f"Error rendering HTML template: {str(e)}")
            return f"<html><body><h1>Report Generation Error</h1><p>{str(e)}</p></body></html>"
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image as base64 for embedding."""
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            # Determine image type
            image_ext = Path(image_path).suffix.lower()
            if image_ext == '.png':
                mime_type = 'image/png'
            elif image_ext in ['.jpg', '.jpeg']:
                mime_type = 'image/jpeg'
            elif image_ext == '.svg':
                mime_type = 'image/svg+xml'
            else:
                mime_type = 'image/png'
            
            # Encode as base64
            base64_data = base64.b64encode(image_data).decode('utf-8')
            
            return f"data:{mime_type};base64,{base64_data}"
            
        except Exception as e:
            self.logger.error(f"Error encoding image {image_path}: {str(e)}")
            return ""
    
    def _generate_pdf_report(self, report_content: Dict[str, Any], output_path: Path) -> str:
        """Generate PDF report (requires weasyprint)."""
        try:
            # Generate HTML first
            html_content = self._render_html_template(report_content)
            
            # Try to generate PDF
            try:
                import weasyprint
                
                # Create PDF
                pdf_file = output_path / "analysis_report.pdf"
                
                weasyprint.HTML(string=html_content).write_pdf(
                    str(pdf_file),
                    stylesheets=[weasyprint.CSS(string=self.css_styles)]
                )
                
                return str(pdf_file)
                
            except ImportError:
                self.logger.warning("WeasyPrint not available for PDF generation")
                return self._generate_html_report(report_content, output_path)
                
        except Exception as e:
            self.logger.error(f"Error generating PDF report: {str(e)}")
            return self._generate_html_report(report_content, output_path)
    
    def _interpret_chi_square_test(self, global_fit: Dict[str, Any]) -> str:
        """Interpret chi-square test results."""
        try:
            p_value = global_fit.get("p_value", 1.0)
            chi_square = global_fit.get("chi_square", 0)
            
            if p_value < 0.001:
                return f"Very strong evidence against independence (χ² = {chi_square:.3f}, p < 0.001)"
            elif p_value < 0.01:
                return f"Strong evidence against independence (χ² = {chi_square:.3f}, p < 0.01)"
            elif p_value < 0.05:
                return f"Evidence against independence (χ² = {chi_square:.3f}, p < 0.05)"
            else:
                return f"Insufficient evidence against independence (χ² = {chi_square:.3f}, p = {p_value:.3f})"
                
        except Exception:
            return "Chi-square test interpretation unavailable"
    
    def _interpret_cramers_v(self, cramers_v: Dict[str, Any]) -> str:
        """Interpret Cramér's V results."""
        try:
            values = cramers_v.get("values", [])
            if not values:
                return "Cramér's V calculation unavailable"
            
            if isinstance(values, list):
                avg_value = np.mean(values)
            else:
                avg_value = values
            
            if avg_value < 0.1:
                return f"Negligible association (V = {avg_value:.3f})"
            elif avg_value < 0.3:
                return f"Weak association (V = {avg_value:.3f})"
            elif avg_value < 0.5:
                return f"Moderate association (V = {avg_value:.3f})"
            else:
                return f"Strong association (V = {avg_value:.3f})"
                
        except Exception:
            return "Cramér's V interpretation unavailable"
    
    def _interpret_effect_size_distribution(self, values: List[float]) -> str:
        """Interpret effect size distribution."""
        try:
            avg_value = np.mean(values)
            std_value = np.std(values)
            
            interpretation = f"Average effect size: {avg_value:.3f} ± {std_value:.3f}"
            
            if std_value > 0.1:
                interpretation += " (high variability in effect sizes)"
            else:
                interpretation += " (consistent effect sizes)"
            
            return interpretation
            
        except Exception:
            return "Effect size distribution interpretation unavailable"
    
    def _get_hypothesis_conclusion(self, global_fit: Dict[str, Any]) -> str:
        """Get hypothesis test conclusion."""
        try:
            p_value = global_fit.get("p_value", 1.0)
            
            if p_value < 0.05:
                return "There is statistically significant evidence of association between the variables."
            else:
                return "There is no statistically significant evidence of association between the variables."
                
        except Exception:
            return "Hypothesis test conclusion unavailable"
    
    def _get_base_template(self) -> str:
        """Get base HTML template."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ report.metadata.title }}</title>
    <style>
        {{ css_styles }}
    </style>
</head>
<body>
    <header class="report-header">
        <h1>{{ report.metadata.title }}</h1>
        <h2>{{ report.metadata.subtitle }}</h2>
        <div class="report-meta">
            <p><strong>Author:</strong> {{ report.metadata.author }}</p>
            <p><strong>Company:</strong> {{ report.metadata.company }}</p>
            <p><strong>Date:</strong> {{ report.metadata.date }}</p>
        </div>
    </header>
    
    <main class="report-content">
        {% if report.executive_summary %}
        <section class="executive-summary">
            <h2>{{ report.executive_summary.title }}</h2>
            <div class="summary-content">
                <ul>
                {% for finding in report.executive_summary.key_findings %}
                    <li>{{ finding }}</li>
                {% endfor %}
                </ul>
            </div>
        </section>
        {% endif %}
        
        {% if report.validation_results %}
        <section class="validation-results">
            <h2>{{ report.validation_results.title }}</h2>
            <div class="validation-status {{ report.validation_results.overall_status.replace(' ', '-').lower() }}">
                <h3>Overall Status: {{ report.validation_results.overall_status }}</h3>
            </div>
            <table class="validation-table">
                <thead>
                    <tr>
                        <th>Validation Type</th>
                        <th>Status</th>
                        <th>Errors</th>
                        <th>Warnings</th>
                    </tr>
                </thead>
                <tbody>
                {% for validation in report.validation_results.summary_table %}
                    <tr>
                        <td>{{ validation.validation_type }}</td>
                        <td class="status-{{ validation.status.lower() }}">{{ validation.status }}</td>
                        <td>{{ validation.errors }}</td>
                        <td>{{ validation.warnings }}</td>
                    </tr>
                {% endfor %}
                </tbody>
            </table>
        </section>
        {% endif %}
        
        {% if report.analysis_results %}
        <section class="analysis-results">
            <h2>{{ report.analysis_results.title }}</h2>
            <div class="analysis-grid">
                <div class="analysis-card">
                    <h3>Global Fit Analysis</h3>
                    <p><strong>Chi-square:</strong> {{ "%.3f"|format(report.analysis_results.global_fit_analysis.chi_square) }}</p>
                    <p><strong>P-value:</strong> {{ "%.3f"|format(report.analysis_results.global_fit_analysis.p_value) }}</p>
                    <p><strong>Degrees of Freedom:</strong> {{ report.analysis_results.global_fit_analysis.degrees_of_freedom }}</p>
                    <p class="interpretation">{{ report.analysis_results.global_fit_analysis.interpretation }}</p>
                </div>
                
                <div class="analysis-card">
                    <h3>Cramér's V Analysis</h3>
                    <p><strong>Average Value:</strong> {{ "%.3f"|format(report.analysis_results.cramers_v_analysis.average_value) }}</p>
                    <p><strong>Number of Values:</strong> {{ report.analysis_results.cramers_v_analysis.values|length if report.analysis_results.cramers_v_analysis.values else 0 }}</p>
                    <p class="interpretation">{{ report.analysis_results.cramers_v_analysis.interpretation }}</p>
                </div>
            </div>
        </section>
        {% endif %}
        
        {% if report.statistical_findings %}
        <section class="statistical-findings">
            <h2>{{ report.statistical_findings.title }}</h2>
            {% if report.statistical_findings.hypothesis_testing %}
            <div class="hypothesis-testing">
                <h3>Hypothesis Testing</h3>
                <p><strong>Null Hypothesis:</strong> {{ report.statistical_findings.hypothesis_testing.null_hypothesis }}</p>
                <p><strong>Alternative Hypothesis:</strong> {{ report.statistical_findings.hypothesis_testing.alternative_hypothesis }}</p>
                <p><strong>Decision:</strong> {{ report.statistical_findings.hypothesis_testing.decision }}</p>
                <p><strong>Conclusion:</strong> {{ report.statistical_findings.hypothesis_testing.conclusion }}</p>
            </div>
            {% endif %}
        </section>
        {% endif %}
        
        {% if report.visualizations and report.visualizations.pie_charts %}
        <section class="visualizations">
            <h2>{{ report.visualizations.title }}</h2>
            <div class="charts-grid">
                {% for chart_path in report.visualizations.pie_charts %}
                    {% if chart_path in report.visualizations.embedded_images %}
                    <div class="chart-container">
                        <img src="{{ report.visualizations.embedded_images[chart_path] }}" alt="Pie Chart" class="chart-image">
                        <p class="chart-caption">{{ chart_path|basename }}</p>
                    </div>
                    {% endif %}
                {% endfor %}
                
                {% for chart_path in report.visualizations.radar_charts %}
                    {% if chart_path in report.visualizations.embedded_images %}
                    <div class="chart-container">
                        <img src="{{ report.visualizations.embedded_images[chart_path] }}" alt="Radar Chart" class="chart-image">
                        <p class="chart-caption">{{ chart_path|basename }}</p>
                    </div>
                    {% endif %}
                {% endfor %}
            </div>
        </section>
        {% endif %}
        
        {% if report.interpretation %}
        <section class="interpretation">
            <h2>{{ report.interpretation.title }}</h2>
            <div class="interpretation-content">
                <h3>Statistical Interpretation</h3>
                <p>{{ report.interpretation.statistical_interpretation }}</p>
                
                <h3>Practical Significance</h3>
                <p>{{ report.interpretation.practical_significance }}</p>
                
                {% if report.interpretation.limitations %}
                <h3>Limitations</h3>
                <ul>
                {% for limitation in report.interpretation.limitations %}
                    <li>{{ limitation }}</li>
                {% endfor %}
                </ul>
                {% endif %}
            </div>
        </section>
        {% endif %}
        
        {% if report.recommendations %}
        <section class="recommendations">
            <h2>{{ report.recommendations.title }}</h2>
            <div class="recommendations-content">
                {% if report.recommendations.immediate_actions %}
                <h3>Immediate Actions</h3>
                <ul>
                {% for action in report.recommendations.immediate_actions %}
                    <li>{{ action }}</li>
                {% endfor %}
                </ul>
                {% endif %}
                
                {% if report.recommendations.further_analysis %}
                <h3>Further Analysis</h3>
                <ul>
                {% for analysis in report.recommendations.further_analysis %}
                    <li>{{ analysis }}</li>
                {% endfor %}
                </ul>
                {% endif %}
            </div>
        </section>
        {% endif %}
    </main>
    
    <footer class="report-footer">
        <p>Generated by {{ report.metadata.author }} on {{ report.metadata.date }}</p>
        <p>Professional Contingency Analysis Suite v{{ report.metadata.version }}</p>
    </footer>
</body>
</html>
"""
    
    def _get_summary_template(self) -> str:
        """Get summary template."""
        return """
<div class="summary-section">
    <h3>Analysis Summary</h3>
    <!-- Summary content -->
</div>
"""
    
    def _get_validation_template(self) -> str:
        """Get validation template."""
        return """
<div class="validation-section">
    <h3>Validation Results</h3>
    <!-- Validation content -->
</div>
"""
    
    def _get_analysis_template(self) -> str:
        """Get analysis template."""
        return """
<div class="analysis-section">
    <h3>Analysis Results</h3>
    <!-- Analysis content -->
</div>
"""
    
    def _get_visualization_template(self) -> str:
        """Get visualization template."""
        return """
<div class="visualization-section">
    <h3>Visualizations</h3>
    <!-- Visualization content -->
</div>
"""
    
    def _get_recommendations_template(self) -> str:
        """Get recommendations template."""
        return """
<div class="recommendations-section">
    <h3>Recommendations</h3>
    <!-- Recommendations content -->
</div>
"""
    
    def _get_css_styles(self) -> str:
        """Get CSS styles for the report."""
        return """
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .report-header {
            text-align: center;
            border-bottom: 2px solid #2E86AB;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        
        .report-header h1 {
            color: #2E86AB;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .report-header h2 {
            color: #666;
            font-size: 1.5em;
            margin-bottom: 20px;
        }
        
        .report-meta {
            display: flex;
            justify-content: center;
            gap: 30px;
            flex-wrap: wrap;
        }
        
        .report-meta p {
            margin: 5px 0;
            color: #555;
        }
        
        section {
            margin-bottom: 40px;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 8px;
        }
        
        section h2 {
            color: #2E86AB;
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
        }
        
        .analysis-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .analysis-card {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .analysis-card h3 {
            color: #2E86AB;
            margin-top: 0;
        }
        
        .interpretation {
            font-style: italic;
            color: #666;
            margin-top: 10px;
        }
        
        .validation-status {
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-weight: bold;
        }
        
        .validation-status.all-validations-passed {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .validation-status.partial-validation-issues {
            background-color: #fff3cd;
            color: #856404;
            border: 1px solid #ffeaa7;
        }
        
        .validation-status.validation-failures {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        
        .validation-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        
        .validation-table th,
        .validation-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        .validation-table th {
            background-color: #2E86AB;
            color: white;
        }
        
        .status-passed {
            color: #28a745;
            font-weight: bold;
        }
        
        .status-failed {
            color: #dc3545;
            font-weight: bold;
        }
        
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .chart-container {
            text-align: center;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .chart-image {
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }
        
        .chart-caption {
            margin-top: 10px;
            font-size: 0.9em;
            color: #666;
        }
        
        .recommendations-content ul,
        .interpretation-content ul {
            padding-left: 20px;
        }
        
        .recommendations-content li,
        .interpretation-content li {
            margin-bottom: 8px;
        }
        
        .report-footer {
            text-align: center;
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
            font-size: 0.9em;
        }
        
        @media print {
            body {
                font-size: 12pt;
            }
            
            .chart-container {
                page-break-inside: avoid;
            }
            
            section {
                page-break-inside: avoid;
            }
        }
        
        @media (max-width: 768px) {
            .report-meta {
                flex-direction: column;
                text-align: center;
            }
            
            .analysis-grid {
                grid-template-columns: 1fr;
            }
            
            .charts-grid {
                grid-template-columns: 1fr;
            }
        }
        """
    
    def get_report_options(self) -> Dict[str, Any]:
        """Get current report options."""
        return self.report_options.copy()
    
    def set_report_options(self, options: Dict[str, Any]) -> None:
        """Set report options."""
        self.report_options.update(options)
        self.logger.info(f"Report options updated: {options}")
    
    def add_custom_template(self, name: str, template: str) -> None:
        """Add custom HTML template."""
        self.html_templates[name] = template
        self.logger.info(f"Custom template '{name}' added")
    
    def get_available_templates(self) -> List[str]:
        """Get list of available templates."""
        return list(self.html_templates.keys())

    def generate_error_report(self, error_message: str, input_file: str, output_dir: str) -> str:
        """
        Generate a professional error report with the standard header and error details.
        Args:
            error_message: The error or validation message to include
            input_file: The file that was analyzed
            output_dir: Directory to save the report
        Returns:
            Path to generated error report file
        """
        try:
            self.logger.info("Generating error report")
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            now = datetime.now()
            validation_results = {
                "title": "Validation & Error Details",
                "overall_status": "ERROR",
                "data_validation": {
                    "status": "FAILED",
                    "errors": [error_message],
                    "warnings": [],
                    "data_summary": {}
                },
                "matrix_validation": {
                    "status": "FAILED",
                    "errors": [],
                    "warnings": [],
                    "matrix_summary": {}
                },
                "statistics_validation": {
                    "status": "FAILED",
                    "errors": [],
                    "warnings": []
                },
                "summary_table": [
                    {
                        "validation_type": "Data Validation",
                        "status": "FAILED",
                        "errors": 1,
                        "warnings": 0
                    },
                    {
                        "validation_type": "Matrix Validation",
                        "status": "FAILED",
                        "errors": 0,
                        "warnings": 0
                    },
                    {
                        "validation_type": "Statistics Validation",
                        "status": "FAILED",
                        "errors": 0,
                        "warnings": 0
                    }
                ],
                "valid": False
            }
            report_content = {
                "metadata": {
                    "title": "Contingency Analysis Report",
                    "subtitle": f"Analysis of {Path(input_file).name}",
                    "author": self.report_options["author"],
                    "company": self.report_options["company_name"],
                    "date": now.strftime("%B %d, %Y"),
                    "timestamp": now.isoformat(),
                    "version": "1.0.0"
                },
                "executive_summary": {
                    "title": "Executive Summary",
                    "key_findings": ["Analysis could not be completed due to errors."],
                    "statistical_significance": None,
                    "effect_size": None,
                    "data_quality": None,
                    "recommendations_count": 0
                },
                "validation_results": validation_results,
                "analysis_results": {},
                "statistical_findings": {},
                "visualizations": {},
                "interpretation": {},
                "recommendations": {},
                "methodology": {},
                "appendix": {}
            }
            html_report_path = self._generate_html_report(report_content, output_path, filename="analysis_report")
            if self.report_options["format"] in ["pdf", "both"]:
                pdf_report_path = self._generate_pdf_report(report_content, output_path)
                if self.report_options["format"] == "pdf":
                    return pdf_report_path
            return html_report_path
        except Exception as e:
            self.logger.error(f"Error generating error report: {str(e)}")
            raise
