#!/usr/bin/env python3
"""
Professional Contingency Analysis Suite
Main Analysis Engine
Version: 1.0.0
License: Commercial
"""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Import validation modules
from validators.data_validator import DataValidator
from validators.matrix_validator import MatrixValidator
from validators.stats_validator import StatisticsValidator

# Import analysis modules
from analysis.contingency_processor import ContingencyProcessor
from analysis.confusion_matrix_converter import ConfusionMatrixConverter
from analysis.global_fit_analyzer import GlobalFitAnalyzer
from analysis.cramers_v_calculator import CramersVCalculator

# Import visualization modules
from visualization.pie_chart_generator import PieChartGenerator
from visualization.radar_chart_generator import RadarChartGenerator

# Import utilities
from utils.excel_reader import ExcelReader
from utils.config_manager import ConfigManager
from utils.logger_setup import setup_logger
from utils.report_generator import ReportGenerator


class ContingencyAnalyzer:
    """
    Main analysis engine for professional contingency table analysis.
    
    Processes Excel files containing contingency squares, converts to confusion matrices,
    performs global fit analysis, calculates Cramér's V, and generates visualizations.
    """
    
    def __init__(self, config_path: str = "config/analysis_config.json"):
        """Initialize the analyzer with configuration."""
        self.config = ConfigManager(config_path)
        self.logger = setup_logger(
            name="ContingencyAnalyzer",
            log_file=self.config.get("logging.log_file", "logs/analysis.log"),
            level=self.config.get("logging.level", "INFO")
        )
        
        # Initialize validators
        self.data_validator = DataValidator()
        self.matrix_validator = MatrixValidator()
        self.stats_validator = StatisticsValidator()
        
        # Initialize processors
        self.contingency_processor = ContingencyProcessor()
        self.confusion_converter = ConfusionMatrixConverter()
        self.global_fit_analyzer = GlobalFitAnalyzer()
        self.cramers_v_calculator = CramersVCalculator()
        
        # Initialize visualization components
        self.pie_generator = PieChartGenerator()
        self.radar_generator = RadarChartGenerator()
        
        # Initialize utilities
        self.excel_reader = ExcelReader()
        self.report_generator = ReportGenerator()
        
        self.logger.info("ContingencyAnalyzer initialized successfully")
    
    def analyze_excel_file(self, file_path: str, output_dir: str = "output", correction_method: str = "None") -> Dict[str, Any]:
        """
        Analyze an Excel file containing contingency tables.
        
        Args:
            file_path: Path to the Excel file
            output_dir: Directory for output files
            correction_method: Statistical correction method ("None", "Yates", "Bonferroni")
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            self.logger.info(f"Starting analysis of {file_path}")
            validation_issues = []
            validation_status = "PASSED"
            
            # Step 1: Read Excel file
            self.logger.info("Step 1: Reading Excel file")
            try:
                excel_data = self.excel_reader.read_contingency_file(file_path)
            except Exception as e:
                raise ValueError(f"Failed to read Excel file: {str(e)}")
            
            # Step 2: Validate input data
            self.logger.info("Step 2: Validating input data")
            try:
                validation_results = self.data_validator.validate_contingency_data(excel_data)
                if not validation_results["valid"]:
                    validation_issues.extend(validation_results["errors"])
            except Exception as e:
                validation_issues.append(f"Data validation error: {str(e)}")
            
            # Step 3: Process contingency squares
            self.logger.info("Step 3: Processing contingency squares")
            try:
                processed_data = self.contingency_processor.process_contingency_squares(excel_data)
            except Exception as e:
                raise ValueError(f"Failed to process contingency squares: {str(e)}")
            
            # Step 4: Convert to confusion matrices
            self.logger.info("Step 4: Converting to confusion matrices")
            try:
                confusion_matrices = self.confusion_converter.convert_to_confusion_matrices(processed_data)
            except Exception as e:
                raise ValueError(f"Failed to convert to confusion matrices: {str(e)}")
            
            # Step 5: Validate confusion matrices
            self.logger.info("Step 5: Validating confusion matrices")
            try:
                matrix_validation = self.matrix_validator.validate_confusion_matrices(confusion_matrices)
                if not matrix_validation["valid"]:
                    validation_issues.extend(matrix_validation["errors"])
            except Exception as e:
                matrix_validation = {"valid": False, "errors": [str(e)], "warnings": []}
                validation_issues.append(f"Matrix validation error: {str(e)}")
                validation_status = "FAILED"
            
            # Step 6: Global fit analysis with correction method
            self.logger.info(f"Step 6: Performing global fit analysis (correction: {correction_method})")
            try:
                # Apply correction method to global fit analyzer
                if correction_method == "Yates":
                    self.global_fit_analyzer.analysis_options["continuity_correction"] = True
                else:
                    self.global_fit_analyzer.analysis_options["continuity_correction"] = False
                
                global_fit_results = self.global_fit_analyzer.analyze_global_fit(confusion_matrices)
                
                # Apply Bonferroni correction if requested
                if correction_method == "Bonferroni":
                    self._apply_bonferroni_correction(global_fit_results)
                    
            except Exception as e:
                global_fit_results = {}
                validation_issues.append(f"Global fit analysis error: {str(e)}")
                validation_status = "FAILED"
            
            # Step 7: Calculate Cramér's V with correction method
            self.logger.info(f"Step 7: Calculating Cramér's V (correction: {correction_method})")
            try:
                # Apply bias correction for Cramér's V if requested
                if correction_method in ["Yates", "Bonferroni"]:
                    self.cramers_v_calculator.calculation_options["bias_correction"] = True
                    self.cramers_v_calculator.calculation_options["small_sample_correction"] = True
                else:
                    self.cramers_v_calculator.calculation_options["bias_correction"] = False
                    self.cramers_v_calculator.calculation_options["small_sample_correction"] = False
                
                cramers_v_results = self.cramers_v_calculator.calculate_cramers_v(confusion_matrices)
            except Exception as e:
                cramers_v_results = {}
                validation_issues.append(f"Cramér's V calculation error: {str(e)}")
                validation_status = "FAILED"
            
            # Step 8: Statistics validation
            self.logger.info("Step 8: Validating statistical results")
            try:
                stats_validation = self.stats_validator.validate_statistics(global_fit_results, cramers_v_results)
                if not stats_validation["valid"]:
                    validation_issues.append(f"Statistics validation failed: {stats_validation['errors']}")
                    validation_status = "FAILED"
            except Exception as e:
                stats_validation = {"valid": False, "errors": [str(e)], "warnings": []}
                validation_issues.append(f"Statistics validation error: {str(e)}")
                validation_status = "FAILED"
            
            # Step 9: Generate visualizations (always attempt)
            self.logger.info("Step 9: Generating visualizations")
            try:
                pie_charts = self.pie_generator.generate_pie_charts(
                    confusion_matrices, 
                    output_dir=os.path.join(output_dir, "pie_charts")
                )
            except Exception as e:
                pie_charts = []
                validation_issues.append(f"Pie chart generation error: {str(e)}")
                validation_status = "FAILED"
            try:
                radar_charts = self.radar_generator.generate_radar_charts(
                    global_fit_results,
                    cramers_v_results,
                    output_dir=os.path.join(output_dir, "radar_charts")
                )
            except Exception as e:
                radar_charts = []
                validation_issues.append(f"Radar chart generation error: {str(e)}")
                validation_status = "FAILED"
            
            # Step 10: Compile results
            results = {
                "analysis_timestamp": datetime.now().isoformat(),
                "input_file": file_path,
                "correction_method": correction_method,
                "validation_results": validation_results,
                "matrix_validation": matrix_validation,
                "stats_validation": stats_validation,
                "contingency_data": processed_data,
                "confusion_matrices": confusion_matrices,
                "global_fit_analysis": global_fit_results,
                "cramers_v_results": cramers_v_results,
                "visualizations": {
                    "pie_charts": pie_charts,
                    "radar_charts": radar_charts
                },
                "validation_status": validation_status,
                "validation_issues": validation_issues
            }
            
            # Step 11: Generate comprehensive report
            self.logger.info("Step 11: Generating comprehensive report")
            try:
                report_path = self.report_generator.generate_report(results, output_dir)
                results["report_path"] = report_path
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                self.logger.error(f"Report generation failed: {str(e)}\n{tb}")
            
            self.logger.info(f"Analysis completed with status: {validation_status}. Results saved to {output_dir}")
            return results
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            # Always generate a professional error report
            report_path = self.report_generator.generate_error_report(str(e), file_path, output_dir)
            return {
                "analysis_timestamp": datetime.now().isoformat(),
                "input_file": file_path,
                "correction_method": correction_method,
                "error": str(e),
                "report_path": report_path,
                "validation_status": "FAILED",
                "validation_issues": validation_issues if 'validation_issues' in locals() else [str(e)]
            }
    
    def _apply_bonferroni_correction(self, global_fit_results: Dict[str, Any]) -> None:
        """Apply Bonferroni correction to global fit results."""
        try:
            # Count total number of statistical tests
            total_tests = 0
            for matrix_name, matrix_results in global_fit_results.get("test_results", {}).items():
                total_tests += len(matrix_results)
            
            if total_tests > 1:
                bonferroni_factor = total_tests
                
                # Apply correction to all p-values
                for matrix_name, matrix_results in global_fit_results.get("test_results", {}).items():
                    for test_name, test_result in matrix_results.items():
                        if isinstance(test_result, dict) and "p_value" in test_result:
                            original_p = test_result["p_value"]
                            adjusted_p = min(original_p * bonferroni_factor, 1.0)
                            test_result["p_value"] = adjusted_p
                            test_result["p_value_original"] = original_p
                            test_result["bonferroni_factor"] = bonferroni_factor
                            test_result["significant"] = adjusted_p < 0.05
                
                self.logger.info(f"Applied Bonferroni correction with factor {bonferroni_factor} to {total_tests} tests")
                
        except Exception as e:
            self.logger.error(f"Error applying Bonferroni correction: {str(e)}")
    
    def batch_analyze(self, input_directory: str, output_directory: str = "batch_output") -> Dict[str, Any]:
        """
        Batch analyze multiple Excel files.
        
        Args:
            input_directory: Directory containing Excel files
            output_directory: Directory for batch output
            
        Returns:
            Dictionary containing batch analysis results
        """
        try:
            self.logger.info(f"Starting batch analysis of {input_directory}")
            
            excel_files = list(Path(input_directory).glob("*.xlsx"))
            excel_files.extend(list(Path(input_directory).glob("*.xls")))
            
            if not excel_files:
                raise ValueError(f"No Excel files found in {input_directory}")
            
            batch_results = {
                "batch_timestamp": datetime.now().isoformat(),
                "total_files": len(excel_files),
                "successful_analyses": 0,
                "failed_analyses": 0,
                "results": {}
            }
            
            for excel_file in excel_files:
                try:
                    file_output_dir = os.path.join(output_directory, excel_file.stem)
                    result = self.analyze_excel_file(str(excel_file), file_output_dir)
                    batch_results["results"][excel_file.name] = result
                    batch_results["successful_analyses"] += 1
                    
                except Exception as e:
                    self.logger.error(f"Failed to analyze {excel_file}: {str(e)}")
                    batch_results["results"][excel_file.name] = {"error": str(e)}
                    batch_results["failed_analyses"] += 1
            
            # Generate batch summary report
            batch_report_path = self.report_generator.generate_batch_report(
                batch_results, output_directory
            )
            batch_results["batch_report_path"] = batch_report_path
            
            self.logger.info(f"Batch analysis completed. {batch_results['successful_analyses']}/{batch_results['total_files']} files processed successfully")
            return batch_results
            
        except Exception as e:
            self.logger.error(f"Batch analysis failed: {str(e)}")
            raise
    
    def get_analysis_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a summary of analysis results.
        
        Args:
            results: Analysis results dictionary
            
        Returns:
            Summary dictionary
        """
        try:
            summary = {
                "file_analyzed": results.get("input_file"),
                "analysis_date": results.get("analysis_timestamp"),
                "validation_status": results.get("validation_status"),
                "matrices_processed": len(results.get("confusion_matrices", [])),
                "global_fit_metrics": {
                    "chi_square": results.get("global_fit_analysis", {}).get("chi_square"),
                    "p_value": results.get("global_fit_analysis", {}).get("p_value"),
                    "degrees_of_freedom": results.get("global_fit_analysis", {}).get("degrees_of_freedom")
                },
                "cramers_v_values": results.get("cramers_v_results", {}).get("values", []),
                "visualizations_generated": {
                    "pie_charts": len(results.get("visualizations", {}).get("pie_charts", [])),
                    "radar_charts": len(results.get("visualizations", {}).get("radar_charts", []))
                }
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to generate summary: {str(e)}")
            raise


def main():
    """Command-line interface for the contingency analyzer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Professional Contingency Analysis Suite")
    parser.add_argument("input_file", help="Path to Excel file containing contingency squares")
    parser.add_argument("-o", "--output", default="output", help="Output directory")
    parser.add_argument("-c", "--config", default="config/analysis_config.json", help="Configuration file")
    parser.add_argument("-b", "--batch", action="store_true", help="Batch process directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    try:
        analyzer = ContingencyAnalyzer(args.config)
        
        if args.batch:
            results = analyzer.batch_analyze(args.input_file, args.output)
            print(f"Batch analysis completed: {results['successful_analyses']}/{results['total_files']} files processed")
        else:
            results = analyzer.analyze_excel_file(args.input_file, args.output)
            summary = analyzer.get_analysis_summary(results)
            
            print("Analysis Summary:")
            print(f"  File: {summary['file_analyzed']}")
            print(f"  Status: {summary['validation_status']}")
            print(f"  Matrices: {summary['matrices_processed']}")
            print(f"  Cramér's V: {summary['cramers_v_values']}")
            print(f"  Report: {results['report_path']}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
