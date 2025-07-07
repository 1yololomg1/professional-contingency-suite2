#!/usr/bin/env python3
"""
Professional Contingency Analysis Suite - Fixed GUI Launcher
Handles packaged executable environment properly
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import sys
import traceback
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# Handle packaged executable environment
def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def get_application_path():
    """Get the path where the application is running from"""
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        return os.path.dirname(sys.executable)
    else:
        # Running as script
        return os.path.dirname(os.path.abspath(__file__))

# Set up basic logging for debugging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import required modules with fallbacks
SCIPY_AVAILABLE = False
try:
    from scipy.stats import chi2_contingency
    SCIPY_AVAILABLE = True
    logger.info("SciPy successfully imported")
except ImportError as e:
    logger.error(f"SciPy import failed: {e}")

MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib
    matplotlib.use('TkAgg')  # Set backend before importing pyplot
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    MATPLOTLIB_AVAILABLE = True
    logger.info("Matplotlib successfully imported")
except ImportError as e:
    logger.error(f"Matplotlib import failed: {e}")

class ContingencyAnalysisGUI:
    """Production-ready GUI that handles all environments properly."""
    
    def __init__(self, root):
        self.root = root
        self.app_path = get_application_path()
        self.temp_dir = None
        
        # Initialize variables first
        self.setup_variables()
        
        # Set up crash protection immediately
        self.setup_crash_protection()
        
        # Initialize GUI
        try:
            self.setup_window()
            self.setup_interface()
            
            # Set up working directory
            self.setup_working_directory()
            
            logger.info("GUI initialized successfully")
            self.log_message("Professional Contingency Analysis Suite initialized", "SUCCESS")
            
        except Exception as e:
            self.handle_startup_error(e)
    
    def setup_crash_protection(self):
        """Set up comprehensive crash protection"""
        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            
            error_msg = f"Unexpected error: {str(exc_value)}"
            logger.error(f"Unhandled exception: {error_msg}")
            logger.error(traceback.format_exc())
            
            try:
                messagebox.showerror(
                    "Application Error",
                    f"An unexpected error occurred:\n\n{error_msg}\n\n"
                    "The application will continue running.\n"
                    "Please save your work and restart if issues persist."
                )
            except:
                pass
        
        sys.excepthook = handle_exception
        self.root.report_callback_exception = handle_exception
    
    def handle_startup_error(self, error):
        """Handle errors during startup"""
        error_msg = f"Failed to initialize application: {str(error)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        # Show basic error dialog
        messagebox.showerror(
            "Startup Error",
            f"{error_msg}\n\n"
            "Please check that all required files are present and try again."
        )
        
        # Still try to show basic interface
        try:
            self.setup_minimal_interface()
        except:
            self.root.quit()
    
    def setup_variables(self):
        """Initialize all tkinter variables"""
        self.input_file_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()
        self.progress_var = tk.DoubleVar()
        self.status_var = tk.StringVar(value="Ready for Analysis")
        
        # Set default output directory in user's home
        default_output = os.path.join(os.path.expanduser("~"), "ContingencyAnalysis_Output")
        self.output_dir_var.set(default_output)
        
        # Analysis state
        self.analysis_thread = None
        self.results = None
        self.confusion_matrix = None
        self.statistical_metrics = {}
    
    def setup_window(self):
        """Set up the main window"""
        self.root.title("Professional Contingency Analysis Suite™")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # Configure grid
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
    
    def setup_interface(self):
        """Set up the main interface"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.grid_rowconfigure(4, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Professional Contingency Analysis Suite™", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, pady=(0, 20))
        
        # File selection
        self.setup_file_selection(main_frame)
        
        # Control buttons
        self.setup_control_buttons(main_frame)
        
        # Progress display
        self.setup_progress_display(main_frame)
        
        # Results and log area
        self.setup_results_area(main_frame)
    
    def setup_file_selection(self, parent):
        """Set up file selection interface"""
        file_frame = ttk.LabelFrame(parent, text="File Selection", padding="10")
        file_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        file_frame.grid_columnconfigure(1, weight=1)
        
        # Input file
        ttk.Label(file_frame, text="Excel File:").grid(row=0, column=0, sticky="w", padx=(0, 10))
        
        input_frame = ttk.Frame(file_frame)
        input_frame.grid(row=0, column=1, sticky="ew", pady=5)
        input_frame.grid_columnconfigure(0, weight=1)
        
        self.input_entry = ttk.Entry(input_frame, textvariable=self.input_file_var, width=50)
        self.input_entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        
        ttk.Button(input_frame, text="Browse...", 
                  command=self.browse_input_file).grid(row=0, column=1)
        
        # Output directory
        ttk.Label(file_frame, text="Output Directory:").grid(row=1, column=0, sticky="w", padx=(0, 10))
        
        output_frame = ttk.Frame(file_frame)
        output_frame.grid(row=1, column=1, sticky="ew", pady=5)
        output_frame.grid_columnconfigure(0, weight=1)
        
        self.output_entry = ttk.Entry(output_frame, textvariable=self.output_dir_var, width=50)
        self.output_entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        
        ttk.Button(output_frame, text="Browse...", 
                  command=self.browse_output_dir).grid(row=0, column=1)
    
    def setup_control_buttons(self, parent):
        """Set up control buttons"""
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=2, column=0, pady=10)
        
        self.analyze_btn = ttk.Button(button_frame, text="🔬 Start Analysis", 
                                     command=self.start_analysis, width=20)
        self.analyze_btn.pack(side="left", padx=5)
        
        self.stop_btn = ttk.Button(button_frame, text="⏹ Stop", 
                                  command=self.stop_analysis, state="disabled", width=15)
        self.stop_btn.pack(side="left", padx=5)
    
    def setup_progress_display(self, parent):
        """Set up progress display"""
        progress_frame = ttk.LabelFrame(parent, text="Progress", padding="10")
        progress_frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        progress_frame.grid_columnconfigure(0, weight=1)
        
        # Status label
        status_label = ttk.Label(progress_frame, textvariable=self.status_var)
        status_label.grid(row=0, column=0, sticky="w", pady=(0, 5))
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                           maximum=100, mode='determinate')
        self.progress_bar.grid(row=1, column=0, sticky="ew")
    
    def setup_results_area(self, parent):
        """Set up results and log area"""
        results_frame = ttk.LabelFrame(parent, text="Results & Log", padding="10")
        results_frame.grid(row=4, column=0, sticky="nsew")
        results_frame.grid_rowconfigure(0, weight=1)
        results_frame.grid_columnconfigure(0, weight=1)
        
        # Create notebook for different result views
        self.results_notebook = ttk.Notebook(results_frame)
        self.results_notebook.grid(row=0, column=0, sticky="nsew")
        
        # Log tab
        log_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(log_frame, text="Analysis Log")
        
        log_frame.grid_rowconfigure(0, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, 
                                                 font=('Consolas', 9), wrap=tk.WORD)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        
        # Results tab
        results_display_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(results_display_frame, text="Statistical Results")
        
        results_display_frame.grid_rowconfigure(0, weight=1)
        results_display_frame.grid_columnconfigure(0, weight=1)
        
        self.results_text = scrolledtext.ScrolledText(results_display_frame, height=15, 
                                                     font=('Consolas', 9), wrap=tk.WORD)
        self.results_text.grid(row=0, column=0, sticky="nsew")
    
    def setup_minimal_interface(self):
        """Set up minimal interface if full setup fails"""
        ttk.Label(self.root, text="Professional Contingency Analysis Suite™", 
                 font=('Arial', 16, 'bold')).pack(pady=20)
        
        ttk.Label(self.root, text="Application encountered initialization issues.\n"
                                 "Some features may not be available.", 
                 foreground='red').pack(pady=10)
        
        ttk.Button(self.root, text="Exit", command=self.root.quit).pack(pady=10)
    
    def setup_working_directory(self):
        """Set up working directory for temporary files"""
        try:
            self.temp_dir = tempfile.mkdtemp(prefix="contingency_")
            logger.info(f"Working directory: {self.temp_dir}")
        except Exception as e:
            logger.error(f"Failed to create working directory: {e}")
            self.temp_dir = None
    
    def browse_input_file(self):
        """Browse for input Excel file"""
        try:
            filetypes = [
                ("Excel files", "*.xlsx *.xls"),
                ("Excel 2007+", "*.xlsx"),
                ("Excel 97-2003", "*.xls"),
                ("All files", "*.*")
            ]
            
            filename = filedialog.askopenfilename(
                title="Select Excel Contingency Table",
                filetypes=filetypes,
                defaultextension=".xlsx"
            )
            
            if filename:
                self.input_file_var.set(filename)
                self.log_message(f"Selected file: {os.path.basename(filename)}", "INFO")
                
        except Exception as e:
            self.log_message(f"Error browsing files: {str(e)}", "ERROR")
    
    def browse_output_dir(self):
        """Browse for output directory"""
        try:
            directory = filedialog.askdirectory(
                title="Select Output Directory",
                initialdir=self.output_dir_var.get()
            )
            
            if directory:
                self.output_dir_var.set(directory)
                self.log_message(f"Output directory: {directory}", "INFO")
                
        except Exception as e:
            self.log_message(f"Error browsing directories: {str(e)}", "ERROR")
    
    def start_analysis(self):
        """Start the analysis process"""
        try:
            # Validate inputs
            input_file = self.input_file_var.get().strip()
            if not input_file:
                messagebox.showerror("Error", "Please select an Excel file")
                return
            
            if not os.path.exists(input_file):
                messagebox.showerror("Error", "Selected file does not exist")
                return
            
            output_dir = self.output_dir_var.get().strip()
            if not output_dir:
                messagebox.showerror("Error", "Please select an output directory")
                return
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Update UI state
            self.analyze_btn['state'] = 'disabled'
            self.stop_btn['state'] = 'normal'
            self.progress_var.set(0)
            self.status_var.set("Starting analysis...")
            
            # Start analysis in thread
            self.analysis_thread = threading.Thread(
                target=self.run_analysis, 
                args=(input_file, output_dir),
                daemon=True
            )
            self.analysis_thread.start()
            
            self.log_message("Analysis started", "INFO")
            
        except Exception as e:
            self.log_message(f"Failed to start analysis: {str(e)}", "ERROR")
            self.reset_ui_state()
    
    def run_analysis(self, input_file, output_dir):
        """Run the analysis in a separate thread"""
        try:
            self.log_message("Loading Excel file...", "INFO")
            self.root.after(0, lambda: self.progress_var.set(10))
            
            # Load data using pandas directly
            data = pd.read_excel(input_file, index_col=0)
            self.confusion_matrix = data
            
            self.log_message(f"Data loaded: {data.shape}", "SUCCESS")
            self.root.after(0, lambda: self.progress_var.set(30))
            
            # Calculate statistics if SciPy is available
            if SCIPY_AVAILABLE:
                self.log_message("Calculating statistical metrics...", "INFO")
                self.statistical_metrics = self.calculate_statistics(data)
                self.root.after(0, lambda: self.progress_var.set(60))
            else:
                self.log_message("SciPy not available - basic statistics only", "WARNING")
                self.statistical_metrics = self.calculate_basic_statistics(data)
            
            # Generate report
            self.log_message("Generating report...", "INFO")
            self.root.after(0, lambda: self.progress_var.set(80))
            
            report_path = self.generate_simple_report(data, output_dir)
            
            # Complete
            self.root.after(0, lambda: self.progress_var.set(100))
            self.root.after(0, lambda: self.status_var.set("Analysis completed"))
            
            self.log_message("Analysis completed successfully!", "SUCCESS")
            self.log_message(f"Report saved to: {report_path}", "SUCCESS")
            
            # Display results
            self.root.after(0, self.display_results)
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            self.log_message(error_msg, "ERROR")
            self.log_message(traceback.format_exc(), "DEBUG")
            
        finally:
            self.root.after(0, self.reset_ui_state)
    
    def calculate_statistics(self, data):
        """Calculate statistical metrics with SciPy"""
        try:
            observed = data.values
            chi2_stat, p_value, dof, expected = chi2_contingency(observed)
            n = observed.sum()
            cramers_v = np.sqrt(chi2_stat / (n * (min(observed.shape) - 1)))
            
            return {
                'cramers_v': cramers_v,
                'chi2_statistic': chi2_stat,
                'p_value': p_value,
                'degrees_freedom': dof,
                'sample_size': n,
                'expected_frequencies': expected
            }
        except Exception as e:
            self.log_message(f"Statistics calculation error: {str(e)}", "ERROR")
            return self.calculate_basic_statistics(data)
    
    def calculate_basic_statistics(self, data):
        """Calculate basic statistics without SciPy"""
        observed = data.values
        n = observed.sum()
        
        return {
            'sample_size': n,
            'matrix_shape': observed.shape,
            'total_observations': n,
            'min_value': observed.min(),
            'max_value': observed.max(),
            'mean_value': observed.mean()
        }
    
    def generate_simple_report(self, data, output_dir):
        """Generate a simple HTML report"""
        try:
            report_path = os.path.join(output_dir, "analysis_report.html")
            
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Contingency Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #1A365D; color: white; padding: 20px; text-align: center; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Professional Contingency Analysis Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="section">
        <h2>Data Summary</h2>
        <p><strong>Matrix Shape:</strong> {data.shape[0]} × {data.shape[1]}</p>
        <p><strong>Total Observations:</strong> {data.sum().sum()}</p>
    </div>
    
    <div class="section">
        <h2>Contingency Table</h2>
        {data.to_html(classes="table")}
    </div>
    
    <div class="section">
        <h2>Statistical Results</h2>
"""
            
            if self.statistical_metrics:
                for key, value in self.statistical_metrics.items():
                    if isinstance(value, (int, float)):
                        html_content += f"<p><strong>{key.replace('_', ' ').title()}:</strong> {value:.4f}</p>\n"
                    else:
                        html_content += f"<p><strong>{key.replace('_', ' ').title()}:</strong> {value}</p>\n"
            
            html_content += """
    </div>
</body>
</html>
"""
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return report_path
            
        except Exception as e:
            self.log_message(f"Report generation error: {str(e)}", "ERROR")
            return None
    
    def display_results(self):
        """Display analysis results"""
        try:
            if self.statistical_metrics:
                results_text = "STATISTICAL ANALYSIS RESULTS\n"
                results_text += "=" * 40 + "\n\n"
                
                for key, value in self.statistical_metrics.items():
                    if isinstance(value, (int, float)):
                        results_text += f"{key.replace('_', ' ').title()}: {value:.4f}\n"
                    else:
                        results_text += f"{key.replace('_', ' ').title()}: {value}\n"
                
                if self.confusion_matrix is not None:
                    results_text += f"\nData Matrix Shape: {self.confusion_matrix.shape}\n"
                    results_text += f"Total Observations: {self.confusion_matrix.sum().sum()}\n"
                
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(1.0, results_text)
                
                # Switch to results tab
                self.results_notebook.select(1)
                
        except Exception as e:
            self.log_message(f"Error displaying results: {str(e)}", "ERROR")
    
    def stop_analysis(self):
        """Stop the running analysis"""
        self.log_message("Analysis stop requested", "WARNING")
        self.reset_ui_state()
    
    def reset_ui_state(self):
        """Reset UI to ready state"""
        self.analyze_btn['state'] = 'normal'
        self.stop_btn['state'] = 'disabled'
        self.status_var.set("Ready for Analysis")
        self.progress_var.set(0)
    
    def log_message(self, message, level="INFO"):
        """Log a message to the log display"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            formatted_message = f"[{timestamp}] [{level}] {message}\n"
            
            if hasattr(self, 'log_text'):
                self.log_text.insert(tk.END, formatted_message)
                self.log_text.see(tk.END)
            
            # Also log to system logger
            if level == "ERROR":
                logger.error(message)
            elif level == "WARNING":
                logger.warning(message)
            else:
                logger.info(message)
                
        except Exception:
            pass  # Don't let logging errors crash the app
    
    def __del__(self):
        """Cleanup when the object is destroyed"""
        try:
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
        except:
            pass

def main():
    """Main application entry point"""
    try:
        # Create application
        root = tk.Tk()
        app = ContingencyAnalysisGUI(root)
        
        # Run application
        root.mainloop()
        
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        logger.error(traceback.format_exc())
        
        # Show basic error message
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                "Application Error",
                f"Failed to start application:\n{str(e)}\n\n"
                "Please check the installation and try again."
            )
        except:
            print(f"FATAL ERROR: {e}")

if __name__ == "__main__":
    main()
