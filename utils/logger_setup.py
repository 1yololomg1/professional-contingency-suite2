#!/usr/bin/env python3
"""
Logger Setup Utility Module
Professional Contingency Analysis Suite
Provides comprehensive logging setup with professional formatting and features
"""

import logging
import logging.handlers
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path
import json
import sys
import os
import traceback


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        """Format log record with colors."""
        if hasattr(record, 'levelname'):
            level_color = self.COLORS.get(record.levelname, '')
            reset_color = self.COLORS['RESET']
            
            # Color the level name
            record.levelname = f"{level_color}{record.levelname}{reset_color}"
        
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'session_id'):
            log_entry['session_id'] = record.session_id
        if hasattr(record, 'analysis_id'):
            log_entry['analysis_id'] = record.analysis_id
        
        return json.dumps(log_entry)


class AnalysisFilter(logging.Filter):
    """Custom filter for analysis-specific logging."""
    
    def __init__(self, analysis_id: Optional[str] = None):
        super().__init__()
        self.analysis_id = analysis_id
    
    def filter(self, record):
        """Filter log records based on analysis context."""
        # Add analysis ID if available
        if self.analysis_id:
            record.analysis_id = self.analysis_id
        
        # Filter out sensitive information
        if hasattr(record, 'args') and record.args:
            # Check for potential sensitive data patterns
            sensitive_patterns = ['password', 'token', 'key', 'secret']
            message = record.getMessage().lower()
            
            for pattern in sensitive_patterns:
                if pattern in message:
                    record.msg = "[SENSITIVE DATA FILTERED]"
                    record.args = ()
                    break
        
        return True


class ProfessionalLoggerSetup:
    """
    Professional logging setup for the contingency analysis suite.
    Provides comprehensive logging with multiple handlers and formatters.
    """
    
    def __init__(self):
        self.loggers = {}
        self.handlers = {}
        self.formatters = {}
        self.filters = {}
        
        # Default configuration
        self.default_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S"
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S"
                },
                "console": {
                    "format": "%(asctime)s - %(levelname)s - %(message)s",
                    "datefmt": "%H:%M:%S"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "console",
                    "stream": "ext://sys.stdout"
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "DEBUG",
                    "formatter": "detailed",
                    "filename": "logs/analysis.log",
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 5,
                    "encoding": "utf8"
                },
                "error_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "ERROR",
                    "formatter": "detailed",
                    "filename": "logs/errors.log",
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 5,
                    "encoding": "utf8"
                }
            },
            "loggers": {
                "": {
                    "handlers": ["console", "file", "error_file"],
                    "level": "DEBUG",
                    "propagate": False
                }
            }
        }
    
    def setup_logger(self, name: str, log_file: Optional[str] = None, 
                     level: str = "INFO", console_output: bool = True,
                     json_format: bool = False, analysis_id: Optional[str] = None) -> logging.Logger:
        """
        Setup a professional logger with comprehensive configuration.
        
        Args:
            name: Logger name
            log_file: Path to log file (optional)
            level: Logging level
            console_output: Whether to output to console
            json_format: Whether to use JSON formatting
            analysis_id: Analysis ID for filtering
            
        Returns:
            Configured logger instance
        """
        try:
            # Create logger
            logger = logging.getLogger(name)
            logger.setLevel(getattr(logging, level.upper()))
            
            # Clear existing handlers
            logger.handlers.clear()
            
            # Setup formatters
            if json_format:
                formatter = JSONFormatter()
                console_formatter = ColoredFormatter(
                    fmt='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S'
                )
            else:
                formatter = logging.Formatter(
                    fmt='%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                console_formatter = ColoredFormatter(
                    fmt='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S'
                )
            
            # Setup file handler
            if log_file:
                # Ensure log directory exists
                log_path = Path(log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Rotating file handler
                file_handler = logging.handlers.RotatingFileHandler(
                    log_file,
                    maxBytes=10*1024*1024,  # 10MB
                    backupCount=5,
                    encoding='utf-8'
                )
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(formatter)
                
                # Add analysis filter if specified
                if analysis_id:
                    file_handler.addFilter(AnalysisFilter(analysis_id))
                
                logger.addHandler(file_handler)
                self.handlers[f"{name}_file"] = file_handler
            
            # Setup console handler
            if console_output:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(getattr(logging, level.upper()))
                console_handler.setFormatter(console_formatter)
                
                # Add analysis filter if specified
                if analysis_id:
                    console_handler.addFilter(AnalysisFilter(analysis_id))
                
                logger.addHandler(console_handler)
                self.handlers[f"{name}_console"] = console_handler
            
            # Setup error file handler
            if log_file:
                error_log_file = log_path.parent / "errors.log"
                error_handler = logging.handlers.RotatingFileHandler(
                    error_log_file,
                    maxBytes=10*1024*1024,  # 10MB
                    backupCount=5,
                    encoding='utf-8'
                )
                error_handler.setLevel(logging.ERROR)
                error_handler.setFormatter(formatter)
                
                logger.addHandler(error_handler)
                self.handlers[f"{name}_error"] = error_handler
            
            # Store logger
            self.loggers[name] = logger
            
            logger.info(f"Logger '{name}' configured successfully")
            return logger
            
        except Exception as e:
            # Fallback to basic logging
            basic_logger = logging.getLogger(name)
            basic_logger.setLevel(logging.INFO)
            
            if not basic_logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                basic_logger.addHandler(handler)
            
            basic_logger.error(f"Failed to setup professional logger: {str(e)}")
            return basic_logger
    
    def setup_from_config(self, config: Dict[str, Any]) -> None:
        """Setup logging from configuration dictionary."""
        try:
            # Merge with default config
            merged_config = self._merge_configs(self.default_config, config)
            
            # Apply configuration
            logging.config.dictConfig(merged_config)
            
            # Get root logger
            root_logger = logging.getLogger()
            root_logger.info("Logging configured from configuration dictionary")
            
        except Exception as e:
            logging.error(f"Failed to setup logging from config: {str(e)}")
            raise
    
    def setup_analysis_logger(self, analysis_id: str, output_dir: str = "logs") -> logging.Logger:
        """Setup logger specific to an analysis run."""
        try:
            logger_name = f"analysis_{analysis_id}"
            log_file = f"{output_dir}/analysis_{analysis_id}.log"
            
            logger = self.setup_logger(
                name=logger_name,
                log_file=log_file,
                level="DEBUG",
                console_output=False,
                json_format=True,
                analysis_id=analysis_id
            )
            
            # Add analysis metadata
            logger.info(f"Analysis session started", extra={
                'analysis_id': analysis_id,
                'timestamp': datetime.now().isoformat()
            })
            
            return logger
            
        except Exception as e:
            logging.error(f"Failed to setup analysis logger: {str(e)}")
            raise
    
    def setup_performance_logger(self, output_dir: str = "logs") -> logging.Logger:
        """Setup logger for performance monitoring."""
        try:
            logger_name = "performance"
            log_file = f"{output_dir}/performance.log"
            
            logger = self.setup_logger(
                name=logger_name,
                log_file=log_file,
                level="INFO",
                console_output=False,
                json_format=True
            )
            
            return logger
            
        except Exception as e:
            logging.error(f"Failed to setup performance logger: {str(e)}")
            raise
    
    def setup_audit_logger(self, output_dir: str = "logs") -> logging.Logger:
        """Setup logger for audit trail."""
        try:
            logger_name = "audit"
            log_file = f"{output_dir}/audit.log"
            
            logger = self.setup_logger(
                name=logger_name,
                log_file=log_file,
                level="INFO",
                console_output=False,
                json_format=True
            )
            
            return logger
            
        except Exception as e:
            logging.error(f"Failed to setup audit logger: {str(e)}")
            raise
    
    def log_analysis_start(self, logger: logging.Logger, analysis_params: Dict[str, Any]) -> None:
        """Log the start of an analysis."""
        logger.info("Analysis started", extra={
            'event_type': 'analysis_start',
            'parameters': analysis_params,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_analysis_end(self, logger: logging.Logger, analysis_results: Dict[str, Any]) -> None:
        """Log the end of an analysis."""
        logger.info("Analysis completed", extra={
            'event_type': 'analysis_end',
            'results_summary': analysis_results,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_performance_metric(self, logger: logging.Logger, metric_name: str, 
                              metric_value: Union[int, float], unit: str = "") -> None:
        """Log a performance metric."""
        logger.info(f"Performance metric: {metric_name}", extra={
            'event_type': 'performance_metric',
            'metric_name': metric_name,
            'metric_value': metric_value,
            'unit': unit,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_validation_result(self, logger: logging.Logger, validation_type: str, 
                             validation_result: Dict[str, Any]) -> None:
        """Log validation results."""
        logger.info(f"Validation completed: {validation_type}", extra={
            'event_type': 'validation_result',
            'validation_type': validation_type,
            'validation_result': validation_result,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_error_with_context(self, logger: logging.Logger, error: Exception, 
                              context: Dict[str, Any]) -> None:
        """Log error with additional context."""
        logger.error(f"Error occurred: {str(error)}", extra={
            'event_type': 'error',
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'timestamp': datetime.now().isoformat()
        }, exc_info=True)
    
    def create_log_summary(self, log_file: str) -> Dict[str, Any]:
        """Create a summary of log file contents."""
        try:
            summary = {
                'total_lines': 0,
                'level_counts': {
                    'DEBUG': 0,
                    'INFO': 0,
                    'WARNING': 0,
                    'ERROR': 0,
                    'CRITICAL': 0
                },
                'first_entry': None,
                'last_entry': None,
                'errors': []
            }
            
            if not Path(log_file).exists():
                return summary
            
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                summary['total_lines'] = len(lines)
                
                if lines:
                    summary['first_entry'] = lines[0].strip()
                    summary['last_entry'] = lines[-1].strip()
                    
                    # Count log levels
                    for line in lines:
                        for level in summary['level_counts']:
                            if f" {level} " in line:
                                summary['level_counts'][level] += 1
                                
                                # Collect error messages
                                if level in ['ERROR', 'CRITICAL']:
                                    summary['errors'].append(line.strip())
            
            return summary
            
        except Exception as e:
            return {'error': f"Failed to create log summary: {str(e)}"}
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configuration dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get_logger(self, name: str) -> Optional[logging.Logger]:
        """Get existing logger by name."""
        return self.loggers.get(name)
    
    def list_loggers(self) -> List[str]:
        """List all configured loggers."""
        return list(self.loggers.keys())
    
    def cleanup_old_logs(self, log_dir: str = "logs", days_to_keep: int = 30) -> None:
        """Clean up old log files."""
        try:
            log_path = Path(log_dir)
            
            if not log_path.exists():
                return
            
            current_time = datetime.now()
            cutoff_time = current_time.timestamp() - (days_to_keep * 24 * 60 * 60)
            
            deleted_files = []
            for log_file in log_path.glob("*.log*"):
                if log_file.stat().st_mtime < cutoff_time:
                    log_file.unlink()
                    deleted_files.append(str(log_file))
            
            if deleted_files:
                logging.info(f"Cleaned up {len(deleted_files)} old log files")
            
        except Exception as e:
            logging.error(f"Failed to cleanup old logs: {str(e)}")


# Global logger setup instance
logger_setup = ProfessionalLoggerSetup()


def setup_logger(name: str, log_file: Optional[str] = None, 
                 level: str = "INFO", console_output: bool = True,
                 json_format: bool = False, analysis_id: Optional[str] = None) -> logging.Logger:
    """
    Convenience function to setup a logger.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        console_output: Whether to output to console
        json_format: Whether to use JSON formatting
        analysis_id: Analysis ID for filtering
        
    Returns:
        Configured logger instance
    """
    return logger_setup.setup_logger(name, log_file, level, console_output, json_format, analysis_id)


def get_logger(name: str) -> Optional[logging.Logger]:
    """Get existing logger by name."""
    return logger_setup.get_logger(name)


def setup_from_config(config: Dict[str, Any]) -> None:
    """Setup logging from configuration dictionary."""
    logger_setup.setup_from_config(config)
