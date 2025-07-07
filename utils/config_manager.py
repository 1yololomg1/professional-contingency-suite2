#!/usr/bin/env python3
"""
Configuration Manager Module
Professional Contingency Analysis Suite
Manages configuration settings and provides flexible configuration handling
"""

import json
import yaml
import os
from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime
from pathlib import Path
import copy
from collections import defaultdict


class ConfigManager:
    """
    Professional configuration manager for the contingency analysis suite.
    Supports JSON and YAML configurations with validation and templating.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        self.config_data = {}
        self.config_schema = {}
        self.config_history = []
        self.environment_overrides = {}
        self.default_config = self._get_default_config()
        
        # Load configuration
        if config_path:
            self.load_config(config_path)
        else:
            self.config_data = self.default_config.copy()
    
    def load_config(self, config_path: str) -> None:
        """Load configuration from file."""
        try:
            config_path = Path(config_path)
            
            if not config_path.exists():
                self.logger.warning(f"Config file not found: {config_path}. Using defaults.")
                self.config_data = self.default_config.copy()
                return
            
            # Determine file format
            if config_path.suffix.lower() == '.json':
                self.config_data = self._load_json_config(config_path)
            elif config_path.suffix.lower() in ['.yaml', '.yml']:
                self.config_data = self._load_yaml_config(config_path)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
            
            # Merge with defaults
            self.config_data = self._merge_configs(self.default_config, self.config_data)
            
            # Apply environment overrides
            self._apply_environment_overrides()
            
            # Validate configuration
            validation_result = self._validate_config()
            if not validation_result["valid"]:
                self.logger.warning(f"Configuration validation issues: {validation_result['warnings']}")
            
            # Save to history
            self._save_to_history("load", config_path)
            
            self.logger.info(f"Configuration loaded from {config_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {str(e)}")
            self.config_data = self.default_config.copy()
    
    def _load_json_config(self, config_path: Path) -> Dict[str, Any]:
        """Load JSON configuration file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON configuration: {str(e)}")
    
    def _load_yaml_config(self, config_path: Path) -> Dict[str, Any]:
        """Load YAML configuration file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {str(e)}")
    
    def save_config(self, config_path: Optional[str] = None) -> None:
        """Save current configuration to file."""
        try:
            if config_path is None:
                config_path = self.config_path
            
            if not config_path:
                raise ValueError("No config path specified")
            
            config_path = Path(config_path)
            
            # Create directory if it doesn't exist
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save based on file extension
            if config_path.suffix.lower() == '.json':
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config_data, f, indent=2, ensure_ascii=False)
            elif config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(self.config_data, f, default_flow_style=False, allow_unicode=True)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
            
            # Save to history
            self._save_to_history("save", config_path)
            
            self.logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {str(e)}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        try:
            keys = key.split('.')
            value = self.config_data
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
            
        except Exception as e:
            self.logger.error(f"Error getting config value '{key}': {str(e)}")
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation."""
        try:
            keys = key.split('.')
            config_dict = self.config_data
            
            # Navigate to the parent dictionary
            for k in keys[:-1]:
                if k not in config_dict:
                    config_dict[k] = {}
                config_dict = config_dict[k]
            
            # Set the value
            config_dict[keys[-1]] = value
            
            # Save to history
            self._save_to_history("set", f"{key} = {value}")
            
            self.logger.debug(f"Configuration value set: {key} = {value}")
            
        except Exception as e:
            self.logger.error(f"Error setting config value '{key}': {str(e)}")
            raise
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration with dictionary."""
        try:
            self.config_data = self._merge_configs(self.config_data, config_dict)
            
            # Save to history
            self._save_to_history("update", f"Updated {len(config_dict)} values")
            
            self.logger.info(f"Configuration updated with {len(config_dict)} values")
            
        except Exception as e:
            self.logger.error(f"Error updating configuration: {str(e)}")
            raise
    
    def delete(self, key: str) -> bool:
        """Delete configuration key."""
        try:
            keys = key.split('.')
            config_dict = self.config_data
            
            # Navigate to the parent dictionary
            for k in keys[:-1]:
                if k not in config_dict:
                    return False
                config_dict = config_dict[k]
            
            # Delete the key
            if keys[-1] in config_dict:
                del config_dict[keys[-1]]
                
                # Save to history
                self._save_to_history("delete", key)
                
                self.logger.debug(f"Configuration key deleted: {key}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error deleting config key '{key}': {str(e)}")
            return False
    
    def has(self, key: str) -> bool:
        """Check if configuration key exists."""
        try:
            keys = key.split('.')
            value = self.config_data
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking config key '{key}': {str(e)}")
            return False
    
    def get_section(self, section_key: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        try:
            section = self.get(section_key, {})
            return section if isinstance(section, dict) else {}
            
        except Exception as e:
            self.logger.error(f"Error getting config section '{section_key}': {str(e)}")
            return {}
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration data."""
        return copy.deepcopy(self.config_data)
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to defaults."""
        try:
            self.config_data = self.default_config.copy()
            
            # Save to history
            self._save_to_history("reset", "Reset to defaults")
            
            self.logger.info("Configuration reset to defaults")
            
        except Exception as e:
            self.logger.error(f"Error resetting configuration: {str(e)}")
            raise
    
    def create_profile(self, profile_name: str, config_subset: Optional[Dict[str, Any]] = None) -> None:
        """Create a configuration profile."""
        try:
            if config_subset is None:
                config_subset = self.config_data
            
            profile_dir = Path("config/profiles")
            profile_dir.mkdir(parents=True, exist_ok=True)
            
            profile_path = profile_dir / f"{profile_name}.json"
            
            with open(profile_path, 'w', encoding='utf-8') as f:
                json.dump(config_subset, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Configuration profile '{profile_name}' created")
            
        except Exception as e:
            self.logger.error(f"Error creating profile '{profile_name}': {str(e)}")
            raise
    
    def load_profile(self, profile_name: str) -> None:
        """Load a configuration profile."""
        try:
            profile_path = Path(f"config/profiles/{profile_name}.json")
            
            if not profile_path.exists():
                raise ValueError(f"Profile '{profile_name}' not found")
            
            with open(profile_path, 'r', encoding='utf-8') as f:
                profile_config = json.load(f)
            
            self.config_data = self._merge_configs(self.default_config, profile_config)
            
            # Save to history
            self._save_to_history("load_profile", profile_name)
            
            self.logger.info(f"Configuration profile '{profile_name}' loaded")
            
        except Exception as e:
            self.logger.error(f"Error loading profile '{profile_name}': {str(e)}")
            raise
    
    def list_profiles(self) -> List[str]:
        """List available configuration profiles."""
        try:
            profile_dir = Path("config/profiles")
            
            if not profile_dir.exists():
                return []
            
            profiles = []
            for profile_file in profile_dir.glob("*.json"):
                profiles.append(profile_file.stem)
            
            return sorted(profiles)
            
        except Exception as e:
            self.logger.error(f"Error listing profiles: {str(e)}")
            return []
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge two configuration dictionaries."""
        try:
            result = copy.deepcopy(base)
            
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = self._merge_configs(result[key], value)
                else:
                    result[key] = copy.deepcopy(value)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error merging configurations: {str(e)}")
            return base
    
    def _apply_environment_overrides(self) -> None:
        """Apply environment variable overrides."""
        try:
            # Look for environment variables with CONTINGENCY_ prefix
            for env_var, env_value in os.environ.items():
                if env_var.startswith("CONTINGENCY_"):
                    # Convert CONTINGENCY_SECTION_KEY to section.key
                    config_key = env_var[12:].lower().replace("_", ".")
                    
                    # Try to parse as JSON, fall back to string
                    try:
                        value = json.loads(env_value)
                    except json.JSONDecodeError:
                        value = env_value
                    
                    self.set(config_key, value)
                    self.environment_overrides[config_key] = value
            
            if self.environment_overrides:
                self.logger.info(f"Applied {len(self.environment_overrides)} environment overrides")
                
        except Exception as e:
            self.logger.error(f"Error applying environment overrides: {str(e)}")
    
    def _validate_config(self) -> Dict[str, Any]:
        """Validate configuration against schema."""
        try:
            validation_result = {
                "valid": True,
                "errors": [],
                "warnings": []
            }
            
            # Basic validation rules
            required_sections = ["analysis_settings", "logging", "output"]
            
            for section in required_sections:
                if not self.has(section):
                    validation_result["errors"].append(f"Missing required section: {section}")
                    validation_result["valid"] = False
            
            # Validate specific settings
            if self.has("analysis_settings.output_precision"):
                precision = self.get("analysis_settings.output_precision")
                if not isinstance(precision, int) or precision < 1 or precision > 15:
                    validation_result["warnings"].append("output_precision should be between 1 and 15")
            
            if self.has("logging.level"):
                level = self.get("logging.level")
                valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
                if level not in valid_levels:
                    validation_result["warnings"].append(f"Invalid logging level: {level}")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error validating configuration: {str(e)}")
            return {
                "valid": False,
                "errors": [f"Validation error: {str(e)}"],
                "warnings": []
            }
    
    def _save_to_history(self, action: str, details: str) -> None:
        """Save configuration change to history."""
        try:
            history_entry = {
                "timestamp": datetime.now().isoformat(),
                "action": action,
                "details": details
            }
            
            self.config_history.append(history_entry)
            
            # Keep only last 100 entries
            if len(self.config_history) > 100:
                self.config_history = self.config_history[-100:]
                
        except Exception as e:
            self.logger.error(f"Error saving to history: {str(e)}")
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get configuration change history."""
        return copy.deepcopy(self.config_history)
    
    def export_config(self, export_path: str, format: str = "json") -> None:
        """Export configuration to file."""
        try:
            export_path = Path(export_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == "json":
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config_data, f, indent=2, ensure_ascii=False)
            elif format.lower() == "yaml":
                with open(export_path, 'w', encoding='utf-8') as f:
                    yaml.dump(self.config_data, f, default_flow_style=False, allow_unicode=True)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            self.logger.info(f"Configuration exported to {export_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting configuration: {str(e)}")
            raise
    
    def get_config_template(self) -> Dict[str, Any]:
        """Get configuration template with descriptions."""
        return {
            "analysis_settings": {
                "version": "1.0.0",
                "analysis_type": "contingency_professional",
                "output_precision": 6,
                "enable_validation": True,
                "enable_visualization": True,
                "enable_reporting": True,
                "_description": "Core analysis settings"
            },
            "data_processing": {
                "missing_value_strategy": "exclude",
                "zero_handling": "preserve",
                "normalization": "none",
                "outlier_detection": True,
                "_description": "Data preprocessing options"
            },
            "logging": {
                "level": "INFO",
                "log_file": "logs/analysis.log",
                "console_output": True,
                "_description": "Logging configuration"
            },
            "output": {
                "base_directory": "output",
                "create_subdirectories": True,
                "include_timestamp": True,
                "_description": "Output settings"
            }
        }
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "analysis_settings": {
                "version": "1.0.0",
                "analysis_type": "contingency_professional",
                "output_precision": 6,
                "enable_validation": True,
                "enable_visualization": True,
                "enable_reporting": True
            },
            "data_processing": {
                "missing_value_strategy": "exclude",
                "zero_handling": "preserve",
                "normalization": "none",
                "outlier_detection": True,
                "outlier_threshold": 3.0
            },
            "contingency_analysis": {
                "require_square_tables": False,
                "min_cell_count": 1,
                "min_expected_frequency": 5,
                "max_expected_frequency_violations": 0.2,
                "confidence_level": 0.95,
                "continuity_correction": True
            },
            "global_fit_analysis": {
                "chi_square_test": {
                    "enabled": True,
                    "significance_level": 0.05
                },
                "likelihood_ratio_test": {
                    "enabled": True,
                    "significance_level": 0.05
                }
            },
            "cramers_v_analysis": {
                "enabled": True,
                "confidence_intervals": True,
                "bootstrap_samples": 1000,
                "bias_correction": True
            },
            "visualization": {
                "pie_charts": {
                    "enabled": True,
                    "style": "professional",
                    "color_scheme": "Set3",
                    "figure_size": [10, 8],
                    "dpi": 300
                },
                "radar_charts": {
                    "enabled": True,
                    "style": "professional",
                    "color_scheme": "viridis",
                    "figure_size": [10, 10],
                    "dpi": 300
                }
            },
            "validation": {
                "data_validator": {
                    "enabled": True,
                    "strict_mode": False
                },
                "matrix_validator": {
                    "enabled": True,
                    "require_square_matrices": False
                },
                "stats_validator": {
                    "enabled": True,
                    "significance_level": 0.05
                }
            },
            "logging": {
                "level": "INFO",
                "log_file": "logs/analysis.log",
                "console_output": True,
                "file_output": True
            },
            "output": {
                "base_directory": "output",
                "create_subdirectories": True,
                "include_timestamp": True,
                "formats": {
                    "results": "json",
                    "tables": "csv",
                    "report": "html"
                }
            }
        }
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return json.dumps(self.config_data, indent=2)
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"ConfigManager(config_path='{self.config_path}', sections={len(self.config_data)})"
