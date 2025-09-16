"""Configuration management utilities."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Configuration manager for RDEX-ABCD analysis pipeline."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.

        Args:
            config_path: Path to configuration file. If None, looks for config.yaml
                        in the repository root.
        """
        if config_path is None:
            # Find repository root (look for .git directory)
            current_dir = Path(__file__).parent
            while current_dir != current_dir.parent:
                if (current_dir / ".git").exists():
                    config_path = current_dir / "config.yaml"
                    break
                current_dir = current_dir.parent
            else:
                raise FileNotFoundError(
                    "Could not find repository root with config.yaml"
                )

        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._resolve_paths()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def _resolve_paths(self):
        """Convert relative paths to absolute paths based on config file location."""
        repo_root = self.config_path.parent

        def resolve_path_recursive(obj):
            if isinstance(obj, dict):
                return {k: resolve_path_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [resolve_path_recursive(item) for item in obj]
            elif isinstance(obj, str) and ("_path" in str(obj) or "dir" in str(obj)):
                # Convert relative paths to absolute
                if not os.path.isabs(obj):
                    return str(repo_root / obj)
                return obj
            else:
                return obj

        # Only resolve paths in data_management, rdex_prediction,
        # and phenotype_prediction sections
        for section in ["data_management", "rdex_prediction", "phenotype_prediction"]:
            if section in self.config:
                self.config[section] = resolve_path_recursive(self.config[section])

    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation.

        Args:
            key_path: Dot-separated path to configuration value
                (e.g., 'rdex_prediction.models')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key_path.split(".")
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section.

        Args:
            section: Section name (e.g., 'rdex_prediction')

        Returns:
            Configuration section as dictionary
        """
        return self.config.get(section, {})

    def get_data_management_config(self) -> Dict[str, Any]:
        """Get data management configuration."""
        return self.get_section("data_management")

    def get_rdex_prediction_config(self) -> Dict[str, Any]:
        """Get RDEX prediction configuration."""
        return self.get_section("rdex_prediction")

    def get_phenotype_prediction_config(self) -> Dict[str, Any]:
        """Get phenotype prediction configuration."""
        return self.get_section("phenotype_prediction")

    def get_processing_config(self) -> Dict[str, Any]:
        """Get processing configuration."""
        return self.get_section("processing")

    def get_visualization_config(self) -> Dict[str, Any]:
        """Get visualization configuration."""
        return self.get_section("visualization")

    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values.

        Args:
            updates: Dictionary of configuration updates using dot notation keys
        """
        for key_path, value in updates.items():
            keys = key_path.split(".")
            config_section = self.config

            # Navigate to the parent of the target key
            for key in keys[:-1]:
                if key not in config_section:
                    config_section[key] = {}
                config_section = config_section[key]

            # Set the final value
            config_section[keys[-1]] = value

    def save_config(self, output_path: Optional[str] = None):
        """Save current configuration to file.

        Args:
            output_path: Path to save configuration. If None, overwrites original file.
        """
        if output_path is None:
            output_path = self.config_path

        with open(output_path, "w") as f:
            yaml.safe_dump(self.config, f, default_flow_style=False, indent=2)


# Global configuration instance
_config = None


def get_config(config_path: Optional[str] = None) -> Config:
    """Get global configuration instance.

    Args:
        config_path: Path to configuration file. Only used on first call.

    Returns:
        Configuration instance
    """
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config


def load_module_config(module_name: str) -> Dict[str, Any]:
    """Load configuration for a specific module.

    Args:
        module_name: Name of the module ('data_management', 'rdex_prediction', etc.)

    Returns:
        Module configuration dictionary
    """
    config = get_config()
    return config.get_section(module_name)
