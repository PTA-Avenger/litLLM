"""Tests for basic project setup and configuration."""

import pytest
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.settings import ConfigManager, SystemConfig
from utils.logging import setup_logging, get_logger
from utils.exceptions import PoetryLLMError, ConfigurationError


class TestProjectStructure:
    """Test that the project structure is set up correctly."""
    
    def test_required_directories_exist(self):
        """Test that all required directories exist."""
        required_dirs = ['src', 'tests', 'data', 'models', 'config']
        
        for directory in required_dirs:
            path = Path(directory)
            assert path.exists(), f"Required directory {directory} does not exist"
            assert path.is_dir(), f"{directory} exists but is not a directory"
    
    def test_config_files_exist(self):
        """Test that configuration files exist."""
        config_files = [
            'config/default.yaml',
            'config/settings.py',
            'requirements.txt',
            'setup.py',
            'README.md'
        ]
        
        for config_file in config_files:
            path = Path(config_file)
            assert path.exists(), f"Required file {config_file} does not exist"
            assert path.is_file(), f"{config_file} exists but is not a file"


class TestConfigurationManager:
    """Test the configuration management system."""
    
    def test_default_config_creation(self):
        """Test that default configuration can be created."""
        config = SystemConfig()
        
        assert config.model.base_model_name == "gpt2"
        assert config.training.learning_rate == 5e-5
        assert config.data.data_dir == "data"
        assert config.log_level == "INFO"
    
    def test_config_manager_initialization(self):
        """Test that ConfigManager can be initialized."""
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        assert isinstance(config, SystemConfig)
        assert config.model.base_model_name is not None
        assert config.training.batch_size > 0
    
    def test_config_update(self):
        """Test that configuration can be updated."""
        config_manager = ConfigManager()
        
        updates = {
            "model": {"temperature": 0.9},
            "training": {"batch_size": 16}
        }
        
        updated_config = config_manager.update_config(updates)
        
        assert updated_config.model.temperature == 0.9
        assert updated_config.training.batch_size == 16


class TestLogging:
    """Test the logging system."""
    
    def test_logger_setup(self):
        """Test that logging can be set up."""
        logger = setup_logging(log_level="DEBUG", console_output=True)
        
        assert logger is not None
        assert logger.level == 10  # DEBUG level
    
    def test_get_logger(self):
        """Test that named loggers can be retrieved."""
        logger = get_logger("test")
        
        assert logger is not None
        assert "test" in logger.name


class TestExceptions:
    """Test the custom exception system."""
    
    def test_base_exception(self):
        """Test the base PoetryLLMError exception."""
        error = PoetryLLMError("Test message", "TEST001", {"key": "value"})
        
        assert str(error) == "[TEST001] Test message"
        assert error.error_code == "TEST001"
        assert error.details == {"key": "value"}
    
    def test_specific_exceptions(self):
        """Test specific exception types."""
        config_error = ConfigurationError("Config error")
        assert isinstance(config_error, PoetryLLMError)
        
        # Test that the exception can be raised and caught
        with pytest.raises(ConfigurationError):
            raise ConfigurationError("Test config error")


class TestMainModule:
    """Test the main module functionality."""
    
    def test_main_module_imports(self):
        """Test that main module can be imported."""
        try:
            from src.main import cli, main
            assert cli is not None
            assert main is not None
        except ImportError as e:
            pytest.fail(f"Failed to import main module: {e}")


if __name__ == "__main__":
    pytest.main([__file__])