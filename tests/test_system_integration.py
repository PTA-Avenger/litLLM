"""
Comprehensive integration tests for the complete poetry generation workflow.

This module tests the end-to-end functionality of the Stylistic Poetry LLM
Framework, ensuring all components work together correctly.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stylometric.system_integration import (
    PoetryLLMSystem, SystemStatus, initialize_global_system, get_global_system
)
from config.settings import SystemConfig
from utils.exceptions import PoetryLLMError, ConfigurationError, ValidationError


class TestSystemIntegration:
    """Test system integration functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test directories
        (self.temp_path / "data").mkdir(exist_ok=True)
        (self.temp_path / "models").mkdir(exist_ok=True)
        (self.temp_path / "results").mkdir(exist_ok=True)
        
        # Create test corpus
        self.test_corpus_dir = self.temp_path / "data" / "test_poet"
        self.test_corpus_dir.mkdir(exist_ok=True)
        
        # Create sample poems
        sample_poems = [
            "The quiet forest whispers secrets\nTo those who listen with their hearts\nNature's wisdom flows like streams",
            "Beneath the starlit sky so vast\nI ponder life's eternal questions\nTime moves forward, memories last",
            "Morning dew on petals bright\nReflects the sun's first golden rays\nA new beginning, fresh delight"
        ]
        
        for i, poem in enumerate(sample_poems):
            poem_file = self.test_corpus_dir / f"poem_{i+1}.txt"
            poem_file.write_text(poem, encoding='utf-8')
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_system_initialization(self):
        """Test complete system initialization."""
        system = PoetryLLMSystem()
        
        # Mock dependencies to avoid actual imports
        with patch('stylometric.system_integration.TextProcessor') as mock_text_processor, \
             patch('stylometric.system_integration.LexicalAnalyzer') as mock_lexical, \
             patch('stylometric.system_integration.StructuralAnalyzer') as mock_structural, \
             patch('stylometric.system_integration.PoetProfileManager') as mock_profile, \
             patch('stylometric.system_integration.TrainingDataProcessor') as mock_training, \
             patch('stylometric.system_integration.PoetryCorpusLoader') as mock_corpus, \
             patch('stylometric.system_integration.FineTuningManager') as mock_fine_tuning, \
             patch('stylometric.system_integration.QuantitativeEvaluator') as mock_evaluator, \
             patch('stylometric.system_integration.EvaluationComparator') as mock_comparator, \
             patch('stylometric.system_integration.DickinsonFeatureExtractor') as mock_dickinson_ext, \
             patch('stylometric.system_integration.DickinsonStyleGenerator') as mock_dickinson_gen, \
             patch('stylometric.system_integration.PoetryOutputFormatter') as mock_formatter, \
             patch('stylometric.system_integration.create_performance_monitor') as mock_monitor, \
             patch('stylometric.system_integration.initialize_logging'), \
             patch('stylometric.system_integration.initialize_error_handling'), \
             patch('stylometric.system_integration.get_logger') as mock_get_logger:
            
            # Setup mocks
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            # Mock component instances
            mock_text_processor.return_value.clean_text.return_value = "cleaned text"
            mock_lexical.return_value.calculate_lexical_metrics.return_value = {"ttr": 0.5}
            mock_structural.return_value.analyze_structure.return_value = {"line_count": 3}
            mock_monitor.return_value = Mock()
            
            # Test initialization
            success = system.initialize(validate_dependencies=False)
            
            assert success is True
            assert system.status.initialized is True
            assert system.status.configuration_valid is True
            assert len(system.status.components_loaded) > 0
            
            # Verify all components were initialized
            expected_components = [
                'text_processor', 'lexical_analyzer', 'structural_analyzer',
                'profile_manager', 'training_processor', 'corpus_loader',
                'fine_tuning_manager', 'quantitative_evaluator', 'evaluation_comparator',
                'dickinson_extractor', 'dickinson_generator', 'output_formatter',
                'performance_monitor'
            ]
            
            for component in expected_components:
                assert component in system.status.components_loaded
                assert system.status.components_loaded[component] is True
    
    def test_system_initialization_failure(self):
        """Test system initialization failure handling."""
        system = PoetryLLMSystem()
        
        # Mock a component initialization failure
        with patch('stylometric.system_integration.TextProcessor') as mock_text_processor, \
             patch('stylometric.system_integration.initialize_logging'), \
             patch('stylometric.system_integration.initialize_error_handling'), \
             patch('stylometric.system_integration.get_logger') as mock_get_logger:
            
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            mock_text_processor.side_effect = Exception("Component initialization failed")
            
            success = system.initialize(validate_dependencies=False)
            
            assert success is False
            assert system.status.initialized is False
            assert len(system.status.errors) > 0
            assert "Component initialization failed" in system.status.errors[0]
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        system = PoetryLLMSystem()
        
        with patch('stylometric.system_integration.initialize_logging'), \
             patch('stylometric.system_integration.get_logger') as mock_get_logger:
            
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            # Test with invalid configuration
            system.config = SystemConfig()
            system.config.model.temperature = 3.0  # Invalid temperature
            
            with pytest.raises(ConfigurationError):
                system._validate_configuration()
            
            # Test with valid configuration
            system.config.model.temperature = 0.8
            system.config.model.top_p = 0.9
            system.config.training.learning_rate = 5e-5
            
            # Should not raise exception
            system._validate_configuration()
    
    def test_dependency_validation(self):
        """Test dependency validation."""
        system = PoetryLLMSystem()
        system.logger = Mock()
        
        # Mock successful imports
        with patch('builtins.__import__') as mock_import:
            mock_import.return_value = Mock()
            
            result = system._validate_dependencies()
            
            assert result is True
            assert len(system.status.dependencies_available) > 0
            assert all(system.status.dependencies_available.values())
        
        # Mock failed imports
        with patch('builtins.__import__') as mock_import:
            mock_import.side_effect = ImportError("Module not found")
            
            result = system._validate_dependencies()
            
            assert result is False
            assert len(system.status.warnings) > 0
            assert "Missing packages" in system.status.warnings[0]
    
    def test_system_status(self):
        """Test system status tracking."""
        system = PoetryLLMSystem()
        
        # Initial status
        status = system.get_system_status()
        assert isinstance(status, SystemStatus)
        assert status.initialized is False
        assert status.configuration_valid is False
        assert len(status.components_loaded) == 0
        assert len(status.errors) == 0
        assert len(status.warnings) == 0
        
        # Add some status information
        system.status.errors.append("Test error")
        system.status.warnings.append("Test warning")
        system.status.components_loaded['test_component'] = True
        
        status = system.get_system_status()
        assert len(status.errors) == 1
        assert len(status.warnings) == 1
        assert status.components_loaded['test_component'] is True
    
    def test_system_cleanup(self):
        """Test system cleanup."""
        system = PoetryLLMSystem()
        system.logger = Mock()
        
        # Add mock active models
        mock_model1 = Mock()
        mock_model2 = Mock()
        system.active_models = {
            'model1': mock_model1,
            'model2': mock_model2
        }
        system.status.initialized = True
        
        # Test cleanup
        system.cleanup()
        
        # Verify models were unloaded
        mock_model1.unload_model.assert_called_once()
        mock_model2.unload_model.assert_called_once()
        
        # Verify status reset
        assert len(system.active_models) == 0
        assert system.status.initialized is False
    
    def test_system_cleanup_with_errors(self):
        """Test system cleanup with model unloading errors."""
        system = PoetryLLMSystem()
        system.logger = Mock()
        
        # Add mock active model that fails to unload
        mock_model = Mock()
        mock_model.unload_model.side_effect = Exception("Unload failed")
        system.active_models = {'failing_model': mock_model}
        system.status.initialized = True
        
        # Test cleanup (should not raise exception)
        system.cleanup()
        
        # Verify cleanup attempted
        mock_model.unload_model.assert_called_once()
        assert len(system.active_models) == 0
        assert system.status.initialized is False
        
        # Verify warning was logged
        system.logger.warning.assert_called()


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test directories
        (self.temp_path / "data").mkdir(exist_ok=True)
        (self.temp_path / "models").mkdir(exist_ok=True)
        (self.temp_path / "results").mkdir(exist_ok=True)
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('stylometric.system_integration.PoetryLLMSystem.initialize')
    def test_global_system_initialization(self, mock_init):
        """Test global system initialization function."""
        mock_init.return_value = True
        
        # Test initialization
        result = initialize_global_system(
            config_path=None,
            log_level="DEBUG",
            enable_error_recovery=False,
            validate_dependencies=False
        )
        
        assert result is True
        mock_init.assert_called_once_with(
            log_level="DEBUG",
            enable_error_recovery=False,
            validate_dependencies=False
        )
    
    def test_get_global_system(self):
        """Test getting global system instance."""
        system = get_global_system()
        assert isinstance(system, PoetryLLMSystem)
        
        # Should return the same instance
        system2 = get_global_system()
        assert system is system2
    
    @patch('stylometric.system_integration.PoetryLLMSystem._initialize_components')
    @patch('stylometric.system_integration.PoetryLLMSystem._validate_system_setup')
    @patch('stylometric.system_integration.PoetryLLMSystem._validate_dependencies')
    @patch('stylometric.system_integration.PoetryLLMSystem._load_configuration')
    @patch('stylometric.system_integration.initialize_logging')
    @patch('stylometric.system_integration.initialize_error_handling')
    @patch('stylometric.system_integration.get_logger')
    def test_complete_initialization_workflow(self, mock_get_logger, mock_init_error,
                                            mock_init_logging, mock_load_config,
                                            mock_validate_deps, mock_validate_setup,
                                            mock_init_components):
        """Test complete initialization workflow."""
        # Setup mocks
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        mock_load_config.return_value = True
        mock_validate_deps.return_value = True
        mock_init_components.return_value = True
        mock_validate_setup.return_value = True
        
        system = PoetryLLMSystem()
        result = system.initialize()
        
        # Verify initialization steps were called in order
        mock_init_logging.assert_called_once()
        mock_init_error.assert_called_once()
        mock_load_config.assert_called_once()
        mock_validate_deps.assert_called_once()
        mock_init_components.assert_called_once()
        mock_validate_setup.assert_called_once()
        
        assert result is True
        assert system.status.initialized is True
    
    def test_error_propagation_in_workflow(self):
        """Test error propagation through the workflow."""
        system = PoetryLLMSystem()
        
        # Mock configuration loading failure
        with patch('stylometric.system_integration.initialize_logging'), \
             patch('stylometric.system_integration.initialize_error_handling'), \
             patch('stylometric.system_integration.get_logger') as mock_get_logger, \
             patch('stylometric.system_integration.config_manager') as mock_config_manager:
            
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            mock_config_manager.get_config.side_effect = Exception("Config error")
            
            result = system.initialize()
            
            assert result is False
            assert system.status.initialized is False
            assert len(system.status.errors) > 0
            assert "Configuration loading failed" in system.status.errors[0]


class TestSystemValidation:
    """Test system validation functionality."""
    
    def test_system_status_dataclass(self):
        """Test SystemStatus dataclass functionality."""
        status = SystemStatus()
        
        # Test default values
        assert status.initialized is False
        assert status.configuration_valid is False
        assert isinstance(status.components_loaded, dict)
        assert isinstance(status.dependencies_available, dict)
        assert isinstance(status.errors, list)
        assert isinstance(status.warnings, list)
        assert status.last_validation is None
        
        # Test modification
        status.initialized = True
        status.components_loaded['test'] = True
        status.errors.append("test error")
        
        assert status.initialized is True
        assert status.components_loaded['test'] is True
        assert "test error" in status.errors
    
    def test_configuration_error_handling(self):
        """Test configuration error handling."""
        system = PoetryLLMSystem()
        system.logger = Mock()
        
        # Test missing configuration
        system.config = None
        with pytest.raises(ConfigurationError, match="No configuration loaded"):
            system._validate_configuration()
        
        # Test invalid temperature
        system.config = SystemConfig()
        system.config.model.temperature = -1.0
        with pytest.raises(ConfigurationError, match="Temperature must be between"):
            system._validate_configuration()
        
        # Test invalid top_p
        system.config.model.temperature = 0.8
        system.config.model.top_p = 1.5
        with pytest.raises(ConfigurationError, match="top_p must be between"):
            system._validate_configuration()
        
        # Test invalid learning rate
        system.config.model.top_p = 0.9
        system.config.training.learning_rate = -0.001
        with pytest.raises(ConfigurationError, match="Learning rate must be positive"):
            system._validate_configuration()
    
    def test_directory_creation_validation(self):
        """Test directory creation during validation."""
        system = PoetryLLMSystem()
        system.logger = Mock()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create valid configuration with test directories
            system.config = SystemConfig()
            system.config.data.data_dir = str(temp_path / "data")
            system.config.data.output_dir = str(temp_path / "models")
            system.config.evaluation.metrics_output_dir = str(temp_path / "results")
            
            # Should create directories and not raise exception
            system._validate_configuration()
            
            # Verify directories were created
            assert (temp_path / "data").exists()
            assert (temp_path / "models").exists()
            assert (temp_path / "results").exists()


if __name__ == "__main__":
    pytest.main([__file__])