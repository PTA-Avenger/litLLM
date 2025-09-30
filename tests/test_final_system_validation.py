"""
Final system validation tests for the complete Stylistic Poetry LLM Framework.

This module provides comprehensive end-to-end validation tests that verify
the entire system works correctly as an integrated whole.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.system_validation import (
    SystemValidator, ValidationResult, SystemValidationReport,
    validate_system_setup
)
from config.settings import SystemConfig
from utils.exceptions import ValidationError, ConfigurationError


class TestSystemValidator:
    """Test system validator functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.validator = SystemValidator()
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_validation_result_creation(self):
        """Test ValidationResult dataclass."""
        result = ValidationResult(
            valid=True,
            message="Test validation passed",
            severity="info"
        )
        
        assert result.valid is True
        assert result.message == "Test validation passed"
        assert result.severity == "info"
        assert isinstance(result.recommendations, list)
        assert len(result.recommendations) == 0
        
        # Test with recommendations
        result_with_recs = ValidationResult(
            valid=False,
            message="Test failed",
            severity="error",
            recommendations=["Fix this", "Fix that"]
        )
        
        assert len(result_with_recs.recommendations) == 2
        assert "Fix this" in result_with_recs.recommendations
    
    def test_system_validation_report_creation(self):
        """Test SystemValidationReport dataclass."""
        report = SystemValidationReport(
            timestamp=datetime.now(),
            overall_valid=True,
            configuration_valid=True,
            dependencies_valid=True,
            directory_structure_valid=True,
            component_tests_valid=True,
            validation_results=[],
            dependency_status={},
            missing_dependencies=[],
            configuration_issues=[],
            recommendations=[]
        )
        
        assert report.overall_valid is True
        assert isinstance(report.validation_results, list)
        assert isinstance(report.dependency_status, dict)
        assert isinstance(report.missing_dependencies, list)
        assert isinstance(report.configuration_issues, list)
        assert isinstance(report.recommendations, list)
    
    @patch('utils.system_validation.config_manager')
    def test_configuration_validation_success(self, mock_config_manager):
        """Test successful configuration validation."""
        # Create valid configuration
        config = SystemConfig()
        config.model.temperature = 0.8
        config.model.top_p = 0.9
        config.model.top_k = 50
        config.model.max_length = 512
        config.training.learning_rate = 5e-5
        config.training.batch_size = 8
        config.training.num_epochs = 3
        config.device = "auto"
        
        # Set up temporary directories
        config.data.data_dir = str(self.temp_path / "data")
        config.data.output_dir = str(self.temp_path / "models")
        config.evaluation.metrics_output_dir = str(self.temp_path / "results")
        
        mock_config_manager.get_config.return_value = config
        
        result = self.validator._validate_configuration()
        
        assert result.valid is True
        assert "Configuration validation passed" in result.message
        assert result.severity == "info"
    
    @patch('utils.system_validation.config_manager')
    def test_configuration_validation_failure(self, mock_config_manager):
        """Test configuration validation with invalid values."""
        # Create invalid configuration
        config = SystemConfig()
        config.model.temperature = 3.0  # Invalid
        config.model.top_p = 1.5  # Invalid
        config.training.learning_rate = -0.001  # Invalid
        
        mock_config_manager.get_config.return_value = config
        
        result = self.validator._validate_configuration()
        
        assert result.valid is False
        assert "Configuration validation failed" in result.message
        assert result.severity == "error"
        assert len(result.recommendations) > 0
    
    @patch('builtins.__import__')
    def test_dependency_validation_success(self, mock_import):
        """Test successful dependency validation."""
        # Mock successful imports
        mock_import.return_value = Mock()
        
        result = self.validator._validate_dependencies()
        
        assert result.valid is True
        assert "All dependencies validated successfully" in result.message
        assert result.severity == "info"
    
    @patch('builtins.__import__')
    def test_dependency_validation_failure(self, mock_import):
        """Test dependency validation with missing packages."""
        # Mock failed imports
        mock_import.side_effect = ImportError("Module not found")
        
        result = self.validator._validate_dependencies()
        
        assert result.valid is False
        assert "Missing required dependencies" in result.message
        assert result.severity == "error"
        assert len(result.recommendations) > 0
    
    @patch('builtins.__import__')
    def test_dependency_validation_with_optional(self, mock_import):
        """Test dependency validation including optional packages."""
        # Mock some packages missing
        def import_side_effect(name):
            if name in ['matplotlib', 'seaborn']:
                raise ImportError("Optional package not found")
            return Mock()
        
        mock_import.side_effect = import_side_effect
        
        result = self.validator._validate_dependencies(check_optional=True)
        
        # Should still be valid since only optional packages are missing
        assert result.valid is True
        assert "Missing optional" in result.message
        assert result.severity == "warning"
    
    def test_directory_structure_validation_success(self):
        """Test successful directory structure validation."""
        # Create required directories
        for directory in self.validator.required_directories:
            (self.temp_path / directory).mkdir(exist_ok=True)
        
        # Temporarily change working directory
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(self.temp_path)
            result = self.validator._validate_directory_structure()
            
            assert result.valid is True
            assert "Directory structure validation passed" in result.message
            assert result.severity == "info"
        finally:
            os.chdir(original_cwd)
    
    def test_directory_structure_validation_failure(self):
        """Test directory structure validation with missing directories."""
        # Don't create directories - they should be missing
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(self.temp_path)
            result = self.validator._validate_directory_structure()
            
            assert result.valid is False
            assert "Missing required directories" in result.message
            assert result.severity == "warning"
            assert len(result.recommendations) > 0
        finally:
            os.chdir(original_cwd)
    
    @patch('utils.system_validation.TextProcessor')
    @patch('utils.system_validation.LexicalAnalyzer')
    @patch('utils.system_validation.StructuralAnalyzer')
    def test_component_functionality_validation_success(self, mock_structural, mock_lexical, mock_text):
        """Test successful component functionality validation."""
        # Mock successful component operations
        mock_text_instance = Mock()
        mock_text_instance.clean_text.return_value = "cleaned text"
        mock_text.return_value = mock_text_instance
        
        mock_lexical_instance = Mock()
        mock_lexical_instance.calculate_lexical_metrics.return_value = {"ttr": 0.5}
        mock_lexical.return_value = mock_lexical_instance
        
        mock_structural_instance = Mock()
        mock_structural_instance.analyze_structure.return_value = {"line_count": 3}
        mock_structural.return_value = mock_structural_instance
        
        result = self.validator._validate_component_functionality()
        
        assert result.valid is True
        assert "Component functionality validation passed" in result.message
        assert result.severity == "info"
    
    @patch('utils.system_validation.TextProcessor')
    def test_component_functionality_validation_failure(self, mock_text):
        """Test component functionality validation with failures."""
        # Mock component failure
        mock_text.side_effect = Exception("Component initialization failed")
        
        result = self.validator._validate_component_functionality()
        
        assert result.valid is False
        assert "Component functionality tests failed" in result.message
        assert result.severity == "error"
        assert len(result.recommendations) > 0
    
    @patch('utils.system_validation.SystemValidator._validate_configuration')
    @patch('utils.system_validation.SystemValidator._validate_dependencies')
    @patch('utils.system_validation.SystemValidator._validate_directory_structure')
    @patch('utils.system_validation.SystemValidator._validate_component_functionality')
    def test_complete_system_validation_success(self, mock_components, mock_dirs, mock_deps, mock_config):
        """Test complete system validation with all checks passing."""
        # Mock all validations as successful
        mock_config.return_value = ValidationResult(True, "Config OK", "info")
        mock_deps.return_value = ValidationResult(True, "Dependencies OK", "info")
        mock_dirs.return_value = ValidationResult(True, "Directories OK", "info")
        mock_components.return_value = ValidationResult(True, "Components OK", "info")
        
        report = self.validator.validate_complete_system()
        
        assert report.overall_valid is True
        assert report.configuration_valid is True
        assert report.dependencies_valid is True
        assert report.directory_structure_valid is True
        assert report.component_tests_valid is True
        assert len(report.validation_results) == 4
    
    @patch('utils.system_validation.SystemValidator._validate_configuration')
    @patch('utils.system_validation.SystemValidator._validate_dependencies')
    @patch('utils.system_validation.SystemValidator._validate_directory_structure')
    @patch('utils.system_validation.SystemValidator._validate_component_functionality')
    def test_complete_system_validation_failure(self, mock_components, mock_dirs, mock_deps, mock_config):
        """Test complete system validation with some checks failing."""
        # Mock some validations as failed
        mock_config.return_value = ValidationResult(False, "Config failed", "error", ["Fix config"])
        mock_deps.return_value = ValidationResult(True, "Dependencies OK", "info")
        mock_dirs.return_value = ValidationResult(False, "Directories missing", "warning", ["Create dirs"])
        mock_components.return_value = ValidationResult(True, "Components OK", "info")
        
        report = self.validator.validate_complete_system()
        
        assert report.overall_valid is False
        assert report.configuration_valid is False
        assert report.dependencies_valid is True
        assert report.directory_structure_valid is False
        assert report.component_tests_valid is True
        assert len(report.validation_results) == 4
        assert len(report.recommendations) > 0
    
    def test_generate_validation_report(self):
        """Test validation report generation."""
        # Create sample report
        report = SystemValidationReport(
            timestamp=datetime.now(),
            overall_valid=False,
            configuration_valid=True,
            dependencies_valid=False,
            directory_structure_valid=True,
            component_tests_valid=True,
            validation_results=[
                ValidationResult(True, "Config OK", "info"),
                ValidationResult(False, "Missing deps", "error", ["Install packages"])
            ],
            dependency_status={"torch": True, "missing_pkg": False},
            missing_dependencies=["missing_pkg"],
            configuration_issues=[],
            recommendations=["Install missing packages", "Check system setup"]
        )
        
        report_text = self.validator.generate_validation_report(report)
        
        assert "STYLISTIC POETRY LLM SYSTEM VALIDATION REPORT" in report_text
        assert "✗ INVALID" in report_text
        assert "Config OK" in report_text
        assert "Missing deps" in report_text
        assert "Install packages" in report_text
        assert "✓ torch" in report_text
        assert "✗ missing_pkg" in report_text
    
    def test_save_validation_report_json(self):
        """Test saving validation report as JSON."""
        # Create sample report
        report = SystemValidationReport(
            timestamp=datetime.now(),
            overall_valid=True,
            configuration_valid=True,
            dependencies_valid=True,
            directory_structure_valid=True,
            component_tests_valid=True,
            validation_results=[
                ValidationResult(True, "All OK", "info")
            ],
            dependency_status={"torch": True},
            missing_dependencies=[],
            configuration_issues=[],
            recommendations=[]
        )
        
        output_path = self.temp_path / "validation_report.json"
        self.validator.save_validation_report_json(report, output_path)
        
        assert output_path.exists()
        
        # Verify JSON content
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert data['overall_valid'] is True
        assert data['configuration_valid'] is True
        assert len(data['validation_results']) == 1
        assert data['validation_results'][0]['message'] == "All OK"


class TestSystemValidationIntegration:
    """Test system validation integration functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('utils.system_validation.SystemValidator.validate_complete_system')
    def test_validate_system_setup_convenience_function(self, mock_validate):
        """Test the convenience function for system validation."""
        # Mock validation result
        mock_report = SystemValidationReport(
            timestamp=datetime.now(),
            overall_valid=True,
            configuration_valid=True,
            dependencies_valid=True,
            directory_structure_valid=True,
            component_tests_valid=True,
            validation_results=[],
            dependency_status={},
            missing_dependencies=[],
            configuration_issues=[],
            recommendations=[]
        )
        mock_validate.return_value = mock_report
        
        # Test without saving report
        report = validate_system_setup()
        
        assert report.overall_valid is True
        mock_validate.assert_called_once_with(None, False)
    
    @patch('utils.system_validation.SystemValidator.validate_complete_system')
    @patch('utils.system_validation.SystemValidator.generate_validation_report')
    def test_validate_system_setup_with_report_saving(self, mock_generate, mock_validate):
        """Test system validation with report saving."""
        # Mock validation result
        mock_report = SystemValidationReport(
            timestamp=datetime.now(),
            overall_valid=True,
            configuration_valid=True,
            dependencies_valid=True,
            directory_structure_valid=True,
            component_tests_valid=True,
            validation_results=[],
            dependency_status={},
            missing_dependencies=[],
            configuration_issues=[],
            recommendations=[]
        )
        mock_validate.return_value = mock_report
        mock_generate.return_value = "Mock report text"
        
        # Test with saving report
        report_path = self.temp_path / "test_report.txt"
        report = validate_system_setup(
            save_report=True,
            report_path=report_path
        )
        
        assert report.overall_valid is True
        mock_validate.assert_called_once()
        mock_generate.assert_called_once_with(mock_report, report_path)
    
    def test_validation_error_handling(self):
        """Test validation error handling."""
        validator = SystemValidator()
        
        # Test with invalid configuration that causes exception
        with patch('utils.system_validation.config_manager') as mock_config_manager:
            mock_config_manager.get_config.side_effect = Exception("Config error")
            
            result = validator._validate_configuration()
            
            assert result.valid is False
            assert "Configuration validation error" in result.message
            assert result.severity == "error"
            assert len(result.recommendations) > 0
    
    def test_package_version_checking(self):
        """Test package version checking functionality."""
        validator = SystemValidator()
        
        # Mock torch with old version
        with patch('builtins.__import__') as mock_import:
            mock_torch = Mock()
            mock_torch.__version__ = "1.9.0"
            
            def import_side_effect(name):
                if name == 'torch':
                    return mock_torch
                elif name == 'transformers':
                    mock_transformers = Mock()
                    mock_transformers.__version__ = "3.5.0"
                    return mock_transformers
                else:
                    raise ImportError("Module not found")
            
            mock_import.side_effect = import_side_effect
            
            recommendations = validator._check_package_versions()
            
            assert len(recommendations) == 2
            assert any("PyTorch" in rec for rec in recommendations)
            assert any("transformers" in rec for rec in recommendations)


class TestValidationReporting:
    """Test validation reporting functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.validator = SystemValidator()
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_comprehensive_report_generation(self):
        """Test comprehensive validation report generation."""
        # Create comprehensive report with various scenarios
        report = SystemValidationReport(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            overall_valid=False,
            configuration_valid=False,
            dependencies_valid=True,
            directory_structure_valid=False,
            component_tests_valid=True,
            validation_results=[
                ValidationResult(False, "Invalid temperature", "error", ["Set temperature to 0.8"]),
                ValidationResult(True, "All dependencies available", "info"),
                ValidationResult(False, "Missing data directory", "warning", ["Create data/ directory"]),
                ValidationResult(True, "Components working", "info")
            ],
            dependency_status={
                "torch": True,
                "transformers": True,
                "nltk": True,
                "missing_package": False
            },
            missing_dependencies=["missing_package"],
            configuration_issues=["Invalid temperature"],
            recommendations=[
                "Set temperature to 0.8",
                "Create data/ directory",
                "Install missing_package"
            ]
        )
        
        report_text = self.validator.generate_validation_report(report)
        
        # Verify report structure
        assert "STYLISTIC POETRY LLM SYSTEM VALIDATION REPORT" in report_text
        assert "2024-01-01 12:00:00" in report_text
        assert "✗ INVALID" in report_text
        
        # Verify summary section
        assert "VALIDATION SUMMARY" in report_text
        assert "Configuration: ✗" in report_text
        assert "Dependencies: ✓" in report_text
        assert "Directory Structure: ✗" in report_text
        assert "Component Tests: ✓" in report_text
        
        # Verify detailed results
        assert "DETAILED VALIDATION RESULTS" in report_text
        assert "Invalid temperature" in report_text
        assert "All dependencies available" in report_text
        assert "Missing data directory" in report_text
        assert "Components working" in report_text
        
        # Verify dependency status
        assert "DEPENDENCY STATUS" in report_text
        assert "✓ torch" in report_text
        assert "✗ missing_package" in report_text
        
        # Verify recommendations
        assert "RECOMMENDATIONS" in report_text
        assert "1. Set temperature to 0.8" in report_text
        assert "2. Create data/ directory" in report_text
        assert "3. Install missing_package" in report_text
    
    def test_report_file_saving(self):
        """Test saving validation report to file."""
        report = SystemValidationReport(
            timestamp=datetime.now(),
            overall_valid=True,
            configuration_valid=True,
            dependencies_valid=True,
            directory_structure_valid=True,
            component_tests_valid=True,
            validation_results=[
                ValidationResult(True, "System OK", "info")
            ],
            dependency_status={},
            missing_dependencies=[],
            configuration_issues=[],
            recommendations=[]
        )
        
        output_path = self.temp_path / "validation_report.txt"
        report_text = self.validator.generate_validation_report(report, output_path)
        
        # Verify file was created
        assert output_path.exists()
        
        # Verify file content matches returned text
        saved_content = output_path.read_text(encoding='utf-8')
        assert saved_content == report_text
        assert "System OK" in saved_content


if __name__ == "__main__":
    pytest.main([__file__])