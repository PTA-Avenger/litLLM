"""
System configuration validation and setup verification utilities.

This module provides comprehensive validation of system setup, configuration,
and dependencies for the Stylistic Poetry LLM Framework.
"""

import sys
import importlib
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import SystemConfig, config_manager
from utils.logging import get_logger
from utils.exceptions import ValidationError, ConfigurationError


@dataclass
class ValidationResult:
    """Result of a validation check."""
    valid: bool
    message: str
    severity: str = "info"  # info, warning, error
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []


@dataclass
class SystemValidationReport:
    """Comprehensive system validation report."""
    timestamp: datetime
    overall_valid: bool
    configuration_valid: bool
    dependencies_valid: bool
    directory_structure_valid: bool
    component_tests_valid: bool
    
    validation_results: List[ValidationResult]
    dependency_status: Dict[str, bool]
    missing_dependencies: List[str]
    configuration_issues: List[str]
    recommendations: List[str]
    
    def __post_init__(self):
        if self.validation_results is None:
            self.validation_results = []
        if self.dependency_status is None:
            self.dependency_status = {}
        if self.missing_dependencies is None:
            self.missing_dependencies = []
        if self.configuration_issues is None:
            self.configuration_issues = []
        if self.recommendations is None:
            self.recommendations = []


class SystemValidator:
    """Comprehensive system validator."""
    
    def __init__(self):
        self.logger = get_logger('system_validator')
        self.required_packages = [
            'torch', 'transformers', 'nltk', 'pyphen', 'pronouncing',
            'numpy', 'pandas', 'scikit-learn', 'pydantic', 'pyyaml',
            'click', 'pathlib'
        ]
        self.optional_packages = [
            'matplotlib', 'seaborn', 'jupyter', 'ipython'
        ]
        self.required_directories = [
            'src', 'tests', 'config', 'data', 'models'
        ]
    
    def validate_complete_system(self, 
                                config_path: Optional[Path] = None,
                                check_optional_deps: bool = False) -> SystemValidationReport:
        """
        Perform comprehensive system validation.
        
        Args:
            config_path: Optional path to configuration file
            check_optional_deps: Whether to check optional dependencies
            
        Returns:
            SystemValidationReport with complete validation results
        """
        self.logger.info("Starting comprehensive system validation")
        
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
        
        try:
            # Validate configuration
            config_result = self._validate_configuration(config_path)
            report.validation_results.append(config_result)
            report.configuration_valid = config_result.valid
            if not config_result.valid:
                report.configuration_issues.append(config_result.message)
                report.overall_valid = False
            
            # Validate dependencies
            deps_result = self._validate_dependencies(check_optional_deps)
            report.validation_results.append(deps_result)
            report.dependencies_valid = deps_result.valid
            if not deps_result.valid:
                report.overall_valid = False
            
            # Validate directory structure
            dirs_result = self._validate_directory_structure()
            report.validation_results.append(dirs_result)
            report.directory_structure_valid = dirs_result.valid
            if not dirs_result.valid:
                report.overall_valid = False
            
            # Validate component functionality
            components_result = self._validate_component_functionality()
            report.validation_results.append(components_result)
            report.component_tests_valid = components_result.valid
            if not components_result.valid:
                report.overall_valid = False
            
            # Collect recommendations
            for result in report.validation_results:
                report.recommendations.extend(result.recommendations)
            
            # Add overall recommendations
            if not report.overall_valid:
                report.recommendations.append(
                    "Address the validation issues above before using the system"
                )
            
            self.logger.info(f"System validation completed. Overall valid: {report.overall_valid}")
            return report
            
        except Exception as e:
            self.logger.error(f"System validation failed: {e}")
            report.overall_valid = False
            report.validation_results.append(
                ValidationResult(
                    valid=False,
                    message=f"Validation process failed: {e}",
                    severity="error",
                    recommendations=["Check system logs for detailed error information"]
                )
            )
            return report
    
    def _validate_configuration(self, config_path: Optional[Path] = None) -> ValidationResult:
        """Validate system configuration."""
        try:
            self.logger.info("Validating system configuration")
            
            # Load configuration
            if config_path:
                config = config_manager.load_config(config_path)
            else:
                config = config_manager.get_config()
            
            issues = []
            recommendations = []
            
            # Validate model configuration
            if config.model.temperature < 0.0 or config.model.temperature > 2.0:
                issues.append("Temperature must be between 0.0 and 2.0")
                recommendations.append("Set temperature to a value between 0.0 and 2.0")
            
            if config.model.top_p < 0.0 or config.model.top_p > 1.0:
                issues.append("top_p must be between 0.0 and 1.0")
                recommendations.append("Set top_p to a value between 0.0 and 1.0")
            
            if config.model.top_k < 1:
                issues.append("top_k must be at least 1")
                recommendations.append("Set top_k to a positive integer")
            
            if config.model.max_length < 10 or config.model.max_length > 2048:
                issues.append("max_length should be between 10 and 2048")
                recommendations.append("Set max_length to a reasonable value (e.g., 512)")
            
            # Validate training configuration
            if config.training.learning_rate <= 0.0:
                issues.append("Learning rate must be positive")
                recommendations.append("Set learning_rate to a small positive value (e.g., 5e-5)")
            
            if config.training.batch_size < 1:
                issues.append("Batch size must be at least 1")
                recommendations.append("Set batch_size to a positive integer")
            
            if config.training.num_epochs < 1:
                issues.append("Number of epochs must be at least 1")
                recommendations.append("Set num_epochs to a positive integer")
            
            # Validate data configuration
            data_dir = Path(config.data.data_dir)
            output_dir = Path(config.data.output_dir)
            metrics_dir = Path(config.evaluation.metrics_output_dir)
            
            # Check if directories can be created
            for dir_path, dir_name in [(data_dir, "data"), (output_dir, "output"), (metrics_dir, "metrics")]:
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    issues.append(f"Cannot create {dir_name} directory: {e}")
                    recommendations.append(f"Ensure {dir_path} is writable or change the path")
            
            # Validate device setting
            if config.device not in ["auto", "cpu", "cuda"]:
                issues.append("Device must be 'auto', 'cpu', or 'cuda'")
                recommendations.append("Set device to 'auto', 'cpu', or 'cuda'")
            
            if issues:
                return ValidationResult(
                    valid=False,
                    message=f"Configuration validation failed: {'; '.join(issues)}",
                    severity="error",
                    recommendations=recommendations
                )
            else:
                return ValidationResult(
                    valid=True,
                    message="Configuration validation passed",
                    severity="info"
                )
                
        except Exception as e:
            return ValidationResult(
                valid=False,
                message=f"Configuration validation error: {e}",
                severity="error",
                recommendations=["Check configuration file format and accessibility"]
            )
    
    def _validate_dependencies(self, check_optional: bool = False) -> ValidationResult:
        """Validate system dependencies."""
        try:
            self.logger.info("Validating system dependencies")
            
            missing_required = []
            missing_optional = []
            available_packages = {}
            
            # Check required packages
            for package in self.required_packages:
                try:
                    importlib.import_module(package)
                    available_packages[package] = True
                except ImportError:
                    available_packages[package] = False
                    missing_required.append(package)
            
            # Check optional packages if requested
            if check_optional:
                for package in self.optional_packages:
                    try:
                        importlib.import_module(package)
                        available_packages[package] = True
                    except ImportError:
                        available_packages[package] = False
                        missing_optional.append(package)
            
            recommendations = []
            
            if missing_required:
                recommendations.append(f"Install missing required packages: pip install {' '.join(missing_required)}")
            
            if missing_optional and check_optional:
                recommendations.append(f"Consider installing optional packages: pip install {' '.join(missing_optional)}")
            
            # Check specific package versions for critical dependencies
            version_issues = self._check_package_versions()
            if version_issues:
                recommendations.extend(version_issues)
            
            if missing_required:
                return ValidationResult(
                    valid=False,
                    message=f"Missing required dependencies: {', '.join(missing_required)}",
                    severity="error",
                    recommendations=recommendations
                )
            elif missing_optional and check_optional:
                return ValidationResult(
                    valid=True,
                    message=f"All required dependencies available. Missing optional: {', '.join(missing_optional)}",
                    severity="warning",
                    recommendations=recommendations
                )
            else:
                return ValidationResult(
                    valid=True,
                    message="All dependencies validated successfully",
                    severity="info"
                )
                
        except Exception as e:
            return ValidationResult(
                valid=False,
                message=f"Dependency validation error: {e}",
                severity="error",
                recommendations=["Check Python environment and package installation"]
            )
    
    def _check_package_versions(self) -> List[str]:
        """Check versions of critical packages."""
        version_recommendations = []
        
        try:
            # Check PyTorch version
            import torch
            torch_version = torch.__version__
            if torch_version.startswith("1."):
                version_recommendations.append(
                    "Consider upgrading PyTorch to version 2.0+ for better performance"
                )
        except ImportError:
            pass
        
        try:
            # Check transformers version
            import transformers
            transformers_version = transformers.__version__
            major_version = int(transformers_version.split('.')[0])
            if major_version < 4:
                version_recommendations.append(
                    "Upgrade transformers to version 4.0+ for compatibility"
                )
        except (ImportError, ValueError):
            pass
        
        return version_recommendations
    
    def _validate_directory_structure(self) -> ValidationResult:
        """Validate required directory structure."""
        try:
            self.logger.info("Validating directory structure")
            
            missing_dirs = []
            recommendations = []
            
            for directory in self.required_directories:
                dir_path = Path(directory)
                if not dir_path.exists():
                    missing_dirs.append(directory)
            
            if missing_dirs:
                recommendations.append(f"Create missing directories: {', '.join(missing_dirs)}")
                return ValidationResult(
                    valid=False,
                    message=f"Missing required directories: {', '.join(missing_dirs)}",
                    severity="warning",
                    recommendations=recommendations
                )
            else:
                return ValidationResult(
                    valid=True,
                    message="Directory structure validation passed",
                    severity="info"
                )
                
        except Exception as e:
            return ValidationResult(
                valid=False,
                message=f"Directory structure validation error: {e}",
                severity="error",
                recommendations=["Check file system permissions and accessibility"]
            )
    
    def _validate_component_functionality(self) -> ValidationResult:
        """Validate basic component functionality."""
        try:
            self.logger.info("Validating component functionality")
            
            test_failures = []
            recommendations = []
            
            # Test text processing
            try:
                from stylometric.text_processing import TextProcessor
                processor = TextProcessor()
                test_text = "This is a test poem line."
                result = processor.clean_text(test_text)
                if not result:
                    test_failures.append("Text processor returned empty result")
            except Exception as e:
                test_failures.append(f"Text processor test failed: {e}")
                recommendations.append("Check text processing module implementation")
            
            # Test lexical analysis
            try:
                from stylometric.lexical_analysis import LexicalAnalyzer
                analyzer = LexicalAnalyzer()
                test_text = "This is a test poem with multiple words."
                result = analyzer.calculate_lexical_metrics(test_text)
                if not result or not isinstance(result, dict):
                    test_failures.append("Lexical analyzer returned invalid result")
            except Exception as e:
                test_failures.append(f"Lexical analyzer test failed: {e}")
                recommendations.append("Check lexical analysis module implementation")
            
            # Test structural analysis
            try:
                from stylometric.structural_analysis import StructuralAnalyzer
                analyzer = StructuralAnalyzer()
                test_lines = ["Line one", "Line two", "Line three"]
                result = analyzer.analyze_structure(test_lines)
                if not result or not isinstance(result, dict):
                    test_failures.append("Structural analyzer returned invalid result")
            except Exception as e:
                test_failures.append(f"Structural analyzer test failed: {e}")
                recommendations.append("Check structural analysis module implementation")
            
            if test_failures:
                return ValidationResult(
                    valid=False,
                    message=f"Component functionality tests failed: {'; '.join(test_failures)}",
                    severity="error",
                    recommendations=recommendations
                )
            else:
                return ValidationResult(
                    valid=True,
                    message="Component functionality validation passed",
                    severity="info"
                )
                
        except Exception as e:
            return ValidationResult(
                valid=False,
                message=f"Component functionality validation error: {e}",
                severity="error",
                recommendations=["Check component module imports and implementations"]
            )
    
    def generate_validation_report(self, report: SystemValidationReport, output_path: Optional[Path] = None) -> str:
        """
        Generate a human-readable validation report.
        
        Args:
            report: SystemValidationReport to format
            output_path: Optional path to save the report
            
        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("STYLISTIC POETRY LLM SYSTEM VALIDATION REPORT")
        lines.append("=" * 60)
        lines.append(f"Timestamp: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Overall Status: {'✓ VALID' if report.overall_valid else '✗ INVALID'}")
        lines.append("")
        
        # Summary
        lines.append("VALIDATION SUMMARY")
        lines.append("-" * 20)
        lines.append(f"Configuration: {'✓' if report.configuration_valid else '✗'}")
        lines.append(f"Dependencies: {'✓' if report.dependencies_valid else '✗'}")
        lines.append(f"Directory Structure: {'✓' if report.directory_structure_valid else '✗'}")
        lines.append(f"Component Tests: {'✓' if report.component_tests_valid else '✗'}")
        lines.append("")
        
        # Detailed results
        lines.append("DETAILED VALIDATION RESULTS")
        lines.append("-" * 30)
        for result in report.validation_results:
            severity_marker = "✓" if result.valid else ("⚠️" if result.severity == "warning" else "✗")
            lines.append(f"{severity_marker} {result.message}")
            if result.recommendations:
                for rec in result.recommendations:
                    lines.append(f"  • {rec}")
        lines.append("")
        
        # Dependencies status
        if report.dependency_status:
            lines.append("DEPENDENCY STATUS")
            lines.append("-" * 18)
            for package, available in report.dependency_status.items():
                status = "✓" if available else "✗"
                lines.append(f"{status} {package}")
            lines.append("")
        
        # Recommendations
        if report.recommendations:
            lines.append("RECOMMENDATIONS")
            lines.append("-" * 15)
            for i, rec in enumerate(report.recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")
        
        lines.append("=" * 60)
        
        report_text = "\n".join(lines)
        
        # Save to file if requested
        if output_path:
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(report_text, encoding='utf-8')
                self.logger.info(f"Validation report saved to {output_path}")
            except Exception as e:
                self.logger.error(f"Failed to save validation report: {e}")
        
        return report_text
    
    def save_validation_report_json(self, report: SystemValidationReport, output_path: Path) -> None:
        """
        Save validation report as JSON.
        
        Args:
            report: SystemValidationReport to save
            output_path: Path to save JSON report
        """
        try:
            # Convert report to dictionary
            report_dict = {
                'timestamp': report.timestamp.isoformat(),
                'overall_valid': report.overall_valid,
                'configuration_valid': report.configuration_valid,
                'dependencies_valid': report.dependencies_valid,
                'directory_structure_valid': report.directory_structure_valid,
                'component_tests_valid': report.component_tests_valid,
                'validation_results': [
                    {
                        'valid': result.valid,
                        'message': result.message,
                        'severity': result.severity,
                        'recommendations': result.recommendations
                    }
                    for result in report.validation_results
                ],
                'dependency_status': report.dependency_status,
                'missing_dependencies': report.missing_dependencies,
                'configuration_issues': report.configuration_issues,
                'recommendations': report.recommendations
            }
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_dict, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"JSON validation report saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save JSON validation report: {e}")
            raise ValidationError(f"Failed to save validation report: {e}")


def validate_system_setup(config_path: Optional[Path] = None,
                         check_optional_deps: bool = False,
                         save_report: bool = False,
                         report_path: Optional[Path] = None) -> SystemValidationReport:
    """
    Convenience function to validate complete system setup.
    
    Args:
        config_path: Optional path to configuration file
        check_optional_deps: Whether to check optional dependencies
        save_report: Whether to save the validation report
        report_path: Optional path to save the report
        
    Returns:
        SystemValidationReport with validation results
    """
    validator = SystemValidator()
    report = validator.validate_complete_system(config_path, check_optional_deps)
    
    if save_report:
        if not report_path:
            report_path = Path("validation_report.txt")
        validator.generate_validation_report(report, report_path)
    
    return report