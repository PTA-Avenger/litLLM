"""
Comprehensive tests for the error handling system.

This module tests all aspects of the error handling framework including
exceptions, recovery strategiend resource constraint handling.
"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json

from src.utils.exceptions import (
    PoetryLLMError, DataQualityError, ResourceConstraintError, 
    GenerationFailureError, ErrorRecoveryManager, create_user_friendly_error_message
)
from src.utils.error_handlers import DataQualityValidator, GenerationErrorHandler


class TestErrorRecoveryManager:
    """Test the ErrorRecoveryManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.logger = Mock()
        self.recovery_manager = ErrorRecoveryManager(self.logger)
    
    def test_generation_failure_recovery(self):
        """Test recovery from generation failures."""
        error = Exception("CUDA out of memory")
        
        result = self.recovery_manager.recover_from_error(error, 'generation_failure')
        
        assert result['success'] is False
        assert 'recovery_suggestions' in result
        assert any('CPU' in suggestion for suggestion in result['recovery_suggestions'])
        assert any('smaller model' in suggestion for suggestion in result['recovery_suggestions'])
    
    def test_data_quality_recovery(self):
        """Test recovery from data quality issues."""
        error = DataQualityError("Insufficient poems in corpus")
        quality_metrics = {
            'total_poems': 3,
            'empty_poems': 1,
            'avg_lines_per_poem': 2.5
        }
        
        result = self.recovery_manager.recover_from_error(
            error, 'data_quality', quality_metrics=quality_metrics
        )
        
        assert result['success'] is False
        assert 'quality_feedback' in result
        assert result['corpus_rejected'] is True
        assert any('10 poems' in feedback for feedback in result['quality_feedback'])
    
    def test_resource_constraint_recovery(self):
        """Test recovery from resource constraints."""
        error = Exception("Not enough memory")
        
        result = self.recovery_manager.recover_from_error(error, 'resource_constraint')
        
        assert result['success'] is False
        assert 'resource_options' in result
        assert result['constraint_type'] == 'memory'
        assert any('smaller model' in option for option in result['resource_options'])
    
    def test_model_loading_failure_recovery(self):
        """Test recovery from model loading failures."""
        error = Exception("Model 'invalid-model' not found")
        
        result = self.recovery_manager.recover_from_error(
            error, 'model_loading', model_name='invalid-model'
        )
        
        assert result['success'] is False
        assert 'model_suggestions' in result
        assert 'alternative_models' in result
        assert 'gpt2' in result['alternative_models']
    
    def test_validation_error_recovery(self):
        """Test recovery from validation errors."""
        error = ValueError("Invalid temperature parameter: 3.0")
        
        result = self.recovery_manager.recover_from_error(error, 'validation_error')
        
        assert result['success'] is False
        assert 'validation_fixes' in result
        assert any('temperature' in fix for fix in result['validation_fixes'])


class TestDataQualityValidator:
    """Test the DataQualityValidator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = DataQualityValidator()
    
    def test_empty_corpus_validation(self):
        """Test validation of empty corpus."""
        poems = []
        
        with pytest.raises(DataQualityError) as exc_info:
            self.validator.validate_corpus_quality(poems, "TestPoet")
        
        assert "No poems found" in str(exc_info.value)
        assert exc_info.value.error_code == "EMPTY_CORPUS"
    
    def test_insufficient_poems_validation(self):
        """Test validation of corpus with too few poems."""
        poems = [
            {"text": "Short poem\nTwo lines", "title": "Poem 1", "poet": "TestPoet"},
            {"text": "Another short\nPoem here", "title": "Poem 2", "poet": "TestPoet"}
        ]
        
        with pytest.raises(DataQualityError) as exc_info:
            self.validator.validate_corpus_quality(poems, "TestPoet")
        
        assert "Critical data quality issues" in str(exc_info.value)
        assert exc_info.value.error_code == "CRITICAL_QUALITY_ISSUES"
    
    def test_valid_corpus_validation(self):
        """Test validation of a valid corpus."""
        poems = []
        for i in range(10):
            poems.append({
                "text": f"This is poem number {i+1}\nWith multiple lines here\nAnd some more content\nTo make it substantial",
                "title": f"Poem {i+1}",
                "poet": "TestPoet"
            })
        
        result = self.validator.validate_corpus_quality(poems, "TestPoet")
        
        assert result['valid'] is True
        assert result['severity'] == 'none'
        assert result['metrics']['total_poems'] == 10
        assert len(result['issues']) == 0
    
    def test_quality_metrics_calculation(self):
        """Test calculation of quality metrics."""
        poems = [
            {"text": "Line one\nLine two\nLine three", "title": "Poem 1", "poet": "TestPoet"},
            {"text": "", "title": "Empty", "poet": "TestPoet"},  # Empty poem
            {"text": "Short", "title": "Short", "poet": "TestPoet"}  # Very short poem
        ]
        
        metrics = self.validator._calculate_quality_metrics(poems)
        
        assert metrics['total_poems'] == 3
        assert metrics['empty_poems'] == 1
        assert metrics['short_poems'] == 2  # Both empty and "Short" are considered short
        assert metrics['total_lines'] == 4  # 3 + 0 + 1
        assert metrics['vocabulary_size'] > 0
    
    def test_quality_feedback_generation(self):
        """Test generation of quality feedback."""
        metrics = {
            'total_poems': 3,
            'empty_poems': 1,
            'short_poems': 2,
            'avg_lines_per_poem': 2.0,
            'total_words': 50,
            'vocabulary_size': 80,
            'formatting_issues': 1
        }
        
        issues = [
            {'type': 'insufficient_data', 'value': 3, 'threshold': 5},
            {'type': 'empty_content', 'severity': 'critical'},
            {'type': 'predominantly_short', 'severity': 'warning'}
        ]
        
        feedback = self.validator._generate_quality_feedback(metrics, issues)
        
        assert len(feedback) > 0
        assert any('Add more poems' in fb for fb in feedback)
        assert any('Remove 1 empty poems' in fb for fb in feedback)
        assert any('formatting' in fb for fb in feedback)
    
    def test_severity_assessment(self):
        """Test assessment of issue severity."""
        critical_issues = [{'severity': 'critical'}, {'severity': 'warning'}]
        warning_issues = [{'severity': 'warning'}, {'severity': 'minor'}]
        minor_issues = [{'severity': 'minor'}]
        no_issues = []
        
        assert self.validator._assess_severity(critical_issues) == 'critical'
        assert self.validator._assess_severity(warning_issues) == 'warning'
        assert self.validator._assess_severity(minor_issues) == 'minor'
        assert self.validator._assess_severity(no_issues) == 'none'


class TestGenerationErrorHandler:
    """Test the GenerationErrorHandler class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.handler = GenerationErrorHandler()
    
    def test_memory_error_handling(self):
        """Test handling of memory errors."""
        error = Exception("CUDA out of memory")
        request_params = {'temperature': 0.8, 'max_length': 500}
        
        result = self.handler.handle_generation_failure(error, request_params)
        
        assert result['success'] is False
        assert result['error_type'] == 'memory_error'
        assert result['retry_recommended'] is True
        assert 'smaller model' in ' '.join(result['suggestions'])
        assert result['fallback_params'] is not None
    
    def test_gpu_error_handling(self):
        """Test handling of GPU errors."""
        error = Exception("CUDA device not available")
        request_params = {'temperature': 0.8}
        
        result = self.handler.handle_generation_failure(error, request_params)
        
        assert result['error_type'] == 'gpu_error'
        assert any('CPU' in suggestion for suggestion in result['suggestions'])
        assert result['fallback_params'] is not None
    
    def test_timeout_error_handling(self):
        """Test handling of timeout errors."""
        error = Exception("Generation timeout after 60 seconds")
        request_params = {'max_length': 1000, 'temperature': 1.5}
        
        result = self.handler.handle_generation_failure(error, request_params)
        
        assert result['error_type'] == 'timeout_error'
        assert any('max_length' in suggestion for suggestion in result['suggestions'])
        assert any('greedy decoding' in suggestion for suggestion in result['suggestions'])
    
    def test_model_error_handling(self):
        """Test handling of model errors."""
        error = Exception("Model 'invalid-model' not found")
        request_params = {}
        
        result = self.handler.handle_generation_failure(error, request_params)
        
        assert result['error_type'] == 'parameter_error'  # "invalid" in message triggers parameter error
        assert result['retry_recommended'] is True  # Parameter errors are retryable with corrected params
        assert any('parameter' in suggestion for suggestion in result['suggestions'])
    
    def test_parameter_error_handling(self):
        """Test handling of parameter errors."""
        error = ValueError("Invalid temperature: 3.0")
        request_params = {'temperature': 3.0}
        
        result = self.handler.handle_generation_failure(error, request_params)
        
        assert result['error_type'] == 'parameter_error'
        assert any('parameter ranges' in suggestion for suggestion in result['suggestions'])
    
    def test_fallback_params_creation(self):
        """Test creation of fallback parameters."""
        error = Exception("CUDA out of memory")
        original_params = {
            'temperature': 1.5,
            'max_length': 500,
        }
        
        result = self.handler.handle_generation_failure(error, original_params)
        fallback = result['fallback_params']
        
        assert fallback['temperature'] <= original_params['temperature']
        assert fallback['max_length'] < original_params['max_length']
    
    def test_progressive_retry_strategy(self):
        """Test progressive retry strategy."""
        error = Exception("Memory error")
        request_params = {}
        
        # First attempt
        result1 = self.handler.handle_generation_failure(error, request_params, attempt_count=1)
        assert result1['retry_recommended'] is True
        
        # Second attempt
        result2 = self.handler.handle_generation_failure(error, request_params, attempt_count=2)
        assert result2['retry_recommended'] is True
        
        # Third attempt (should reach max)
        result3 = self.handler.handle_generation_failure(error, request_params, attempt_count=3)
        assert result3['retry_recommended'] is False


class TestUserFriendlyErrorMessages:
    """Test user-friendly error message generation."""
    
    def test_file_not_found_message(self):
        """Test file not found error message."""
        error = FileNotFoundError("No such file or directory: 'missing.txt'")
        message = create_user_friendly_error_message(error, "loading corpus")
        
        assert "Could not find the required file" in message
        assert "check the file path" in message
    
    def test_permission_denied_message(self):
        """Test permission denied error message."""
        error = PermissionError("Permission denied")
        message = create_user_friendly_error_message(error)
        
        assert "Permission denied" in message
        assert "check file permissions" in message
    
    def test_memory_error_message(self):
        """Test memory error message."""
        error = Exception("CUDA out of memory")
        message = create_user_friendly_error_message(error, "model loading")
        
        assert "Not enough memory available" in message
        assert "smaller model" in message
    
    def test_network_error_message(self):
        """Test network error message."""
        error = Exception("Connection timeout")
        message = create_user_friendly_error_message(error)
        
        assert "Network connection issue" in message
        assert "internet connection" in message
    
    def test_generic_error_message(self):
        """Test generic error message fallback."""
        error = Exception("Some unexpected error")
        message = create_user_friendly_error_message(error, "testing")
        
        assert "An error occurred during testing" in message
        assert "Some unexpected error" in message


class TestErrorHandlingIntegration:
    """Integration tests for error handling across components."""
    
    def test_data_quality_error_chain(self):
        """Test error handling chain for data quality issues."""
        validator = DataQualityValidator()
        
        # Create problematic corpus
        poems = [
            {"text": "", "title": "Empty", "poet": "TestPoet"},  # Empty
            {"text": "Short", "title": "Short", "poet": "TestPoet"}  # Too short
        ]
        
        with pytest.raises(DataQualityError) as exc_info:
            validator.validate_corpus_quality(poems, "TestPoet")
        
        error = exc_info.value
        assert error.error_code == "CRITICAL_QUALITY_ISSUES"
        assert 'metrics' in error.details
        assert 'feedback' in error.details
        
        # Test that the error contains actionable feedback
        feedback = error.details['feedback']
        assert len(feedback) > 0
        assert any('Add more poems' in fb for fb in feedback)
    
    def test_generation_error_with_recovery(self):
        """Test generation error handling with recovery suggestions."""
        handler = GenerationErrorHandler()
        
        # Simulate a memory error during generation
        error = Exception("RuntimeError: CUDA out of memory")
        params = {
            'temperature': 1.0,
            'max_length': 300,
            'poet_style': 'emily_dickinson'
        }
        
        result = handler.handle_generation_failure(error, params)
        
        # Verify comprehensive error handling
        assert result['success'] is False
        assert result['error_type'] == 'memory_error'
        assert len(result['suggestions']) > 0
        assert result['fallback_params'] is not None
        assert result['retry_recommended'] is True
        
        # Verify fallback params are more conservative
        fallback = result['fallback_params']
        assert fallback['temperature'] <= params['temperature']
        assert fallback['max_length'] < params['max_length']
        # Note: poet_style is not preserved in fallback params as it's not a generation parameter
    
    def test_resource_constraint_handling_basic(self):
        """Test basic resource constraint handling without psutil dependency."""
        from src.utils.error_handlers import ResourceConstraintHandler
        handler = ResourceConstraintHandler()
        
        error = Exception("Not enough memory")
        result = handler.handle_resource_constraint(error, "model_loading")
        
        assert result['success'] is False
        assert result['constraint_type'] == 'memory'
        assert 'alternatives' in result
        assert len(result['alternatives']) > 0


if __name__ == '__main__':
    pytest.main([__file__])