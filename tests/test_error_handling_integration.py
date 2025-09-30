"""
Integration tests for comprehensive error handling system (Task 8.1).

This module tests the complete error handling system integration including:
- End-to-end error handling workflows
- Recovery mechanism effectiveness
- User-friendly error message generation
- Logging and monitoring integration
"""

import pytest
import logging
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.utils.exceptions import (
    PoetryLLMError, DataQualityError, GenerationFailureError, 
    ResourceConstraintError, ValidationError, create_user_friendly_error_message
)
from src.utils.error_integration import (
    SystemErrorHandler, get_system_error_handler, handle_with_recovery, safe_execute
)


class TestEndToEndErrorHandling:
    """Test complete error handling workflows from user input to recovery."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.system_handler = SystemErrorHandler(log_errors=True, enable_recovery=True)
    
    def test_complete_generation_error_workflow(self):
        """Test complete error handling workflow for poetry generation."""
        # Test the complete error handling chain
        error = ValueError("Invalid temperature: 3.5")
        context = {
            'operation': 'poetry_generation',
            'request_params': {'temperature': 3.5, 'max_length': 200},
            'attempt_count': 1
        }
        
        result = self.system_handler.handle_generation_error(error, context)
        
        # Verify comprehensive error handling
        assert result['success'] is False
        assert 'error_message' in result
        assert 'suggestions' in result or 'recovery_suggestions' in result
    
    def test_data_quality_error_workflow(self):
        """Test complete error handling workflow for data quality issues."""
        # Create a corpus with quality issues
        problematic_poems = [
            {"text": "", "title": "Empty Poem", "poet": "TestPoet"},  # Empty
            {"text": "Short", "title": "Too Short", "poet": "TestPoet"},  # Too short
        ]
        
        # Test the complete data quality handling workflow
        context = {
            'operation': 'corpus_validation',
            'poems': problematic_poems,
            'poet_name': 'TestPoet'
        }
        
        # Simulate data quality error
        error = DataQualityError(
            "Corpus quality issues found",
            error_code="QUALITY_ISSUES",
            details={'poems': problematic_poems, 'poet_name': 'TestPoet'}
        )
        
        result = self.system_handler.handle_data_processing_error(error, context)
        
        # Verify comprehensive handling
        assert result['success'] is False
        assert 'recovery_suggestions' in result
        assert len(result['recovery_suggestions']) > 0


class TestRecoveryMechanisms:
    """Test the effectiveness of error recovery mechanisms."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.system_handler = SystemErrorHandler(log_errors=True, enable_recovery=True)
    
    def test_generation_parameter_recovery(self):
        """Test recovery from invalid generation parameters."""
        from src.utils.error_handlers import GenerationErrorHandler
        
        handler = GenerationErrorHandler()
        
        # Test temperature recovery
        error = ValueError("Invalid temperature: 3.5")
        request_params = {'temperature': 3.5, 'max_length': 200}
        
        result = handler.handle_generation_failure(error, request_params, 1)
        
        assert result['error_type'] == 'parameter_error'
        assert 'fallback_params' in result
        
        fallback_params = result['fallback_params']
        assert 0.0 <= fallback_params.get('temperature', 0.8) <= 2.0
        assert result['retry_recommended'] is True


class TestUserFriendlyMessaging:
    """Test user-friendly error message generation."""
    
    def test_technical_to_user_friendly_conversion(self):
        """Test conversion of technical errors to user-friendly messages."""
        test_cases = [
            (
                FileNotFoundError("No such file or directory: '/path/to/missing.txt'"),
                "Could not find the required file"
            ),
            (
                PermissionError("Permission denied: '/restricted/file.txt'"),
                "Permission denied"
            ),
            (
                ValueError("Invalid temperature: 3.5. Must be between 0.0 and 2.0"),
                "Invalid input provided"
            )
        ]
        
        for error, expected_phrase in test_cases:
            message = create_user_friendly_error_message(error, "test operation")
            assert expected_phrase.lower() in message.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])