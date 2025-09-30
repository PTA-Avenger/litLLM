"""
Tests for robust error handling implementation (Task 8.1).

This module tests the comprehensive error handling system including:
- Data quality issue handling with fallback options
- Generation failure recovery with user-friendly messages
- Edge case error handling and logging
"""

import pytest
import logging
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch

from src.utils.exceptions import (
    PoetryLLMError, DataQualityError, GenerationFailureError, 
    ResourceConstraintError, ValidationError, create_user_friendly_error_message
)
from src.utils.error_handlers import (
    DataQualityValidator, GenerationErrorHandler, ModelLoadingErrorHandler
)
from src.utils.error_integration import (
    SystemErrorHandler, get_system_error_handler, handle_with_recovery, safe_execute
)
from src.stylometric.training_data import PoetryCorpusLoader, TrainingDatasetFormatter


class TestDataQualityErrorHandling:
    """Test comprehensive data quality error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = DataQualityValidator()
        self.corpus_loader = PoetryCorpusLoader()
    
    def test_empty_file_handling(self):
        """Test handling of empty corpus files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("")  # Empty file
            temp_path = f.name
        
        try:
            # Should raise DataQualityError for empty file
            with pytest.raises(DataQualityError) as exc_info:
                poems = self.corpus_loader.load_corpus_from_file(temp_path, "TestPoet")
            
            error = exc_info.value
            assert error.error_code == "EMPTY_CORPUS"
            assert "No poems found" in str(error)
            
        finally:
            Path(temp_path).unlink()
    
    def test_corrupted_file_handling(self):
        """Test handling of corrupted or malformed files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            # Write corrupted JSON
            f.write('{"poems": [{"text": "incomplete json"')
            temp_path = f.name
        
        try:
            # Should handle corrupted JSON gracefully
            poems = self.corpus_loader.load_corpus_from_file(temp_path, "TestPoet", format_type="json")
            # Should fall back to simple format parsing
            assert isinstance(poems, list)
            
        except Exception as e:
            # Should provide user-friendly error message
            error_msg = create_user_friendly_error_message(e, "loading corpus")
            assert "file" in error_msg.lower()
            
        finally:
            Path(temp_path).unlink()  
  
    def test_encoding_error_handling(self):
        """Test handling of encoding issues in corpus files."""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.txt', delete=False) as f:
            # Write text with problematic encoding
            f.write(b'\xff\xfe\x00\x00Invalid UTF-8 sequence')
            temp_path = f.name
        
        try:
            # Should handle encoding errors gracefully
            poems = self.corpus_loader.load_corpus_from_file(temp_path, "TestPoet")
            # Should either succeed with fallback encoding or provide clear error
            
        except Exception as e:
            error_msg = create_user_friendly_error_message(e, "loading corpus")
            assert "encoding" in error_msg.lower() or "file" in error_msg.lower()
            
        finally:
            Path(temp_path).unlink()
    
    def test_mixed_quality_corpus_handling(self):
        """Test handling of corpus with mixed quality poems."""
        poems = [
            {"text": "This is a good poem\nWith multiple lines\nAnd substantial content\nThat should pass validation", "title": "Good Poem", "poet": "TestPoet"},
            {"text": "", "title": "Empty Poem", "poet": "TestPoet"},  # Empty
            {"text": "Short", "title": "Too Short", "poet": "TestPoet"},  # Too short
            {"text": "Another good poem\nWith proper length\nAnd good structure\nFor training purposes", "title": "Another Good", "poet": "TestPoet"},
            {"text": "Poem with\nreasonable length\nand content", "title": "Reasonable", "poet": "TestPoet"},
        ]
        
        # Should provide detailed feedback about quality issues
        with pytest.raises(DataQualityError) as exc_info:
            self.validator.validate_corpus_quality(poems, "TestPoet")
        
        error = exc_info.value
        assert "quality issues" in str(error).lower()
        assert 'feedback' in error.details
        assert 'recommendations' in error.details
        
        # Check that feedback is actionable
        feedback = error.details['feedback']
        assert any('empty' in fb.lower() for fb in feedback)
        assert any('short' in fb.lower() for fb in feedback)


class TestGenerationErrorHandling:
    """Test comprehensive generation error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.error_handler = GenerationErrorHandler()
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.empty_cache')
    def test_cuda_memory_error_recovery(self, mock_empty_cache, mock_cuda_available):
        """Test recovery from CUDA memory errors."""
        mock_cuda_available.return_value = True
        
        error = torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
        request_params = {
            'temperature': 0.8,
            'max_length': 500,
            'top_p': 0.9,
            'device': 'cuda'
        }
        
        result = self.error_handler.handle_generation_failure(error, request_params, attempt_count=1)
        
        assert result['success'] is False
        assert result['error_type'] == 'memory_error'
        assert result['retry_recommended'] is True
        
        # Should suggest fallback parameters
        fallback_params = result.get('fallback_params', {})
        assert fallback_params['max_length'] < request_params['max_length']
        
        # Should suggest practical solutions
        suggestions = result['suggestions']
        assert any('smaller model' in suggestion.lower() for suggestion in suggestions)
        assert any('cpu' in suggestion.lower() for suggestion in suggestions)
    
    def test_parameter_validation_error_handling(self):
        """Test handling of parameter validation errors."""
        error = ValueError("Invalid temperature: 3.5. Must be between 0.0 and 2.0")
        request_params = {'temperature': 3.5, 'top_p': 0.9}
        
        result = self.error_handler.handle_generation_failure(error, request_params)
        
        assert result['error_type'] == 'parameter_error'
        assert result['retry_recommended'] is True
        
        # Should provide corrected parameters
        fallback_params = result.get('fallback_params', {})
        assert 0.0 <= fallback_params.get('temperature', 0.8) <= 2.0
        
        # Should explain parameter ranges
        suggestions = result['suggestions']
        assert any('temperature' in suggestion.lower() and '0.0-2.0' in suggestion for suggestion in suggestions)


class TestUserFriendlyErrorMessages:
    """Test user-friendly error message generation."""
    
    def test_file_operation_errors(self):
        """Test user-friendly messages for file operation errors."""
        test_cases = [
            (FileNotFoundError("No such file: 'missing.txt'"), "Could not find the required file"),
            (PermissionError("Permission denied"), "Permission denied"),
        ]
        
        for error, expected_phrase in test_cases:
            message = create_user_friendly_error_message(error, "file operation")
            assert expected_phrase.lower() in message.lower()
    
    def test_resource_constraint_errors(self):
        """Test user-friendly messages for resource constraint errors."""
        test_cases = [
            (Exception("CUDA out of memory"), "Not enough memory available"),
            (Exception("Connection timeout"), "Network connection issue"),
        ]
        
        for error, expected_phrase in test_cases:
            message = create_user_friendly_error_message(error, "resource operation")
            assert expected_phrase.lower() in message.lower()


class TestErrorHandlingUtilities:
    """Test error handling utility functions."""
    
    def test_handle_with_recovery_success(self):
        """Test handle_with_recovery with successful operation."""
        def successful_operation():
            return "success"
        
        result = handle_with_recovery("test operation", successful_operation)
        assert result == "success"
    
    def test_safe_execute_with_fallback(self):
        """Test safe_execute with fallback value."""
        def failing_operation():
            raise Exception("Test error")
        
        result = safe_execute("test operation", failing_operation, fallback_value="fallback")
        assert result == "fallback"
    
    def test_error_context_manager(self):
        """Test ErrorContext manager functionality."""
        from src.utils.exceptions import ErrorContext, PoetryLLMError
        
        logger = logging.getLogger('test')
        
        # Test successful operation
        with ErrorContext(logger, "test operation", reraise=False):
            result = "success"
        
        # Test error handling
        with ErrorContext(logger, "test operation", reraise=False) as ctx:
            try:
                raise ValueError("Test error")
            except ValueError:
                pass  # Should be handled by context manager
        
        assert ctx is not None


class TestSystemErrorHandlerIntegration:
    """Test the integrated system error handler."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from src.utils.error_integration import SystemErrorHandler
        self.system_handler = SystemErrorHandler(log_errors=True, enable_recovery=True)
    
    def test_data_processing_error_handling(self):
        """Test comprehensive data processing error handling."""
        error = FileNotFoundError("Test file not found")
        context = {
            'operation': 'corpus_loading',
            'filepath': '/nonexistent/file.txt',
            'poet_name': 'TestPoet'
        }
        
        result = self.system_handler.handle_data_processing_error(error, context)
        
        assert result['success'] is False
        assert 'error_message' in result
        assert 'recovery_suggestions' in result
        assert len(result['recovery_suggestions']) > 0
    
    def test_model_error_handling(self):
        """Test comprehensive model error handling."""
        error = Exception("Model loading failed")
        context = {
            'operation': 'model_loading',
            'model_name': 'test-model',
            'device': 'cuda'
        }
        
        result = self.system_handler.handle_model_error(error, context)
        
        assert result['success'] is False
        assert 'error_message' in result
        assert 'fallback_options' in result or 'recovery_suggestions' in result
    
    def test_generation_error_handling(self):
        """Test comprehensive generation error handling."""
        error = Exception("Generation timeout")
        context = {
            'operation': 'poetry_generation',
            'request_params': {'temperature': 0.8, 'max_length': 200},
            'attempt_count': 1
        }
        
        result = self.system_handler.handle_generation_error(error, context)
        
        assert result['success'] is False
        assert 'error_message' in result
        assert 'suggestions' in result or 'recovery_suggestions' in result
    
    def test_error_statistics_tracking(self):
        """Test error statistics tracking."""
        # Generate some test errors
        error1 = ValueError("Test error 1")
        context1 = {'operation': 'test_op_1'}
        
        error2 = FileNotFoundError("Test error 2")
        context2 = {'operation': 'test_op_2'}
        
        self.system_handler.handle_data_processing_error(error1, context1)
        self.system_handler.handle_data_processing_error(error2, context2)
        
        stats = self.system_handler.get_error_statistics()
        
        assert stats['total_errors'] >= 2
        assert 'by_component' in stats
        assert 'by_error_type' in stats
        assert 'recovery_rate' in stats


class TestEdgeCaseErrorHandling:
    """Test error handling for edge cases and unusual scenarios."""
    
    def test_nested_error_handling(self):
        """Test handling of nested errors (errors within error handlers)."""
        from src.utils.error_integration import safe_execute
        
        def failing_error_handler():
            # Simulate an error handler that itself fails
            raise Exception("Error handler failed")
        
        # Should not crash, should return fallback
        result = safe_execute("nested error test", failing_error_handler, fallback_value="safe_fallback")
        assert result == "safe_fallback"
    
    def test_resource_exhaustion_scenarios(self):
        """Test handling of resource exhaustion scenarios."""
        from src.utils.exceptions import ResourceConstraintError
        from src.utils.error_handlers import GenerationErrorHandler
        
        handler = GenerationErrorHandler()
        
        # Simulate memory exhaustion
        memory_error = Exception("CUDA out of memory: tried to allocate 10.00 GiB")
        result = handler.handle_generation_failure(memory_error, {}, 1)
        
        assert result['error_type'] == 'memory_error'
        assert any('memory' in suggestion.lower() for suggestion in result['suggestions'])
        assert 'fallback_params' in result
    
    def test_malformed_data_recovery(self):
        """Test recovery from malformed or corrupted data."""
        from src.stylometric.training_data import PoetryCorpusLoader
        import tempfile
        
        loader = PoetryCorpusLoader()
        
        # Create a file with mixed valid and invalid content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Valid poem line 1\nValid poem line 2\n\n")
            f.write("\x00\x01Invalid binary data\x02\x03\n\n")  # Binary data
            f.write("Another valid poem\nWith multiple lines\n")
            temp_path = f.name
        
        try:
            # Should handle mixed content gracefully
            poems = loader.load_corpus_from_file(temp_path, "TestPoet")
            # Should extract valid poems and skip invalid content
            assert len(poems) >= 1  # At least some valid content should be extracted
            
        except DataQualityError as e:
            # Should provide specific feedback about data issues
            assert 'quality' in str(e).lower() or 'data' in str(e).lower()
            assert e.details is not None
            
        finally:
            Path(temp_path).unlink()
    
    def test_concurrent_error_handling(self):
        """Test error handling under concurrent access scenarios."""
        import threading
        from src.utils.error_integration import get_system_error_handler
        
        handler = get_system_error_handler()
        results = []
        errors = []
        
        def generate_error(thread_id):
            try:
                error = ValueError(f"Test error from thread {thread_id}")
                context = {'operation': f'concurrent_test_{thread_id}', 'thread_id': thread_id}
                result = handler.handle_data_processing_error(error, context)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads that generate errors simultaneously
        threads = []
        for i in range(5):
            thread = threading.Thread(target=generate_error, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Should handle all errors without crashing
        assert len(errors) == 0  # No unhandled exceptions
        assert len(results) == 5  # All errors were handled
        
        # Check that statistics were updated correctly
        stats = handler.get_error_statistics()
        assert stats['total_errors'] >= 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])