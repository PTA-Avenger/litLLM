"""
Comprehensive error handling integration for the Stylistic Poetry LLM Framework.

This module provides centralized error handling that integrates all error handling
components and provides a unified interface for error management across the system.
"""

import logging
import traceback
from typing import Dict, List, Optional, Any, Union, Callable
from pathlib import Path
from datetime import datetime
import json

from .exceptions import (
    PoetryLLMError, DataQualityError, ModelError, GenerationFailureError,
    ResourceConstraintError, ValidationError, ErrorContext, ErrorRecoveryManager
)
from .error_handlers import (
    DataQualityValidator, GenerationErrorHandler, ModelLoadingErrorHandler,
    ComprehensiveErrorLogger
)
from .logging import get_logger

logger = get_logger(__name__)


class SystemErrorHandler:
    """
    Centralized error handling system that coordinates all error handling components.
    
    This class provides a unified interface for error handling across the entire
    Poetry LLM system, including data processing, model operations, and generation.
    """
    
    def __init__(self, log_errors: bool = True, enable_recovery: bool = True):
        """
        Initialize the system error handler.
        
        Args:
            log_errors: Whether to log errors comprehensively
            enable_recovery: Whether to attempt error recovery
        """
        self.log_errors = log_errors
        self.enable_recovery = enable_recovery
        
        # Initialize specialized handlers
        self.data_validator = DataQualityValidator()
        self.generation_handler = GenerationErrorHandler()
        self.model_handler = ModelLoadingErrorHandler()
        self.error_logger = ComprehensiveErrorLogger()
        self.recovery_manager = ErrorRecoveryManager(logger)
        
        # Error handling statistics
        self.error_stats = {
            'total_errors': 0,
            'recovered_errors': 0,
            'critical_errors': 0,
            'by_component': {},
            'by_error_type': {}
        }
        
        logger.info("System error handler initialized")
    
    def handle_data_processing_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle errors that occur during data processing operations.
        
        Args:
            error: The exception that occurred
            context: Context information about the operation
            
        Returns:
            Error handling result with recovery options
        """
        self._update_error_stats('data_processing', error)
        
        try:
            # Attempt data quality validation if applicable
            if isinstance(error, DataQualityError) or 'corpus' in str(error).lower():
                recovery_result = self._handle_data_quality_error(error, context)
            else:
                recovery_result = self._handle_generic_data_error(error, context)
            
            # Log the error with context
            if self.log_errors:
                self.error_logger.log_error_with_context(
                    error, context, 
                    recovery_attempted=True,
                    recovery_successful=recovery_result.get('success', False)
                )
            
            return recovery_result
            
        except Exception as handling_error:
            logger.error(f"Error handling failed: {handling_error}")
            return self._create_fallback_response(error, context)
    
    def handle_model_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle errors that occur during model operations.
        
        Args:
            error: The exception that occurred
            context: Context information about the operation
            
        Returns:
            Error handling result with recovery options
        """
        self._update_error_stats('model_operations', error)
        
        try:
            model_name = context.get('model_name', 'unknown')
            device = context.get('device', 'auto')
            
            if 'loading' in str(error).lower() or isinstance(error, ModelError):
                recovery_result = self.model_handler.handle_model_loading_error(error, model_name, device)
            else:
                recovery_result = self._handle_generic_model_error(error, context)
            
            # Log the error with context
            if self.log_errors:
                self.error_logger.log_error_with_context(
                    error, context,
                    recovery_attempted=True,
                    recovery_successful=recovery_result.get('success', False)
                )
            
            return recovery_result
            
        except Exception as handling_error:
            logger.error(f"Model error handling failed: {handling_error}")
            return self._create_fallback_response(error, context)
    
    def handle_generation_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle errors that occur during poetry generation.
        
        Args:
            error: The exception that occurred
            context: Context information about the operation
            
        Returns:
            Error handling result with recovery options
        """
        self._update_error_stats('generation', error)
        
        try:
            request_params = context.get('request_params', {})
            attempt_count = context.get('attempt_count', 1)
            
            recovery_result = self.generation_handler.handle_generation_failure(
                error, request_params, attempt_count
            )
            
            # Log the error with context
            if self.log_errors:
                self.error_logger.log_error_with_context(
                    error, context,
                    recovery_attempted=True,
                    recovery_successful=recovery_result.get('success', False)
                )
            
            return recovery_result
            
        except Exception as handling_error:
            logger.error(f"Generation error handling failed: {handling_error}")
            return self._create_fallback_response(error, context)
    
    def handle_validation_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle validation errors with specific feedback.
        
        Args:
            error: The validation exception that occurred
            context: Context information about what was being validated
            
        Returns:
            Error handling result with validation feedback
        """
        self._update_error_stats('validation', error)
        
        validation_type = context.get('validation_type', 'unknown')
        invalid_data = context.get('data', {})
        
        recovery_result = {
            'success': False,
            'error_type': 'validation_error',
            'validation_type': validation_type,
            'error_message': str(error),
            'invalid_data': invalid_data,
            'corrections': self._generate_validation_corrections(error, context),
            'retry_recommended': True
        }
        
        # Log validation error
        if self.log_errors:
            self.error_logger.log_error_with_context(
                error, context,
                recovery_attempted=True,
                recovery_successful=False
            )
        
        return recovery_result
    
    def create_error_context(self, operation: str, **kwargs) -> ErrorContext:
        """
        Create an error context manager for consistent error handling.
        
        Args:
            operation: Description of the operation being performed
            **kwargs: Additional context parameters
            
        Returns:
            ErrorContext manager configured for this system
        """
        return ErrorContext(
            logger=logger,
            context=operation,
            reraise=True,
            error_type=PoetryLLMError,
            recovery_manager=self.recovery_manager if self.enable_recovery else None
        )
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error handling statistics."""
        stats = self.error_stats.copy()
        stats['recovery_rate'] = (
            stats['recovered_errors'] / max(stats['total_errors'], 1) * 100
        )
        stats['logger_stats'] = self.error_logger.get_error_statistics()
        return stats
    
    def reset_statistics(self):
        """Reset error handling statistics."""
        self.error_stats = {
            'total_errors': 0,
            'recovered_errors': 0,
            'critical_errors': 0,
            'by_component': {},
            'by_error_type': {}
        }
        logger.info("Error handling statistics reset")
    
    def _handle_data_quality_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data quality specific errors."""
        poems = context.get('poems', [])
        poet_name = context.get('poet_name', 'Unknown')
        
        if poems:
            try:
                validation_result = self.data_validator.validate_corpus_quality(poems, poet_name)
                return {
                    'success': False,
                    'error_type': 'data_quality',
                    'validation_result': validation_result,
                    'recovery_suggestions': validation_result.get('recommendations', [])
                }
            except Exception:
                pass
        
        return {
            'success': False,
            'error_type': 'data_quality',
            'error_message': str(error),
            'recovery_suggestions': [
                "Check data format and encoding",
                "Verify corpus contains valid poetry",
                "Ensure sufficient data volume"
            ]
        }
    
    def _handle_generic_data_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle generic data processing errors."""
        return {
            'success': False,
            'error_type': 'data_processing',
            'error_message': str(error),
            'context': context,
            'recovery_suggestions': [
                "Check file paths and permissions",
                "Verify data format",
                "Ensure sufficient disk space",
                "Check file encoding (UTF-8 recommended)"
            ]
        }
    
    def _handle_generic_model_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle generic model operation errors."""
        return {
            'success': False,
            'error_type': 'model_error',
            'error_message': str(error),
            'context': context,
            'recovery_suggestions': [
                "Check model configuration",
                "Verify system resources",
                "Try a different model",
                "Check network connectivity"
            ]
        }
    
    def _generate_validation_corrections(self, error: Exception, context: Dict[str, Any]) -> List[str]:
        """Generate specific corrections for validation errors."""
        corrections = []
        error_str = str(error).lower()
        
        if 'prompt' in error_str:
            corrections.extend([
                "Ensure prompt is not empty",
                "Check prompt length (reasonable size)",
                "Remove special characters if causing issues"
            ])
        
        if 'parameter' in error_str:
            corrections.extend([
                "Verify parameter ranges (temperature: 0.0-2.0)",
                "Check numeric parameter validity",
                "Use default values if unsure"
            ])
        
        if 'format' in error_str:
            corrections.extend([
                "Check data format specification",
                "Verify required fields are present",
                "Ensure proper data structure"
            ])
        
        return corrections or ["Review input data and try again"]
    
    def _create_fallback_response(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a fallback response when error handling fails."""
        return {
            'success': False,
            'error_type': 'system_error',
            'error_message': str(error),
            'context': context,
            'fallback_response': True,
            'recovery_suggestions': [
                "Check system logs for details",
                "Restart the application",
                "Contact support if issue persists"
            ]
        }
    
    def _update_error_stats(self, component: str, error: Exception):
        """Update error handling statistics."""
        self.error_stats['total_errors'] += 1
        
        # Track by component
        if component not in self.error_stats['by_component']:
            self.error_stats['by_component'][component] = 0
        self.error_stats['by_component'][component] += 1
        
        # Track by error type
        error_type = type(error).__name__
        if error_type not in self.error_stats['by_error_type']:
            self.error_stats['by_error_type'][error_type] = 0
        self.error_stats['by_error_type'][error_type] += 1
        
        # Track critical errors
        if isinstance(error, (ResourceConstraintError, ModelError)):
            self.error_stats['critical_errors'] += 1


# Global error handler instance
_system_error_handler = None


def get_system_error_handler() -> SystemErrorHandler:
    """Get the global system error handler instance."""
    global _system_error_handler
    if _system_error_handler is None:
        _system_error_handler = SystemErrorHandler()
    return _system_error_handler


def initialize_error_handling(log_errors: bool = True, enable_recovery: bool = True):
    """Initialize the global error handling system."""
    global _system_error_handler
    _system_error_handler = SystemErrorHandler(log_errors, enable_recovery)
    logger.info("Global error handling system initialized")


# Convenience functions for common error handling patterns
def handle_with_recovery(operation: str, func: Callable, *args, **kwargs) -> Any:
    """
    Execute a function with comprehensive error handling and recovery.
    
    Args:
        operation: Description of the operation
        func: Function to execute
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Function result or error recovery result
    """
    error_handler = get_system_error_handler()
    
    try:
        with error_handler.create_error_context(operation):
            return func(*args, **kwargs)
    except Exception as e:
        # Determine error handling strategy based on operation type
        context = {
            'operation': operation,
            'function': func.__name__,
            'args': str(args)[:200],  # Truncate for logging
            'kwargs': {k: str(v)[:100] for k, v in kwargs.items()}  # Truncate values
        }
        
        if 'data' in operation.lower() or 'corpus' in operation.lower():
            return error_handler.handle_data_processing_error(e, context)
        elif 'model' in operation.lower():
            return error_handler.handle_model_error(e, context)
        elif 'generat' in operation.lower():
            return error_handler.handle_generation_error(e, context)
        elif 'validat' in operation.lower():
            return error_handler.handle_validation_error(e, context)
        else:
            # Generic error handling
            logger.error(f"Unhandled error in {operation}: {e}")
            raise


def safe_execute(operation: str, func: Callable, fallback_value: Any = None, 
                reraise: bool = False) -> Any:
    """
    Safely execute a function with error handling but no recovery attempts.
    
    Args:
        operation: Description of the operation
        func: Function to execute
        fallback_value: Value to return if function fails
        reraise: Whether to reraise exceptions after logging
        
    Returns:
        Function result or fallback value
    """
    try:
        return func()
    except Exception as e:
        logger.error(f"Error in {operation}: {e}")
        if reraise:
            raise
        return fallback_value