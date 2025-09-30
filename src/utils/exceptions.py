"""Custom exceptions for the Stylistic Poetry LLM Framework."""

from typing import Optional, Any, List


class PoetryLLMError(Exception):
    """Base exception for all Poetry LLM related errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ConfigurationError(PoetryLLMError):
    """Raised when there's an issue with configuration."""
    pass


class DataProcessingError(PoetryLLMError):
    """Raised when there's an issue with data processing."""
    pass


class ModelError(PoetryLLMError):
    """Raised when there's an issue with model operations."""
    pass


class TrainingError(PoetryLLMError):
    """Raised when there's an issue during training."""
    pass


class GenerationError(PoetryLLMError):
    """Raised when there's an issue during poetry generation."""
    pass


class EvaluationError(PoetryLLMError):
    """Raised when there's an issue during evaluation."""
    pass


class StyleAnalysisError(PoetryLLMError):
    """Raised when there's an issue with stylometric analysis."""
    pass


class ValidationError(PoetryLLMError):
    """Raised when data validation fails."""
    pass


def handle_error(
    error: Exception,
    logger,
    context: str = "",
    reraise: bool = True,
    fallback_value: Any = None
) -> Any:
    """
    Generic error handler with logging and optional fallback.
    
    Args:
        error: The exception that occurred
        logger: Logger instance to use for logging
        context: Additional context about where the error occurred
        reraise: Whether to reraise the exception after logging
        fallback_value: Value to return if not reraising
        
    Returns:
        fallback_value if reraise is False, otherwise raises the exception
    """
    error_msg = f"{context}: {str(error)}" if context else str(error)
    
    if isinstance(error, PoetryLLMError):
        logger.error(f"Poetry LLM Error - {error_msg}")
        if error.details:
            logger.debug(f"Error details: {error.details}")
    else:
        logger.error(f"Unexpected error - {error_msg}", exc_info=True)
    
    if reraise:
        raise error
    
    return fallback_value


class DataQualityError(PoetryLLMError):
    """Raised when data quality is insufficient for processing."""
    pass


class ResourceConstraintError(PoetryLLMError):
    """Raised when system resources are insufficient."""
    pass


class GenerationFailureError(PoetryLLMError):
    """Raised when poetry generation fails with recovery options."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 details: Optional[dict] = None, recovery_suggestions: Optional[List[str]] = None):
        super().__init__(message, error_code, details)
        self.recovery_suggestions = recovery_suggestions or []


class ErrorRecoveryManager:
    """Manages error recovery strategies and fallback options."""
    
    def __init__(self, logger):
        self.logger = logger
        self.recovery_strategies = {
            'generation_failure': self._handle_generation_failure,
            'data_quality': self._handle_data_quality_issues,
            'resource_constraint': self._handle_resource_constraints,
            'model_loading': self._handle_model_loading_failure,
            'validation_error': self._handle_validation_error
        }
    
    def recover_from_error(self, error: Exception, strategy: str, **kwargs) -> Any:
        """
        Attempt to recover from an error using the specified strategy.
        
        Args:
            error: The exception that occurred
            strategy: Recovery strategy to use
            **kwargs: Additional parameters for recovery
            
        Returns:
            Recovery result or raises if recovery fails
        """
        if strategy not in self.recovery_strategies:
            self.logger.error(f"Unknown recovery strategy: {strategy}")
            raise error
        
        try:
            self.logger.info(f"Attempting recovery using strategy: {strategy}")
            return self.recovery_strategies[strategy](error, **kwargs)
        except Exception as recovery_error:
            self.logger.error(f"Recovery failed: {recovery_error}")
            raise error
    
    def _handle_generation_failure(self, error: Exception, **kwargs) -> dict:
        """Handle poetry generation failures with fallback options."""
        fallback_options = []
        
        # Suggest parameter adjustments
        if "temperature" in str(error).lower():
            fallback_options.append("Try reducing temperature (0.3-0.7)")
        if "length" in str(error).lower():
            fallback_options.append("Try reducing max_length parameter")
        if "memory" in str(error).lower() or "cuda" in str(error).lower():
            fallback_options.append("Try using CPU instead of GPU")
            fallback_options.append("Try a smaller model")
        
        # Generic fallbacks
        fallback_options.extend([
            "Simplify the prompt",
            "Try a different poet style",
            "Use greedy decoding (no_sample=True)"
        ])
        
        return {
            'success': False,
            'error_message': str(error),
            'recovery_suggestions': fallback_options,
            'fallback_available': True
        }
    
    def _handle_data_quality_issues(self, error: Exception, **kwargs) -> dict:
        """Handle data quality issues with specific feedback."""
        corpus_data = kwargs.get('corpus_data', {})
        quality_metrics = kwargs.get('quality_metrics', {})
        
        feedback = []
        
        if quality_metrics.get('empty_poems', 0) > 0:
            feedback.append(f"Remove {quality_metrics['empty_poems']} empty poems")
        
        if quality_metrics.get('total_poems', 0) < 10:
            feedback.append("Corpus too small - need at least 10 poems for reliable training")
        
        if quality_metrics.get('avg_lines_per_poem', 0) < 4:
            feedback.append("Poems too short - average should be at least 4 lines")
        
        if quality_metrics.get('short_poems', 0) > quality_metrics.get('total_poems', 1) * 0.5:
            feedback.append("Too many short poems - consider filtering or combining")
        
        # Generic feedback
        if not feedback:
            feedback.extend([
                "Check for proper poem formatting",
                "Ensure poems are separated by blank lines",
                "Verify text encoding (UTF-8 recommended)",
                "Remove non-poetry content (headers, footers, etc.)"
            ])
        
        return {
            'success': False,
            'error_message': str(error),
            'quality_feedback': feedback,
            'metrics': quality_metrics,
            'corpus_rejected': True
        }
    
    def _handle_resource_constraints(self, error: Exception, **kwargs) -> dict:
        """Handle resource constraint issues with alternative options."""
        resource_options = []
        
        if "memory" in str(error).lower():
            resource_options.extend([
                "Use a smaller model (e.g., gpt2 instead of gpt2-large)",
                "Reduce batch size for training",
                "Enable gradient checkpointing",
                "Use CPU instead of GPU if memory limited"
            ])
        
        if "cuda" in str(error).lower():
            resource_options.extend([
                "Fall back to CPU processing",
                "Check CUDA installation and drivers",
                "Reduce model precision (float16)"
            ])
        
        if "disk" in str(error).lower():
            resource_options.extend([
                "Clear temporary files",
                "Use streaming data loading",
                "Reduce dataset size"
            ])
        
        # Generic resource options
        resource_options.extend([
            "Close other applications to free memory",
            "Use cloud computing resources",
            "Process data in smaller batches"
        ])
        
        return {
            'success': False,
            'error_message': str(error),
            'resource_options': resource_options,
            'constraint_type': self._identify_constraint_type(error)
        }
    
    def _handle_model_loading_failure(self, error: Exception, **kwargs) -> dict:
        """Handle model loading failures."""
        model_name = kwargs.get('model_name', 'unknown')
        
        suggestions = [
            f"Check if model '{model_name}' exists and is accessible",
            "Verify internet connection for downloading models",
            "Try a different model name",
            "Check available disk space",
            "Clear model cache and retry"
        ]
        
        if "permission" in str(error).lower():
            suggestions.append("Check file permissions for model directory")
        
        if "network" in str(error).lower() or "connection" in str(error).lower():
            suggestions.extend([
                "Check internet connection",
                "Try downloading model manually",
                "Use a local model if available"
            ])
        
        return {
            'success': False,
            'error_message': str(error),
            'model_suggestions': suggestions,
            'alternative_models': ['gpt2', 'distilgpt2', 'gpt2-medium']
        }
    
    def _handle_validation_error(self, error: Exception, **kwargs) -> dict:
        """Handle validation errors with correction suggestions."""
        validation_fixes = []
        
        if "prompt" in str(error).lower():
            validation_fixes.extend([
                "Ensure prompt is not empty",
                "Check prompt length (should be reasonable)",
                "Remove special characters if causing issues"
            ])
        
        if "parameter" in str(error).lower():
            validation_fixes.extend([
                "Check parameter ranges (temperature: 0.0-2.0, top_p: 0.0-1.0)",
                "Ensure numeric parameters are valid numbers",
                "Use default parameters if unsure"
            ])
        
        return {
            'success': False,
            'error_message': str(error),
            'validation_fixes': validation_fixes
        }
    
    def _identify_constraint_type(self, error: Exception) -> str:
        """Identify the type of resource constraint."""
        error_str = str(error).lower()
        
        if "memory" in error_str or "oom" in error_str:
            return "memory"
        elif "cuda" in error_str or "gpu" in error_str:
            return "gpu"
        elif "disk" in error_str or "space" in error_str:
            return "disk"
        elif "cpu" in error_str:
            return "cpu"
        else:
            return "unknown"


class ErrorContext:
    """Context manager for handling errors with consistent logging and recovery."""
    
    def __init__(
        self,
        logger,
        context: str,
        reraise: bool = True,
        fallback_value: Any = None,
        error_type: type = PoetryLLMError,
        recovery_manager: Optional[ErrorRecoveryManager] = None,
        recovery_strategy: Optional[str] = None
    ):
        self.logger = logger
        self.context = context
        self.reraise = reraise
        self.fallback_value = fallback_value
        self.error_type = error_type
        self.recovery_manager = recovery_manager
        self.recovery_strategy = recovery_strategy
        self.recovery_result = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Attempt recovery if manager and strategy provided
            if self.recovery_manager and self.recovery_strategy:
                try:
                    self.recovery_result = self.recovery_manager.recover_from_error(
                        exc_val, self.recovery_strategy
                    )
                    self.logger.info(f"Error recovery attempted for {self.context}")
                except Exception:
                    pass  # Recovery failed, continue with normal error handling
            
            if not issubclass(exc_type, PoetryLLMError):
                # Wrap non-Poetry LLM errors
                wrapped_error = self.error_type(
                    f"{self.context}: {str(exc_val)}",
                    details={"original_error": str(exc_val), "error_type": exc_type.__name__}
                )
                handle_error(wrapped_error, self.logger, reraise=self.reraise)
            else:
                handle_error(exc_val, self.logger, self.context, reraise=self.reraise)
            
            # Suppress the exception if not reraising
            return not self.reraise
        
        return False


def create_user_friendly_error_message(error: Exception, context: str = "") -> str:
    """
    Create user-friendly error messages from technical exceptions.
    
    Args:
        error: The exception that occurred
        context: Additional context about the operation
        
    Returns:
        User-friendly error message
    """
    error_str = str(error).lower()
    
    # Common error patterns and their user-friendly messages
    if "file not found" in error_str or "no such file" in error_str:
        return f"Could not find the required file. Please check the file path and try again."
    
    elif "permission denied" in error_str:
        return f"Permission denied. Please check file permissions or run with appropriate privileges."
    
    elif "memory" in error_str or "out of memory" in error_str:
        return f"Not enough memory available. Try using a smaller model or closing other applications."
    
    elif "cuda" in error_str and "available" in error_str:
        return f"GPU not available. The system will fall back to CPU processing (this may be slower)."
    
    elif "network" in error_str or "connection" in error_str:
        return f"Network connection issue. Please check your internet connection and try again."
    
    elif "model" in error_str and ("not found" in error_str or "does not exist" in error_str):
        return f"The specified model could not be found. Please check the model name or try a different model."
    
    elif "timeout" in error_str:
        return f"Operation timed out. Please try again or use simpler parameters."
    
    elif "invalid" in error_str or "validation" in error_str:
        return f"Invalid input provided. Please check your parameters and try again."
    
    else:
        # Generic fallback
        base_msg = f"An error occurred"
        if context:
            base_msg += f" during {context}"
        base_msg += f": {str(error)}"
        return base_msg