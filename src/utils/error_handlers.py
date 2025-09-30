"""
Specialized error handlers for different components of the Poetry LLM system.

This module provides specific error handling implementations for data processing,
model operations, and generation tasks with recovery strategies and user-friendly feedback.
"""

import logging
import traceback
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import json

from .exceptions import (
    PoetryLLMError, DataQualityError, ResourceConstraintError, 
    GenerationFailureError, ErrorRecoveryManager, create_user_friendly_error_message
)


logger = logging.getLogger(__name__)


class DataQualityValidator:
    """Validates data quality and provides specific feedback for issues."""
    
    def __init__(self):
        self.recovery_manager = ErrorRecoveryManager(logger)
        self.quality_thresholds = {
            'min_poems': 5,
            'min_avg_lines': 3,
            'max_empty_ratio': 0.2,
            'max_short_poem_ratio': 0.6,
            'min_total_words': 100
        }
    
    def validate_corpus_quality(self, poems: List[Dict[str, Any]], 
                               poet_name: str = "Unknown") -> Dict[str, Any]:
        """
        Comprehensive validation of corpus quality with detailed feedback.
        
        Args:
            poems: List of poem dictionaries
            poet_name: Name of the poet for context
            
        Returns:
            Validation result with quality metrics and feedback
            
        Raises:
            DataQualityError: If corpus quality is insufficient
        """
        try:
            if not poems:
                raise DataQualityError(
                    f"No poems found in corpus for {poet_name}",
                    error_code="EMPTY_CORPUS",
                    details={"poet_name": poet_name, "poem_count": 0}
                )
            
            # Calculate quality metrics
            metrics = self._calculate_quality_metrics(poems)
            
            # Validate against thresholds
            validation_issues = self._check_quality_thresholds(metrics)
            
            # Generate feedback and warnings
            feedback = self._generate_quality_feedback(metrics, validation_issues)
            
            # Determine if corpus is acceptable
            is_valid = len(validation_issues) == 0
            severity_level = self._assess_severity(validation_issues)
            
            result = {
                'valid': is_valid,
                'severity': severity_level,
                'metrics': metrics,
                'issues': validation_issues,
                'feedback': feedback,
                'poet_name': poet_name,
                'recommendations': self._generate_recommendations(metrics, validation_issues)
            }
            
            # Log validation results
            if is_valid:
                logger.info(f"Corpus validation passed for {poet_name}: {metrics['total_poems']} poems")
            else:
                logger.warning(f"Corpus validation issues for {poet_name}: {len(validation_issues)} issues found")
            
            # Raise error if critical issues found
            if severity_level == 'critical':
                raise DataQualityError(
                    f"Critical data quality issues found in {poet_name} corpus",
                    error_code="CRITICAL_QUALITY_ISSUES",
                    details=result
                )
            
            return result
            
        except Exception as e:
            if isinstance(e, DataQualityError):
                raise
            
            # Handle unexpected validation errors
            logger.error(f"Unexpected error during corpus validation: {e}")
            raise DataQualityError(
                f"Corpus validation failed for {poet_name}: {str(e)}",
                error_code="VALIDATION_ERROR",
                details={"original_error": str(e), "poet_name": poet_name}
            )   
 
    def _calculate_quality_metrics(self, poems: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive quality metrics for the corpus."""
        metrics = {
            'total_poems': len(poems),
            'total_lines': 0,
            'total_words': 0,
            'empty_poems': 0,
            'short_poems': 0,
            'avg_lines_per_poem': 0,
            'avg_words_per_poem': 0,
            'vocabulary_size': 0
        }
        
        unique_words = set()
        
        for poem in poems:
            text = poem.get('text', '')
            
            if not text.strip():
                metrics['empty_poems'] += 1
                continue
            
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            words = text.split()
            
            metrics['total_lines'] += len(lines)
            metrics['total_words'] += len(words)
            
            if len(lines) < 4:
                metrics['short_poems'] += 1
            
            unique_words.update(word.lower().strip('.,!?;:"()[]') for word in words)
        
        # Calculate averages
        if metrics['total_poems'] > 0:
            metrics['avg_lines_per_poem'] = metrics['total_lines'] / metrics['total_poems']
            metrics['avg_words_per_poem'] = metrics['total_words'] / metrics['total_poems']
        
        metrics['vocabulary_size'] = len(unique_words)
        
        return metrics
    
    def _check_quality_thresholds(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check metrics against quality thresholds."""
        issues = []
        
        # Check minimum poem count
        if metrics['total_poems'] < self.quality_thresholds['min_poems']:
            issues.append({
                'type': 'insufficient_data',
                'severity': 'critical',
                'message': f"Only {metrics['total_poems']} poems found, need at least {self.quality_thresholds['min_poems']}",
                'metric': 'total_poems',
                'value': metrics['total_poems'],
                'threshold': self.quality_thresholds['min_poems']
            })
        
        # Check empty poem ratio
        empty_ratio = metrics['empty_poems'] / metrics['total_poems'] if metrics['total_poems'] > 0 else 0
        if empty_ratio > self.quality_thresholds['max_empty_ratio']:
            issues.append({
                'type': 'empty_content',
                'severity': 'critical',
                'message': f"{metrics['empty_poems']} empty poems ({empty_ratio:.1%}), maximum allowed is {self.quality_thresholds['max_empty_ratio']:.1%}",
                'metric': 'empty_poems',
                'value': empty_ratio,
                'threshold': self.quality_thresholds['max_empty_ratio']
            })
        
        return issues
    
    def _generate_quality_feedback(self, metrics: Dict[str, Any], 
                                 issues: List[Dict[str, Any]]) -> List[str]:
        """Generate specific feedback for quality issues."""
        feedback = []
        
        for issue in issues:
            if issue['type'] == 'insufficient_data':
                feedback.append(f"Add more poems to the corpus (currently {issue['value']}, need {issue['threshold']})")
            elif issue['type'] == 'empty_content':
                feedback.append(f"Remove {metrics['empty_poems']} empty poems from the corpus")
        
        return feedback
    
    def _generate_recommendations(self, metrics: Dict[str, Any], 
                                issues: List[Dict[str, Any]]) -> List[str]:
        """Generate actionable recommendations for improving corpus quality."""
        recommendations = []
        
        # Critical issues first
        critical_issues = [issue for issue in issues if issue['severity'] == 'critical']
        if critical_issues:
            recommendations.append("CRITICAL: Address the following issues before proceeding:")
            for issue in critical_issues:
                recommendations.append(f"  - {issue['message']}")
        
        # General recommendations
        recommendations.extend([
            "Ensure poems are properly formatted with clear line breaks",
            "Use consistent encoding (UTF-8 recommended)",
            "Separate poems with blank lines"
        ])
        
        return recommendations
    
    def _assess_severity(self, issues: List[Dict[str, Any]]) -> str:
        """Assess overall severity of quality issues."""
        if not issues:
            return 'none'
        
        severities = [issue['severity'] for issue in issues]
        
        if 'critical' in severities:
            return 'critical'
        elif 'warning' in severities:
            return 'warning'
        else:
            return 'minor'


class GenerationErrorHandler:
    """Handles poetry generation errors with recovery strategies."""
    
    def __init__(self):
        self.recovery_manager = ErrorRecoveryManager(logger)
        self.fallback_configs = {
            'conservative': {
                'temperature': 0.3,
                'top_p': 0.8,
                'top_k': 20,
                'max_length': 100,
                'do_sample': True
            },
            'safe': {
                'temperature': 0.1,
                'top_p': 0.9,
                'top_k': 10,
                'max_length': 50,
                'do_sample': False
            }
        }
    
    def handle_generation_failure(self, error: Exception, request_params: Dict[str, Any], 
                                attempt_count: int = 1) -> Dict[str, Any]:
        """
        Handle poetry generation failures with specific recovery strategies.
        
        Args:
            error: The exception that occurred
            request_params: Original generation parameters
            attempt_count: Current attempt number
            
        Returns:
            Recovery result with suggestions and fallback options
        """
        try:
            error_type = self._classify_generation_error(error)
            recovery_result = self._apply_recovery_strategy(error_type, error, request_params, attempt_count)
            
            # Log recovery attempt
            logger.info(f"Generation error recovery attempted: {error_type} (attempt {attempt_count})")
            
            return recovery_result
            
        except Exception as recovery_error:
            logger.error(f"Error recovery failed: {recovery_error}")
            return {
                'success': False,
                'error_message': str(error),
                'recovery_failed': True,
                'fallback_suggestions': [
                    "Try with default parameters",
                    "Use a different model",
                    "Simplify the prompt"
                ]
            }
    
    def _classify_generation_error(self, error: Exception) -> str:
        """Classify the type of generation error."""
        error_str = str(error).lower()
        
        if "memory" in error_str or "oom" in error_str:
            return "memory_error"
        elif "cuda" in error_str or "gpu" in error_str:
            return "gpu_error"
        elif "timeout" in error_str:
            return "timeout_error"
        elif "invalid" in error_str or "parameter" in error_str:
            return "parameter_error"
        else:
            return "unknown_error"
    
    def _apply_recovery_strategy(self, error_type: str, error: Exception, 
                               request_params: Dict[str, Any], attempt_count: int) -> Dict[str, Any]:
        """Apply specific recovery strategy based on error type."""
        
        if error_type == "memory_error":
            return self._handle_memory_error(error, request_params, attempt_count)
        elif error_type == "parameter_error":
            return self._handle_parameter_error(error, request_params, attempt_count)
        else:
            return self._handle_unknown_error(error, request_params, attempt_count)
    
    def _handle_memory_error(self, error: Exception, request_params: Dict[str, Any], 
                           attempt_count: int) -> Dict[str, Any]:
        """Handle memory-related errors."""
        fallback_params = request_params.copy()
        
        # Reduce memory usage
        if attempt_count == 1:
            fallback_params['max_length'] = min(fallback_params.get('max_length', 200), 100)
        elif attempt_count == 2:
            fallback_params.update(self.fallback_configs['safe'])
        
        return {
            'success': False,
            'error_type': 'memory_error',
            'error_message': create_user_friendly_error_message(error),
            'fallback_params': fallback_params,
            'suggestions': [
                "Reduce max_length parameter",
                "Use CPU instead of GPU",
                "Try a smaller model"
            ],
            'retry_recommended': attempt_count < 3
        }
    
    def _handle_parameter_error(self, error: Exception, request_params: Dict[str, Any], 
                              attempt_count: int) -> Dict[str, Any]:
        """Handle parameter validation errors."""
        fallback_params = self.fallback_configs['conservative'].copy()
        
        return {
            'success': False,
            'error_type': 'parameter_error',
            'error_message': create_user_friendly_error_message(error),
            'fallback_params': fallback_params,
            'suggestions': [
                "Check parameter ranges (temperature: 0.0-2.0, top_p: 0.0-1.0)",
                "Use default parameters if unsure"
            ],
            'retry_recommended': True
        }
    
    def _handle_unknown_error(self, error: Exception, request_params: Dict[str, Any], 
                            attempt_count: int) -> Dict[str, Any]:
        """Handle unknown errors with generic recovery."""
        fallback_params = self.fallback_configs['safe'].copy()
        
        return {
            'success': False,
            'error_type': 'unknown_error',
            'error_message': create_user_friendly_error_message(error),
            'fallback_params': fallback_params,
            'suggestions': [
                "Try with safe default parameters",
                "Simplify the prompt",
                "Check system resources"
            ],
            'retry_recommended': attempt_count < 2
        }


class ModelLoadingErrorHandler:
    """Specialized handler for model loading errors."""
    
    def __init__(self):
        self.logger = logger
        self.fallback_models = ['gpt2', 'distilgpt2', 'gpt2-medium']
        self.device_fallbacks = ['cuda', 'cpu']
    
    def handle_model_loading_error(self, error: Exception, model_name: str, 
                                 device: str = 'auto') -> Dict[str, Any]:
        """
        Handle model loading errors with fallback strategies.
        
        Args:
            error: The loading error that occurred
            model_name: Name of the model that failed to load
            device: Target device for model loading
            
        Returns:
            Recovery strategy result
        """
        error_type = self._classify_loading_error(error)
        
        recovery_strategy = {
            'original_model': model_name,
            'original_device': device,
            'error_type': error_type,
            'error_message': create_user_friendly_error_message(error),
            'fallback_options': []
        }
        
        if error_type == 'memory_error':
            recovery_strategy['fallback_options'].extend([
                {'action': 'device_fallback', 'device': 'cpu', 'reason': 'Insufficient GPU memory'},
                {'action': 'model_fallback', 'models': self.fallback_models, 'reason': 'Try smaller models'}
            ])
        
        elif error_type == 'network_error':
            recovery_strategy['fallback_options'].extend([
                {'action': 'retry', 'delay': 5, 'reason': 'Temporary network issue'},
                {'action': 'local_model', 'reason': 'Use locally cached model if available'}
            ])
        
        else:
            recovery_strategy['fallback_options'].extend([
                {'action': 'model_fallback', 'models': self.fallback_models, 'reason': 'Try alternative models'},
                {'action': 'device_fallback', 'device': 'cpu', 'reason': 'Fallback to CPU'}
            ])
        
        return recovery_strategy
    
    def _classify_loading_error(self, error: Exception) -> str:
        """Classify the type of model loading error."""
        error_str = str(error).lower()
        
        if "memory" in error_str or "oom" in error_str:
            return "memory_error"
        elif "network" in error_str or "connection" in error_str or "download" in error_str:
            return "network_error"
        elif "permission" in error_str or "access" in error_str:
            return "permission_error"
        elif "not found" in error_str or "does not exist" in error_str:
            return "model_not_found"
        else:
            return "unknown_error"


class ComprehensiveErrorLogger:
    """Enhanced error logging with context and recovery tracking."""
    
    def __init__(self, logger_name: str = "error_handler"):
        self.logger = logging.getLogger(logger_name)
        self.error_counts = {}
        self.recovery_stats = {}
    
    def log_error_with_context(self, error: Exception, context: Dict[str, Any], 
                             recovery_attempted: bool = False, recovery_successful: bool = False):
        """
        Log error with comprehensive context and recovery information.
        
        Args:
            error: The exception that occurred
            context: Additional context information
            recovery_attempted: Whether recovery was attempted
            recovery_successful: Whether recovery was successful
        """
        error_type = type(error).__name__
        operation = context.get('operation', 'unknown')
        
        # Update error counts
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
        self.error_counts[error_type] += 1
        
        # Update recovery stats
        if recovery_attempted:
            if error_type not in self.recovery_stats:
                self.recovery_stats[error_type] = {'attempted': 0, 'successful': 0}
            self.recovery_stats[error_type]['attempted'] += 1
            if recovery_successful:
                self.recovery_stats[error_type]['successful'] += 1
        
        # Log at appropriate level
        if recovery_successful:
            self.logger.info(f"Error recovered: {error_type} in {operation}")
        elif recovery_attempted:
            self.logger.warning(f"Recovery failed: {error_type} in {operation}")
        else:
            self.logger.error(f"Unhandled error: {error_type} in {operation}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        total_errors = sum(self.error_counts.values())
        total_recoveries = sum(stats['successful'] for stats in self.recovery_stats.values())
        
        return {
            'total_errors': total_errors,
            'total_recoveries': total_recoveries,
            'recovery_rate': (total_recoveries / max(total_errors, 1)) * 100,
            'error_counts': self.error_counts.copy(),
            'recovery_stats': self.recovery_stats.copy()
        }


class ErrorHandlingCoordinator:
    """Coordinates error handling across different system components."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_validator = DataQualityValidator()
        self.generation_handler = GenerationErrorHandler()
        self.model_handler = ModelLoadingErrorHandler()
        self.error_logger = ComprehensiveErrorLogger()
    
    def handle_component_error(self, component: str, error: Exception, 
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle errors from specific system components.
        
        Args:
            component: Name of the component where error occurred
            error: The exception that occurred
            context: Context information about the operation
            
        Returns:
            Error handling result with recovery options
        """
        try:
            if component == 'data_processing':
                return self._handle_data_error(error, context)
            elif component == 'generation':
                return self._handle_generation_error(error, context)
            else:
                return self._handle_generic_error(error, context, component)
                
        except Exception as handling_error:
            self.logger.error(f"Error handling failed for {component}: {handling_error}")
            return {
                'success': False,
                'error_message': str(error),
                'handling_failed': True,
                'component': component
            }
    
    def _handle_data_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data processing errors."""
        return {
            'success': False,
            'error_type': 'data_processing',
            'error_message': create_user_friendly_error_message(error),
            'recovery_suggestions': [
                "Check file paths and permissions",
                "Verify data format",
                "Ensure sufficient disk space"
            ]
        }
    
    def _handle_generation_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle generation errors."""
        request_params = context.get('request_params', {})
        attempt_count = context.get('attempt_count', 1)
        
        return self.generation_handler.handle_generation_failure(error, request_params, attempt_count)
    
    def _handle_generic_error(self, error: Exception, context: Dict[str, Any], 
                            component: str) -> Dict[str, Any]:
        """Handle generic errors from unknown components."""
        return {
            'success': False,
            'error_type': 'generic_error',
            'component': component,
            'error_message': create_user_friendly_error_message(error),
            'recovery_suggestions': [
                "Check system logs for details",
                "Verify system resources",
                "Restart the application if issues persist"
            ]
        }