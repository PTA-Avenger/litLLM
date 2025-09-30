"""
CLI-specific error handlers for the Poetry LLM system.

This module provides specialized error handling for CLI operations,
including user-friendly error messages and recovery suggestions.
"""

import logging
import click
import sys
from typing import Dict, Any, Optional, List
from pathlib import Path

from .exceptions import create_user_friendly_error_message
from .error_integration import get_system_error_handler, safe_execute

logger = logging.getLogger(__name__)


class CLIErrorHandler:
    """Handles CLI-specific errors with user-friendly messages and recovery."""
    
    def __init__(self, enable_recovery: bool = True):
        self.enable_recovery = enable_recovery
        self.system_handler = get_system_error_handler() if enable_recovery else None
    
    def handle_generation_error(self, error: Exception, context: Dict[str, Any], 
                              attempt: int = 1) -> Dict[str, Any]:
        """
        Handle poetry generation errors in CLI context.
        
        Args:
            error: The exception that occurred
            context: Context information about the generation
            attempt: Current attempt number
            
        Returns:
            Recovery result with CLI-specific guidance
        """
        if not self.enable_recovery:
            return {
                'success': False,
                'error_message': create_user_friendly_error_message(error),
                'should_exit': True
            }
        
        # Use system error handler for recovery
        recovery_result = self.system_handler.handle_generation_error(error, context)
        
        # Add CLI-specific guidance
        cli_result = recovery_result.copy()
        cli_result['cli_suggestions'] = self._get_cli_suggestions(error, recovery_result)
        cli_result['should_retry'] = recovery_result.get('retry_recommended', False) and attempt < 3
        cli_result['should_exit'] = not cli_result['should_retry']
        
        return cli_result
    
    def handle_model_loading_error(self, error: Exception, model_name: str) -> Dict[str, Any]:
        """Handle model loading errors with CLI-specific guidance."""
        if not self.enable_recovery:
            return {
                'success': False,
                'error_message': create_user_friendly_error_message(error),
                'should_exit': True
            }
        
        context = {'model_name': model_name}
        recovery_result = self.system_handler.handle_model_error(error, context)
        
        cli_result = recovery_result.copy()
        cli_result['cli_suggestions'] = self._get_model_cli_suggestions(error, model_name)
        cli_result['should_exit'] = True  # Model loading failures are typically fatal
        
        return cli_result
    
    def handle_file_error(self, error: Exception, file_path: str, operation: str) -> Dict[str, Any]:
        """Handle file operation errors."""
        error_msg = create_user_friendly_error_message(error, f"{operation} {file_path}")
        
        suggestions = []
        error_str = str(error).lower()
        
        if "permission" in error_str:
            suggestions.extend([
                f"Check file permissions for {file_path}",
                "Run with appropriate privileges if needed",
                "Ensure the file is not open in another application"
            ])
        elif "not found" in error_str:
            suggestions.extend([
                f"Verify that {file_path} exists",
                "Check the file path for typos",
                "Use absolute path if relative path fails"
            ])
        elif "space" in error_str:
            suggestions.extend([
                "Free up disk space",
                "Choose a different output location",
                "Clean up temporary files"
            ])
        else:
            suggestions.extend([
                f"Verify {file_path} is accessible",
                "Check file format and encoding",
                "Try a different file location"
            ])
        
        return {
            'success': False,
            'error_message': error_msg,
            'cli_suggestions': suggestions,
            'should_exit': False  # File errors are often recoverable
        }
    
    def _get_cli_suggestions(self, error: Exception, recovery_result: Dict[str, Any]) -> List[str]:
        """Get CLI-specific suggestions for generation errors."""
        suggestions = []
        
        # Add CLI command suggestions based on error type
        error_type = recovery_result.get('error_type', 'unknown')
        
        if error_type == 'memory_error':
            suggestions.extend([
                "Try: --max-length 50 (reduce output length)",
                "Try: --model gpt2 (use smaller model)",
                "Try: --no-sample (use greedy decoding)"
            ])
        
        elif error_type == 'parameter_error':
            suggestions.extend([
                "Try: --temperature 0.8 (standard temperature)",
                "Try: --top-p 0.9 (standard top-p)",
                "Remove custom parameters to use defaults"
            ])
        
        elif error_type == 'timeout_error':
            suggestions.extend([
                "Try: --max-length 100 (shorter output)",
                "Try: --no-sample (faster generation)",
                "Use a simpler prompt"
            ])
        
        # Add general CLI suggestions
        suggestions.extend([
            "Use --verbose for more detailed error information",
            "Use --no-recovery to see raw error messages",
            "Check 'poetry-cli status' for system information"
        ])
        
        return suggestions
    
    def _get_model_cli_suggestions(self, error: Exception, model_name: str) -> List[str]:
        """Get CLI-specific suggestions for model loading errors."""
        suggestions = []
        error_str = str(error).lower()
        
        if "not found" in error_str:
            suggestions.extend([
                f"Try: --model gpt2 (instead of {model_name})",
                "Check available models with --help",
                "Verify internet connection for model download"
            ])
        
        elif "memory" in error_str:
            suggestions.extend([
                "Try: --model gpt2 (smaller model)",
                "Close other applications to free memory",
                "Use CPU instead of GPU if available"
            ])
        
        elif "network" in error_str:
            suggestions.extend([
                "Check internet connection",
                "Try again in a few minutes",
                "Use a locally available model"
            ])
        
        return suggestions


def display_error_with_recovery(error_result: Dict[str, Any], operation: str):
    """Display error information with recovery suggestions in CLI format."""
    
    # Display main error message
    click.echo(f"❌ {operation.title()} failed: {error_result['error_message']}", err=True)
    
    # Display CLI-specific suggestions
    if 'cli_suggestions' in error_result and error_result['cli_suggestions']:
        click.echo("\n💡 Suggestions:", err=True)
        for suggestion in error_result['cli_suggestions'][:5]:  # Show top 5
            click.echo(f"   {suggestion}", err=True)
    
    # Display recovery suggestions if available
    if 'suggestions' in error_result and error_result['suggestions']:
        click.echo("\n🔧 Recovery options:", err=True)
        for suggestion in error_result['suggestions'][:3]:  # Show top 3
            click.echo(f"   • {suggestion}", err=True)
    
    # Display fallback parameters if available
    if 'fallback_params' in error_result:
        click.echo("\n⚙️  Suggested parameters:", err=True)
        for key, value in error_result['fallback_params'].items():
            click.echo(f"   --{key.replace('_', '-')} {value}", err=True)


def safe_cli_execute(operation: str, func, *args, **kwargs):
    """
    Execute a function with CLI-appropriate error handling.
    
    Args:
        operation: Description of the operation
        func: Function to execute
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Function result or exits on error
    """
    try:
        return func(*args, **kwargs)
    except KeyboardInterrupt:
        click.echo(f"\n{operation.title()} interrupted by user", err=True)
        sys.exit(1)
    except Exception as e:
        error_msg = create_user_friendly_error_message(e, operation)
        click.echo(f"❌ {operation.title()} failed: {error_msg}", err=True)
        sys.exit(1)


def validate_cli_parameters(**params) -> Dict[str, Any]:
    """
    Validate CLI parameters and return validation results.
    
    Args:
        **params: Parameters to validate
        
    Returns:
        Validation result dictionary
    """
    issues = []
    
    # Validate temperature
    if 'temperature' in params:
        temp = params['temperature']
        if temp is not None and (temp < 0.0 or temp > 2.0):
            issues.append("Temperature must be between 0.0 and 2.0")
    
    # Validate top_p
    if 'top_p' in params:
        top_p = params['top_p']
        if top_p is not None and (top_p < 0.0 or top_p > 1.0):
            issues.append("top_p must be between 0.0 and 1.0")
    
    # Validate max_length
    if 'max_length' in params:
        max_len = params['max_length']
        if max_len is not None and (max_len < 10 or max_len > 1000):
            issues.append("max_length must be between 10 and 1000")
    
    # Validate prompt
    if 'prompt' in params:
        prompt = params['prompt']
        if not prompt or not prompt.strip():
            issues.append("Prompt cannot be empty")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues
    }


def show_progress_with_recovery(operation: str, enable_recovery: bool = True):
    """Show progress indicator with recovery information."""
    recovery_status = "with recovery" if enable_recovery else "fail-fast mode"
    click.echo(f"🔄 {operation.title()} ({recovery_status})...")


def show_success_with_stats(operation: str, stats: Optional[Dict[str, Any]] = None):
    """Show success message with optional statistics."""
    click.echo(f"✅ {operation.title()} completed successfully!")
    
    if stats and stats.get('total_errors', 0) > 0:
        recovery_rate = stats.get('recovery_rate', 0)
        click.echo(f"   Handled {stats['total_errors']} errors with {recovery_rate:.1f}% recovery rate")