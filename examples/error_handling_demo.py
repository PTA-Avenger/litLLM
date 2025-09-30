#!/usr/bin/env python3
"""
Demonstration of comprehensive error handling system (Task 8.1).

This script demonstrates the robust error handling capabilities including:
- Data quality issue handling with fallback options
- Generation failure recovery with user-friendly messages
- Edge case error handling and logging
"""

import sys
from pathlib import Path
import tempfile
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.exceptions import (
    PoetryLLMError, DataQualityError, GenerationFailureError, 
    create_user_friendly_error_message
)
from utils.error_integration import SystemErrorHandler, handle_with_recovery, safe_execute
from utils.error_handlers import GenerationErrorHandler, DataQualityValidator
from stylometric.training_data import PoetryCorpusLoader


def demo_user_friendly_error_messages():
    """Demonstrate user-friendly error message generation."""
    print("=" * 60)
    print("DEMO 1: User-Friendly Error Messages")
    print("=" * 60)
    
    # Test various error types
    test_errors = [
        (FileNotFoundError("No such file or directory: '/path/to/missing.txt'"), "file operation"),
        (PermissionError("Permission denied: '/restricted/file.txt'"), "file access"),
        (ValueError("Invalid temperature: 3.5. Must be between 0.0 and 2.0"), "parameter validation"),
        (ConnectionError("Network connection failed"), "model download"),
        (Exception("CUDA out of memory: tried to allocate 10.00 GiB"), "model loading")
    ]
    
    for error, context in test_errors:
        technical_message = str(error)
        user_friendly = create_user_friendly_error_message(error, context)
        
        print(f"\nContext: {context}")
        print(f"Technical: {technical_message}")
        print(f"User-friendly: {user_friendly}")


def demo_generation_error_recovery():
    """Demonstrate generation error recovery mechanisms."""
    print("\n" + "=" * 60)
    print("DEMO 2: Generation Error Recovery")
    print("=" * 60)
    
    handler = GenerationErrorHandler()
    
    # Test different types of generation errors
    test_scenarios = [
        {
            'error': ValueError("Invalid temperature: 3.5"),
            'params': {'temperature': 3.5, 'max_length': 200},
            'description': "Invalid parameter error"
        },
        {
            'error': Exception("CUDA out of memory"),
            'params': {'max_length': 1000, 'device': 'cuda'},
            'description': "Memory constraint error"
        },
        {
            'error': Exception("Generation timeout after 60 seconds"),
            'params': {'temperature': 0.8, 'max_length': 500},
            'description': "Timeout error"
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n--- {scenario['description']} ---")
        print(f"Original error: {scenario['error']}")
        print(f"Original params: {scenario['params']}")
        
        result = handler.handle_generation_failure(
            scenario['error'], 
            scenario['params'], 
            attempt_count=1
        )
        
        print(f"Error type: {result['error_type']}")
        print(f"User message: {result['error_message']}")
        print(f"Retry recommended: {result.get('retry_recommended', False)}")
        
        if 'fallback_params' in result:
            print(f"Fallback params: {result['fallback_params']}")
        
        if 'suggestions' in result:
            print("Recovery suggestions:")
            for suggestion in result['suggestions'][:3]:  # Show top 3
                print(f"  • {suggestion}")


def demo_data_quality_handling():
    """Demonstrate data quality error handling."""
    print("\n" + "=" * 60)
    print("DEMO 3: Data Quality Error Handling")
    print("=" * 60)
    
    validator = DataQualityValidator()
    
    # Create test corpus with quality issues
    problematic_poems = [
        {"text": "", "title": "Empty Poem", "poet": "TestPoet"},  # Empty
        {"text": "Short", "title": "Too Short", "poet": "TestPoet"},  # Too short
        {"text": "A proper poem\nWith multiple lines\nAnd good structure\nFor training", "title": "Good Poem", "poet": "TestPoet"},
        {"text": "Another good poem\nWith proper length\nAnd structure", "title": "Another Good", "poet": "TestPoet"},
    ]
    
    print(f"Testing corpus with {len(problematic_poems)} poems...")
    print("Issues: 1 empty poem, 1 very short poem, 2 good poems")
    
    try:
        result = validator.validate_corpus_quality(problematic_poems, "TestPoet")
        print(f"\nValidation result: {'PASSED' if result['valid'] else 'FAILED'}")
        print(f"Severity: {result['severity']}")
        
        if result['feedback']:
            print("\nFeedback:")
            for feedback in result['feedback']:
                print(f"  • {feedback}")
        
        if result['recommendations']:
            print("\nRecommendations:")
            for rec in result['recommendations'][:3]:  # Show top 3
                print(f"  • {rec}")
                
    except DataQualityError as e:
        print(f"\nData quality error caught: {e}")
        print(f"Error code: {e.error_code}")
        
        if e.details and 'feedback' in e.details:
            print("\nSpecific feedback:")
            for feedback in e.details['feedback']:
                print(f"  • {feedback}")


def demo_file_handling_errors():
    """Demonstrate file handling error recovery."""
    print("\n" + "=" * 60)
    print("DEMO 4: File Handling Error Recovery")
    print("=" * 60)
    
    loader = PoetryCorpusLoader()
    
    # Test 1: Non-existent file
    print("--- Test 1: Non-existent file ---")
    try:
        poems = loader.load_corpus_from_file("/nonexistent/file.txt", "TestPoet")
    except DataQualityError as e:
        print(f"Error handled gracefully: {e}")
        print(f"Error code: {e.error_code}")
    
    # Test 2: Empty file
    print("\n--- Test 2: Empty file ---")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("")  # Empty file
        temp_path = f.name
    
    try:
        poems = loader.load_corpus_from_file(temp_path, "TestPoet")
    except DataQualityError as e:
        print(f"Empty file error handled: {e}")
        print(f"Error code: {e.error_code}")
    finally:
        Path(temp_path).unlink()
    
    # Test 3: Malformed content
    print("\n--- Test 3: Malformed content ---")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("{'invalid': json content without proper structure")
        temp_path = f.name
    
    try:
        poems = loader.load_corpus_from_file(temp_path, "TestPoet", format_type="json")
    except DataQualityError as e:
        print(f"Malformed content error handled: {e}")
        print(f"Error code: {e.error_code}")
    finally:
        Path(temp_path).unlink()


def demo_system_integration():
    """Demonstrate integrated system error handling."""
    print("\n" + "=" * 60)
    print("DEMO 5: System Integration")
    print("=" * 60)
    
    # Initialize system error handler
    system_handler = SystemErrorHandler(log_errors=True, enable_recovery=True)
    
    # Test integrated error handling
    test_operations = [
        {
            'operation': 'data_processing',
            'error': FileNotFoundError("Corpus file not found"),
            'context': {'operation': 'corpus_loading', 'poet_name': 'TestPoet'}
        },
        {
            'operation': 'generation',
            'error': ValueError("Invalid temperature: 3.5"),
            'context': {'operation': 'poetry_generation', 'request_params': {'temperature': 3.5}}
        },
        {
            'operation': 'model_operations',
            'error': Exception("Model 'nonexistent-model' not found"),
            'context': {'operation': 'model_loading', 'model_name': 'nonexistent-model'}
        }
    ]
    
    for test in test_operations:
        print(f"\n--- {test['operation'].title()} Error ---")
        
        if test['operation'] == 'data_processing':
            result = system_handler.handle_data_processing_error(test['error'], test['context'])
        elif test['operation'] == 'generation':
            result = system_handler.handle_generation_error(test['error'], test['context'])
        elif test['operation'] == 'model_operations':
            result = system_handler.handle_model_error(test['error'], test['context'])
        
        print(f"Success: {result['success']}")
        print(f"Error type: {result.get('error_type', 'unknown')}")
        print(f"User message: {result.get('error_message', 'No message')}")
        
        suggestions_key = 'recovery_suggestions' if 'recovery_suggestions' in result else 'suggestions'
        if suggestions_key in result:
            print(f"Suggestions ({len(result[suggestions_key])}):")
            for suggestion in result[suggestions_key][:2]:  # Show top 2
                print(f"  • {suggestion}")
    
    # Show system statistics
    print("\n--- System Statistics ---")
    stats = system_handler.get_error_statistics()
    print(f"Total errors handled: {stats['total_errors']}")
    print(f"Recovery rate: {stats['recovery_rate']:.1f}%")
    print(f"Errors by component: {stats.get('by_component', {})}")


def demo_safe_execution():
    """Demonstrate safe execution utilities."""
    print("\n" + "=" * 60)
    print("DEMO 6: Safe Execution Utilities")
    print("=" * 60)
    
    # Test safe_execute with fallback
    print("--- Safe Execute with Fallback ---")
    
    def failing_operation():
        raise Exception("This operation always fails")
    
    result = safe_execute(
        "test operation", 
        failing_operation, 
        fallback_value="Safe fallback result"
    )
    print(f"Result: {result}")
    
    # Test handle_with_recovery
    print("\n--- Handle with Recovery ---")
    
    def operation_with_recovery():
        raise ValueError("Recoverable error")
    
    try:
        result = handle_with_recovery("recoverable operation", operation_with_recovery)
    except Exception as e:
        print(f"Error handled through recovery system: {type(e).__name__}")
        print(f"Error message: {str(e)}")


def main():
    """Run all error handling demonstrations."""
    print("🔧 COMPREHENSIVE ERROR HANDLING DEMONSTRATION")
    print("Task 8.1: Implement robust error handling")
    print()
    
    # Set up logging to show error handling in action
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        demo_user_friendly_error_messages()
        demo_generation_error_recovery()
        demo_data_quality_handling()
        demo_file_handling_errors()
        demo_system_integration()
        demo_safe_execution()
        
        print("\n" + "=" * 60)
        print("✅ ALL ERROR HANDLING DEMONSTRATIONS COMPLETED")
        print("=" * 60)
        print()
        print("Key Features Demonstrated:")
        print("• User-friendly error message generation")
        print("• Generation failure recovery with fallback options")
        print("• Data quality validation with specific feedback")
        print("• File handling error recovery")
        print("• Integrated system error handling")
        print("• Safe execution utilities")
        print("• Comprehensive logging and monitoring")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("This demonstrates that even the demo has error handling!")


if __name__ == "__main__":
    main()