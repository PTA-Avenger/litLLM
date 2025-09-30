"""
Example script demonstrating the fine-tuning capability.

This script shows how to use the fine-tuning system to train a model
on poetry data, including data preparation, training, and evaluation.
"""

import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.stylometric.training_data import TrainingExample
from src.stylometric.fine_tuning import (
    FineTuningConfig,
    PoetryFineTuner,
    create_fine_tuning_config
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_training_data():
    """Create sample training data for demonstration."""
    training_examples = [
        TrainingExample(
            instruction="Write a poem in the style of Emily Dickinson.",
            input_text="",
            output_text="""I'm Nobody! Who are you?
Are you - Nobody - Too?
Then there's a pair of us!
Don't tell! they'd advertise - you know!""",
            stylometric_features={
                "total_lines": 4,
                "syllable_count": 32,
                "rhyme_scheme": "ABAB",
                "meter": "common_meter"
            },
            poet_name="Emily Dickinson",
            poem_title="I'm Nobody! Who are you?"
        ),
        TrainingExample(
            instruction="Continue this poem in the style of Emily Dickinson:",
            input_text="Because I could not stop for Death -",
            output_text="He kindly stopped for me -",
            stylometric_features={
                "total_lines": 1,
                "syllable_count": 8,
                "meter": "common_meter"
            },
            poet_name="Emily Dickinson",
            poem_title="Because I could not stop for Death",
            line_number=2
        ),
        TrainingExample(
            instruction="Write a poem in the style of Emily Dickinson.",
            input_text="",
            output_text="""Hope is the thing with feathers -
That perches in the soul -
And sings the tune without the words -
And never stops - at all -""",
            stylometric_features={
                "total_lines": 4,
                "syllable_count": 28,
                "rhyme_scheme": "ABCB",
                "meter": "common_meter"
            },
            poet_name="Emily Dickinson",
            poem_title="Hope is the thing with feathers"
        ),
        TrainingExample(
            instruction="Write a short poem in the style of Emily Dickinson with dashes and slant rhyme.",
            input_text="",
            output_text="""I heard a Fly buzz - when I died -
The Stillness in the Room
Was like the Stillness in the Air -
Between the Heaves of Storm -""",
            stylometric_features={
                "total_lines": 4,
                "syllable_count": 30,
                "special_characters": {"dashes": 6},
                "rhyme_scheme": "ABCB"
            },
            poet_name="Emily Dickinson",
            poem_title="I heard a Fly buzz - when I died"
        )
    ]
    
    return training_examples


def demonstrate_config_creation():
    """Demonstrate configuration creation."""
    logger.info("Creating fine-tuning configuration...")
    
    # Create a basic configuration
    config = create_fine_tuning_config(
        base_model="gpt2",
        output_dir="./models/dickinson_fine_tuned",
        epochs=2,  # Small number for demo
        batch_size=2,
        learning_rate=5e-5,
        gradient_checkpointing=True,
        eval_steps=10,
        save_steps=10,
        logging_steps=5
    )
    
    logger.info(f"Configuration created:")
    logger.info(f"  Base model: {config.base_model_name}")
    logger.info(f"  Output directory: {config.output_dir}")
    logger.info(f"  Epochs: {config.num_train_epochs}")
    logger.info(f"  Batch size: {config.per_device_train_batch_size}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    
    return config


def demonstrate_data_preparation():
    """Demonstrate training data preparation."""
    logger.info("Preparing training data...")
    
    # Create sample data
    training_examples = create_sample_training_data()
    
    logger.info(f"Created {len(training_examples)} training examples:")
    for i, example in enumerate(training_examples):
        logger.info(f"  Example {i+1}: {example.instruction[:50]}...")
        logger.info(f"    Poet: {example.poet_name}")
        logger.info(f"    Output length: {len(example.output_text)} characters")
        logger.info(f"    Features: {list(example.stylometric_features.keys())}")
    
    # Split data for training and validation
    split_point = int(len(training_examples) * 0.8)
    train_examples = training_examples[:split_point]
    val_examples = training_examples[split_point:]
    
    logger.info(f"Data split: {len(train_examples)} training, {len(val_examples)} validation")
    
    return train_examples, val_examples


def demonstrate_fine_tuner_setup():
    """Demonstrate fine-tuner setup (without actual training)."""
    logger.info("Setting up fine-tuner...")
    
    # Create configuration
    config = demonstrate_config_creation()
    
    # Create fine-tuner
    fine_tuner = PoetryFineTuner(config)
    
    logger.info("Fine-tuner created successfully")
    logger.info(f"  Configuration: {type(config).__name__}")
    logger.info(f"  Model loaded: {fine_tuner.model is not None}")
    logger.info(f"  Tokenizer loaded: {fine_tuner.tokenizer is not None}")
    
    return fine_tuner


def demonstrate_training_workflow():
    """Demonstrate the complete training workflow (simulation)."""
    logger.info("Demonstrating complete training workflow...")
    
    try:
        # Prepare data
        train_examples, val_examples = demonstrate_data_preparation()
        
        # Setup fine-tuner
        fine_tuner = demonstrate_fine_tuner_setup()
        
        # Simulate training process
        logger.info("Training process would include:")
        logger.info("  1. Loading base model and tokenizer")
        logger.info("  2. Preparing datasets with tokenization")
        logger.info("  3. Setting up HuggingFace Trainer with callbacks")
        logger.info("  4. Running training loop with progress monitoring")
        logger.info("  5. Saving checkpoints and final model")
        logger.info("  6. Evaluating on validation data")
        
        # Simulate training results
        simulated_results = {
            'success': True,
            'model_path': './models/dickinson_fine_tuned',
            'metrics': {
                'train_loss': 2.1,
                'eval_loss': 2.3,
                'train_runtime': 300.0,
                'train_steps': 50,
                'epoch': 2.0
            },
            'config': fine_tuner.config.__dict__,
            'corpus_info': {
                'poet_name': 'Emily Dickinson',
                'total_examples': len(train_examples) + len(val_examples),
                'train_examples': len(train_examples),
                'val_examples': len(val_examples)
            }
        }
        
        logger.info("Simulated training results:")
        logger.info(f"  Success: {simulated_results['success']}")
        logger.info(f"  Final train loss: {simulated_results['metrics']['train_loss']:.3f}")
        logger.info(f"  Final eval loss: {simulated_results['metrics']['eval_loss']:.3f}")
        logger.info(f"  Training steps: {simulated_results['metrics']['train_steps']}")
        logger.info(f"  Model saved to: {simulated_results['model_path']}")
        
        return simulated_results
        
    except Exception as e:
        logger.error(f"Training workflow demonstration failed: {e}")
        return {'success': False, 'error': str(e)}


def demonstrate_evaluation_metrics():
    """Demonstrate evaluation metrics calculation."""
    logger.info("Demonstrating evaluation metrics...")
    
    # Simulate evaluation metrics
    eval_metrics = {
        'eval_loss': 2.3,
        'perplexity': 10.2,
        'eval_runtime': 30.0,
        'eval_samples_per_second': 2.5,
        'eval_steps_per_second': 1.2
    }
    
    logger.info("Evaluation metrics would include:")
    for metric, value in eval_metrics.items():
        logger.info(f"  {metric}: {value}")
    
    # Demonstrate custom poetry metrics
    poetry_metrics = {
        'stylistic_consistency': 0.85,
        'rhyme_scheme_accuracy': 0.78,
        'meter_consistency': 0.82,
        'vocabulary_similarity': 0.73
    }
    
    logger.info("Poetry-specific metrics would include:")
    for metric, value in poetry_metrics.items():
        logger.info(f"  {metric}: {value:.3f}")
    
    return eval_metrics, poetry_metrics


def main():
    """Main demonstration function."""
    logger.info("=== Poetry Fine-Tuning Demonstration ===")
    
    try:
        # Demonstrate each component
        logger.info("\n1. Configuration Creation")
        config = demonstrate_config_creation()
        
        logger.info("\n2. Data Preparation")
        train_examples, val_examples = demonstrate_data_preparation()
        
        logger.info("\n3. Training Workflow")
        results = demonstrate_training_workflow()
        
        logger.info("\n4. Evaluation Metrics")
        eval_metrics, poetry_metrics = demonstrate_evaluation_metrics()
        
        logger.info("\n=== Demonstration Complete ===")
        logger.info("The fine-tuning system is ready for use!")
        logger.info("To run actual training, ensure you have:")
        logger.info("  - Sufficient GPU memory or CPU resources")
        logger.info("  - A larger poetry corpus for better results")
        logger.info("  - Proper transformers library installation")
        
        return True
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)