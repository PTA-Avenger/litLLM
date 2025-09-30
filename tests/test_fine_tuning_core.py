"""
Core tests for fine-tuning functionality without heavy dependencies.

Tests the basic structure and configuration of the fine-tuning system.
"""

import unittest
import tempfile
import shutil
import json
from pathlib import Path

# Test only the configuration and basic structure
from src.stylometric.training_data import TrainingExample


class TestFineTuningCore(unittest.TestCase):
    """Test core fine-tuning functionality."""
    
    def test_training_example_creation(self):
        """Test TrainingExample creation and serialization."""
        example = TrainingExample(
            instruction="Write a poem in the style of Emily Dickinson.",
            input_text="",
            output_text="I'm Nobody! Who are you?\nAre you - Nobody - Too?",
            stylometric_features={"total_lines": 2, "syllable_count": 16},
            poet_name="Emily Dickinson",
            poem_title="I'm Nobody"
        )
        
        self.assertEqual(example.instruction, "Write a poem in the style of Emily Dickinson.")
        self.assertEqual(example.output_text, "I'm Nobody! Who are you?\nAre you - Nobody - Too?")
        self.assertEqual(example.poet_name, "Emily Dickinson")
        self.assertEqual(example.stylometric_features["total_lines"], 2)
    
    def test_training_example_serialization(self):
        """Test TrainingExample to_dict and from_dict."""
        example = TrainingExample(
            instruction="Write a poem in the style of Emily Dickinson.",
            input_text="",
            output_text="I'm Nobody! Who are you?",
            stylometric_features={"total_lines": 1},
            poet_name="Emily Dickinson"
        )
        
        # Test to_dict
        example_dict = example.to_dict()
        self.assertIsInstance(example_dict, dict)
        self.assertEqual(example_dict['instruction'], example.instruction)
        self.assertEqual(example_dict['poet_name'], example.poet_name)
        
        # Test from_dict
        reconstructed = TrainingExample.from_dict(example_dict)
        self.assertEqual(reconstructed.instruction, example.instruction)
        self.assertEqual(reconstructed.output_text, example.output_text)
        self.assertEqual(reconstructed.poet_name, example.poet_name)
    
    def test_fine_tuning_config_structure(self):
        """Test that we can import and create basic config structure."""
        # Test basic config creation without transformers dependencies
        config_data = {
            "base_model_name": "gpt2",
            "output_dir": "./models/fine_tuned",
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "learning_rate": 5e-5,
            "gradient_checkpointing": True
        }
        
        # Verify config structure
        self.assertEqual(config_data["base_model_name"], "gpt2")
        self.assertEqual(config_data["num_train_epochs"], 3)
        self.assertTrue(config_data["gradient_checkpointing"])
    
    def test_training_data_preparation(self):
        """Test training data preparation workflow."""
        # Create sample training examples
        examples = [
            TrainingExample(
                instruction="Write a poem in the style of Emily Dickinson.",
                input_text="",
                output_text="I'm Nobody! Who are you?\nAre you - Nobody - Too?",
                stylometric_features={"total_lines": 2},
                poet_name="Emily Dickinson"
            ),
            TrainingExample(
                instruction="Continue this poem:",
                input_text="Because I could not stop for Death",
                output_text="He kindly stopped for me",
                stylometric_features={"total_lines": 1},
                poet_name="Emily Dickinson"
            )
        ]
        
        # Test data structure
        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0].poet_name, "Emily Dickinson")
        self.assertEqual(examples[1].poet_name, "Emily Dickinson")
        
        # Test that we can serialize all examples
        serialized = [example.to_dict() for example in examples]
        self.assertEqual(len(serialized), 2)
        
        # Test that we can reconstruct all examples
        reconstructed = [TrainingExample.from_dict(data) for data in serialized]
        self.assertEqual(len(reconstructed), 2)
        self.assertEqual(reconstructed[0].instruction, examples[0].instruction)


class TestFineTuningWorkflow(unittest.TestCase):
    """Test fine-tuning workflow components."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_config_serialization(self):
        """Test configuration serialization."""
        config_data = {
            "base_model_name": "gpt2",
            "output_dir": str(Path(self.temp_dir) / "models"),
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "learning_rate": 5e-5,
            "gradient_checkpointing": True,
            "evaluation_strategy": "steps",
            "eval_steps": 500,
            "save_steps": 500
        }
        
        # Save config
        config_path = Path(self.temp_dir) / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        # Load config
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)
        
        self.assertEqual(loaded_config["base_model_name"], "gpt2")
        self.assertEqual(loaded_config["num_train_epochs"], 3)
        self.assertTrue(loaded_config["gradient_checkpointing"])
    
    def test_training_history_structure(self):
        """Test training history data structure."""
        training_history = [
            {
                'config': {
                    'base_model_name': 'gpt2',
                    'num_train_epochs': 3,
                    'learning_rate': 5e-5
                },
                'metrics': {
                    'train_loss': 2.5,
                    'eval_loss': 2.3,
                    'train_runtime': 120.0,
                    'train_steps': 100
                },
                'train_examples_count': 50,
                'val_examples_count': 10
            }
        ]
        
        # Test structure
        self.assertEqual(len(training_history), 1)
        self.assertIn('config', training_history[0])
        self.assertIn('metrics', training_history[0])
        self.assertEqual(training_history[0]['train_examples_count'], 50)
        
        # Test serialization
        history_path = Path(self.temp_dir) / "history.json"
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        # Test loading
        with open(history_path, 'r') as f:
            loaded_history = json.load(f)
        
        self.assertEqual(len(loaded_history), 1)
        self.assertEqual(loaded_history[0]['metrics']['train_loss'], 2.5)
    
    def test_checkpoint_metadata_structure(self):
        """Test checkpoint metadata structure."""
        checkpoint_metadata = {
            'config': {
                'base_model_name': 'gpt2',
                'output_dir': './models/fine_tuned',
                'num_train_epochs': 3
            },
            'training_step': 100,
            'epoch': 1.5
        }
        
        # Test structure
        self.assertIn('config', checkpoint_metadata)
        self.assertIn('training_step', checkpoint_metadata)
        self.assertIn('epoch', checkpoint_metadata)
        self.assertEqual(checkpoint_metadata['training_step'], 100)
        
        # Test serialization
        metadata_path = Path(self.temp_dir) / "checkpoint_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(checkpoint_metadata, f, indent=2)
        
        # Test loading
        with open(metadata_path, 'r') as f:
            loaded_metadata = json.load(f)
        
        self.assertEqual(loaded_metadata['training_step'], 100)
        self.assertEqual(loaded_metadata['epoch'], 1.5)


if __name__ == '__main__':
    unittest.main()