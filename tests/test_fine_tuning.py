"""
Integration tests for the fine-tuning system.

Tests the complete fine-tuning workflow including data preparation,
model training, and evaluation.
"""

import unittest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.stylometric.fine_tuning import (
    FineTuningConfig,
    PoetryDataset,
    PoetryFineTuner,
    PoetryTrainingCallback,
    create_fine_tuning_config,
    run_fine_tuning_pipeline
)
from src.stylometric.training_data import TrainingExample


class TestFineTuningConfig(unittest.TestCase):
    """Test fine-tuning configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = FineTuningConfig()
        
        self.assertEqual(config.base_model_name, "gpt2")
        self.assertEqual(config.num_train_epochs, 3)
        self.assertEqual(config.per_device_train_batch_size, 4)
        self.assertEqual(config.learning_rate, 5e-5)
        self.assertTrue(config.gradient_checkpointing)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = FineTuningConfig(
            base_model_name="microsoft/DialoGPT-small",
            num_train_epochs=5,
            learning_rate=1e-4
        )
        
        self.assertEqual(config.base_model_name, "microsoft/DialoGPT-small")
        self.assertEqual(config.num_train_epochs, 5)
        self.assertEqual(config.learning_rate, 1e-4)
    
    def test_to_training_arguments(self):
        """Test conversion to HuggingFace TrainingArguments."""
        config = FineTuningConfig(output_dir="./test_output")
        training_args = config.to_training_arguments()
        
        self.assertEqual(training_args.output_dir, "./test_output")
        self.assertEqual(training_args.num_train_epochs, 3)
        self.assertEqual(training_args.per_device_train_batch_size, 4)


class TestPoetryDataset(unittest.TestCase):
    """Test poetry dataset class."""
    
    def setUp(self):
        """Set up test data."""
        self.training_examples = [
            TrainingExample(
                instruction="Write a poem in the style of Emily Dickinson.",
                input_text="",
                output_text="I'm Nobody! Who are you?\nAre you - Nobody - Too?",
                stylometric_features={"total_lines": 2, "syllable_count": 16},
                poet_name="Emily Dickinson",
                poem_title="I'm Nobody"
            ),
            TrainingExample(
                instruction="Continue this poem:",
                input_text="Because I could not stop for Death",
                output_text="He kindly stopped for me",
                stylometric_features={"total_lines": 1, "syllable_count": 8},
                poet_name="Emily Dickinson",
                poem_title="Because I could not stop for Death"
            )
        ]
    
    @patch('src.stylometric.fine_tuning.AutoTokenizer')
    def test_dataset_creation(self, mock_tokenizer):
        """Test dataset creation."""
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.return_value = {
            'input_ids': [[1, 2, 3, 4, 5]],
            'attention_mask': [[1, 1, 1, 1, 1]]
        }
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        dataset = PoetryDataset(self.training_examples, mock_tokenizer_instance)
        
        self.assertEqual(len(dataset), 2)
        self.assertEqual(len(dataset.training_examples), 2)
    
    @patch('src.stylometric.fine_tuning.AutoTokenizer')
    def test_dataset_getitem(self, mock_tokenizer):
        """Test dataset item retrieval."""
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.return_value = {
            'input_ids': [[1, 2, 3, 4, 5] + [0] * 507],  # Padded to max_length
            'attention_mask': [[1, 1, 1, 1, 1] + [0] * 507]
        }
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        dataset = PoetryDataset(self.training_examples, mock_tokenizer_instance, max_length=512)
        
        item = dataset[0]
        
        self.assertIn('input_ids', item)
        self.assertIn('attention_mask', item)
        self.assertIn('labels', item)
        self.assertIn('poet_name', item)


class TestPoetryTrainingCallback(unittest.TestCase):
    """Test poetry training callback."""
    
    def setUp(self):
        """Set up test data."""
        self.validation_examples = [
            TrainingExample(
                instruction="Write a poem in the style of Emily Dickinson.",
                input_text="",
                output_text="I'm Nobody! Who are you?",
                stylometric_features={"total_lines": 1},
                poet_name="Emily Dickinson"
            )
        ]
    
    def test_callback_creation(self):
        """Test callback creation."""
        callback = PoetryTrainingCallback(self.validation_examples)
        
        self.assertEqual(len(callback.validation_examples), 1)
        self.assertEqual(len(callback.training_metrics), 0)
    
    def test_on_log(self):
        """Test logging callback."""
        callback = PoetryTrainingCallback()
        
        # Mock trainer state
        mock_state = Mock()
        mock_state.global_step = 100
        mock_state.epoch = 1.0
        
        logs = {'train_loss': 2.5, 'eval_loss': 2.3}
        
        callback.on_log(None, mock_state, None, logs=logs)
        
        self.assertEqual(len(callback.training_metrics), 1)
        self.assertEqual(callback.training_metrics[0]['step'], 100)
        self.assertEqual(callback.training_metrics[0]['train_loss'], 2.5)


class TestPoetryFineTuner(unittest.TestCase):
    """Test poetry fine-tuner."""
    
    def setUp(self):
        """Set up test configuration and data."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = FineTuningConfig(
            base_model_name="gpt2",
            output_dir=self.temp_dir,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            logging_steps=1,
            save_steps=10,
            eval_steps=10
        )
        
        self.training_examples = [
            TrainingExample(
                instruction="Write a poem in the style of Emily Dickinson.",
                input_text="",
                output_text="I'm Nobody! Who are you?\nAre you - Nobody - Too?",
                stylometric_features={"total_lines": 2},
                poet_name="Emily Dickinson"
            )
        ]
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_fine_tuner_creation(self):
        """Test fine-tuner creation."""
        fine_tuner = PoetryFineTuner(self.config)
        
        self.assertEqual(fine_tuner.config.base_model_name, "gpt2")
        self.assertIsNone(fine_tuner.model)
        self.assertIsNone(fine_tuner.tokenizer)
        self.assertEqual(len(fine_tuner.training_history), 0)
    
    @patch('src.stylometric.fine_tuning.AutoTokenizer')
    @patch('src.stylometric.fine_tuning.AutoModelForCausalLM')
    def test_prepare_model_and_tokenizer(self, mock_model, mock_tokenizer):
        """Test model and tokenizer preparation."""
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # Mock model
        mock_model_instance = Mock()
        mock_model_instance.resize_token_embeddings = Mock()
        mock_model_instance.gradient_checkpointing_enable = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        fine_tuner = PoetryFineTuner(self.config)
        result = fine_tuner.prepare_model_and_tokenizer()
        
        self.assertTrue(result)
        self.assertIsNotNone(fine_tuner.model)
        self.assertIsNotNone(fine_tuner.tokenizer)
        mock_tokenizer.from_pretrained.assert_called_once_with("gpt2")
        mock_model.from_pretrained.assert_called_once()
    
    @patch('src.stylometric.fine_tuning.AutoTokenizer')
    def test_prepare_datasets(self, mock_tokenizer):
        """Test dataset preparation."""
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        fine_tuner = PoetryFineTuner(self.config)
        fine_tuner.tokenizer = mock_tokenizer_instance
        
        train_dataset, val_dataset = fine_tuner.prepare_datasets(
            self.training_examples, self.training_examples
        )
        
        self.assertIsNotNone(train_dataset)
        self.assertIsNotNone(val_dataset)
        self.assertEqual(len(train_dataset), 1)
        self.assertEqual(len(val_dataset), 1)
    
    def test_prepare_datasets_without_tokenizer(self):
        """Test dataset preparation without tokenizer."""
        fine_tuner = PoetryFineTuner(self.config)
        
        with self.assertRaises(ValueError):
            fine_tuner.prepare_datasets(self.training_examples)


class TestFineTuningUtilities(unittest.TestCase):
    """Test fine-tuning utility functions."""
    
    def test_create_fine_tuning_config(self):
        """Test configuration creation utility."""
        config = create_fine_tuning_config(
            base_model="microsoft/DialoGPT-small",
            epochs=5,
            batch_size=2,
            learning_rate=1e-4,
            gradient_checkpointing=False
        )
        
        self.assertEqual(config.base_model_name, "microsoft/DialoGPT-small")
        self.assertEqual(config.num_train_epochs, 5)
        self.assertEqual(config.per_device_train_batch_size, 2)
        self.assertEqual(config.learning_rate, 1e-4)
        self.assertFalse(config.gradient_checkpointing)
    
    @patch('src.stylometric.fine_tuning.PoetryCorpusLoader')
    @patch('src.stylometric.fine_tuning.PoetProfileBuilder')
    @patch('src.stylometric.fine_tuning.TrainingDatasetFormatter')
    @patch('src.stylometric.fine_tuning.PoetryFineTuner')
    def test_run_fine_tuning_pipeline(self, mock_fine_tuner, mock_formatter, 
                                     mock_profile_builder, mock_corpus_loader):
        """Test complete fine-tuning pipeline."""
        # Mock corpus loader
        mock_loader_instance = Mock()
        mock_poems = [{'text': 'Test poem', 'poet': 'Test Poet', 'title': 'Test'}]
        mock_loader_instance.load_corpus_from_file.return_value = mock_poems
        mock_loader_instance.validate_corpus_quality.return_value = {
            'valid': True, 'warnings': [], 'metrics': {'total_poems': 1}
        }
        mock_corpus_loader.return_value = mock_loader_instance
        
        # Mock profile builder
        mock_profile_instance = Mock()
        mock_profile = Mock()
        mock_profile_instance.build_profile_from_poems.return_value = mock_profile
        mock_profile_builder.return_value = mock_profile_instance
        
        # Mock formatter
        mock_formatter_instance = Mock()
        mock_examples = [Mock()]
        mock_formatter_instance.create_instruction_output_pairs.return_value = mock_examples
        mock_formatter_instance.create_dataset_splits.return_value = {
            'train': mock_examples, 'val': mock_examples
        }
        mock_formatter.return_value = mock_formatter_instance
        
        # Mock fine-tuner
        mock_fine_tuner_instance = Mock()
        mock_fine_tuner_instance.train.return_value = {
            'success': True, 'model_path': './test_model', 'metrics': {}
        }
        mock_fine_tuner.return_value = mock_fine_tuner_instance
        
        # Run pipeline
        result = run_fine_tuning_pipeline(
            corpus_path="test_corpus.txt",
            poet_name="Test Poet",
            base_model="gpt2"
        )
        
        self.assertTrue(result['success'])
        self.assertIn('corpus_info', result)
        self.assertEqual(result['corpus_info']['poet_name'], 'Test Poet')


class TestFineTuningIntegration(unittest.TestCase):
    """Integration tests for fine-tuning workflow."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample corpus file
        self.corpus_path = Path(self.temp_dir) / "test_corpus.txt"
        with open(self.corpus_path, 'w') as f:
            f.write("""I'm Nobody! Who are you?
Are you - Nobody - Too?
Then there's a pair of us!
Don't tell! they'd advertise - you know!

Because I could not stop for Death -
He kindly stopped for me -
The Carriage held but just Ourselves -
And Immortality.""")
    
    def tearDown(self):
        """Clean up integration test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('transformers.Trainer')
    @patch('src.stylometric.fine_tuning.AutoTokenizer')
    @patch('src.stylometric.fine_tuning.AutoModelForCausalLM')
    def test_end_to_end_workflow(self, mock_model, mock_tokenizer, mock_trainer):
        """Test end-to-end fine-tuning workflow."""
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer_instance.save_pretrained = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # Mock model
        mock_model_instance = Mock()
        mock_model_instance.resize_token_embeddings = Mock()
        mock_model_instance.gradient_checkpointing_enable = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        # Mock trainer
        mock_trainer_instance = Mock()
        mock_train_result = Mock()
        mock_train_result.training_loss = 1.5
        mock_train_result.global_step = 100
        mock_train_result.metrics = {'train_runtime': 60.0, 'epoch': 1.0}
        mock_trainer_instance.train.return_value = mock_train_result
        mock_trainer_instance.save_model = Mock()
        mock_trainer_instance.state = Mock()
        mock_trainer_instance.state.log_history = []
        mock_trainer.return_value = mock_trainer_instance
        
        # Run the pipeline
        result = run_fine_tuning_pipeline(
            corpus_path=str(self.corpus_path),
            poet_name="Emily Dickinson",
            base_model="gpt2",
            output_dir=str(Path(self.temp_dir) / "fine_tuned_model"),
            num_train_epochs=1,
            per_device_train_batch_size=1
        )
        
        # Verify results
        self.assertTrue(result['success'])
        self.assertIn('model_path', result)
        self.assertIn('metrics', result)
        self.assertIn('corpus_info', result)
        self.assertEqual(result['corpus_info']['poet_name'], 'Emily Dickinson')
        
        # Verify trainer was called
        mock_trainer_instance.train.assert_called_once()
        mock_trainer_instance.save_model.assert_called_once()


if __name__ == '__main__':
    unittest.main()