"""
Fine-tuning system for poetry generation models using HuggingFace Trainer.

This module implements supervised fine-tuning (SFT) methodology with progress monitoring,
checkpointing, and validation metrics calculation during training.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from transformers.trainer_callback import TrainerCallback
import numpy as np

from .training_data import TrainingExample, TrainingDatasetFormatter
from .poet_profile import PoetProfile

logger = logging.getLogger(__name__)


@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning parameters."""
    
    # Model configuration
    base_model_name: str = "gpt2"
    model_max_length: int = 512
    
    # Training parameters
    output_dir: str = "./models/fine_tuned"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    
    # Optimization
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 0
    
    # Evaluation and saving
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.01
    
    # Logging
    logging_steps: int = 100
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    
    # Data processing
    max_seq_length: int = 512
    padding_side: str = "right"
    truncation: bool = True
    
    def to_training_arguments(self) -> TrainingArguments:
        """Convert to HuggingFace TrainingArguments."""
        return TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_steps=self.warmup_steps,
            fp16=self.fp16,
            bf16=self.bf16,
            gradient_checkpointing=self.gradient_checkpointing,
            dataloader_num_workers=self.dataloader_num_workers,
            evaluation_strategy=self.evaluation_strategy,
            eval_steps=self.eval_steps,
            save_strategy=self.save_strategy,
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            load_best_model_at_end=self.load_best_model_at_end,
            metric_for_best_model=self.metric_for_best_model,
            greater_is_better=self.greater_is_better,
            logging_steps=self.logging_steps,
            report_to=self.report_to,
            remove_unused_columns=False,
            push_to_hub=False
        )

class PoetryDataset(Dataset):
    """Dataset class for poetry training data."""
    
    def __init__(self, training_examples: List[TrainingExample], 
                 tokenizer: AutoTokenizer, max_length: int = 512):
        """
        Initialize the poetry dataset.
        
        Args:
            training_examples: List of TrainingExample objects
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.training_examples = training_examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Format examples for training
        self.formatter = TrainingDatasetFormatter()
        self.formatted_examples = self.formatter.format_for_huggingface(
            training_examples, format_style='instruction_following'
        )
        
        logger.info(f"Created dataset with {len(self.formatted_examples)} examples")
    
    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.formatted_examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training example.
        
        Args:
            idx: Index of the example
            
        Returns:
            Dictionary with tokenized input and labels
        """
        example = self.formatted_examples[idx]
        
        # Create the full text for training
        instruction = example['instruction']
        input_text = example['input']
        output_text = example['output']
        
        # Format as instruction-following
        if input_text:
            full_text = f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output_text}"
        else:
            full_text = f"Instruction: {instruction}\nOutput: {output_text}"
        
        # Tokenize
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # For causal language modeling, labels are the same as input_ids
        input_ids = tokenized['input_ids'].squeeze()
        attention_mask = tokenized['attention_mask'].squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone(),
            'poet_name': example['poet_name'],
            'stylometric_features': example['stylometric_features']
        }


class PoetryTrainingCallback(TrainerCallback):
    """Custom callback for monitoring poetry training progress."""
    
    def __init__(self, validation_examples: Optional[List[TrainingExample]] = None):
        """
        Initialize the callback.
        
        Args:
            validation_examples: Optional validation examples for custom metrics
        """
        self.validation_examples = validation_examples
        self.training_metrics = []
        
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Called when logging occurs."""
        if logs:
            # Store training metrics
            step_metrics = {
                'step': state.global_step,
                'epoch': state.epoch,
                **logs
            }
            self.training_metrics.append(step_metrics)
            
            # Log poetry-specific information
            if 'eval_loss' in logs:
                logger.info(f"Step {state.global_step}: "
                           f"Train Loss: {logs.get('train_loss', 'N/A'):.4f}, "
                           f"Eval Loss: {logs['eval_loss']:.4f}")
    
    def on_evaluate(self, args, state, control, model=None, tokenizer=None, **kwargs):
        """Called after evaluation."""
        if self.validation_examples and model and tokenizer:
            # Generate sample poetry for qualitative assessment
            self._generate_validation_samples(model, tokenizer, state.global_step)
    
    def _generate_validation_samples(self, model, tokenizer, step: int):
        """Generate sample poetry for validation."""
        try:
            # Take a few validation examples
            sample_examples = self.validation_examples[:3]
            
            logger.info(f"Generating validation samples at step {step}")
            
            for i, example in enumerate(sample_examples):
                instruction = example.instruction
                input_text = example.input_text
                
                # Create prompt
                if input_text:
                    prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput:"
                else:
                    prompt = f"Instruction: {instruction}\nOutput:"
                
                # Tokenize and generate
                inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=256)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=100,
                        temperature=0.8,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                # Decode generated text
                generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_output = generated[len(prompt):].strip()
                
                logger.info(f"Validation Sample {i+1}:")
                logger.info(f"Prompt: {instruction}")
                logger.info(f"Generated: {generated_output[:200]}...")
                logger.info(f"Expected: {example.output_text[:200]}...")
                logger.info("-" * 50)
                
        except Exception as e:
            logger.warning(f"Failed to generate validation samples: {e}")
class PoetryFineTuner:
    """Main class for fine-tuning poetry generation models."""
    
    def __init__(self, config: FineTuningConfig):
        """
        Initialize the fine-tuner.
        
        Args:
            config: Fine-tuning configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.training_history = []
        
    def prepare_model_and_tokenizer(self) -> bool:
        """
        Load and prepare the base model and tokenizer.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Loading base model: {self.config.base_model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name)
            
            # Set special tokens
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Resize token embeddings if needed
            self.tokenizer.padding_side = self.config.padding_side
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_name,
                torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Resize token embeddings to match tokenizer
            self.model.resize_token_embeddings(len(self.tokenizer))
            
            # Enable gradient checkpointing if specified
            if self.config.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
            
            logger.info("Model and tokenizer prepared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to prepare model and tokenizer: {e}")
            return False
    
    def prepare_datasets(self, train_examples: List[TrainingExample],
                        val_examples: Optional[List[TrainingExample]] = None) -> Tuple[PoetryDataset, Optional[PoetryDataset]]:
        """
        Prepare training and validation datasets.
        
        Args:
            train_examples: Training examples
            val_examples: Optional validation examples
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer not prepared. Call prepare_model_and_tokenizer() first.")
        
        logger.info(f"Preparing datasets - Train: {len(train_examples)}, "
                   f"Val: {len(val_examples) if val_examples else 0}")
        
        train_dataset = PoetryDataset(
            train_examples, 
            self.tokenizer, 
            max_length=self.config.max_seq_length
        )
        
        val_dataset = None
        if val_examples:
            val_dataset = PoetryDataset(
                val_examples,
                self.tokenizer,
                max_length=self.config.max_seq_length
            )
        
        return train_dataset, val_dataset
    
    def setup_trainer(self, train_dataset: PoetryDataset, 
                     val_dataset: Optional[PoetryDataset] = None,
                     custom_callbacks: Optional[List[TrainerCallback]] = None) -> bool:
        """
        Set up the HuggingFace Trainer.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            custom_callbacks: Optional custom callbacks
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.model or not self.tokenizer:
                raise ValueError("Model and tokenizer not prepared")
            
            # Create training arguments
            training_args = self.config.to_training_arguments()
            
            # Create data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,  # We're doing causal language modeling
                pad_to_multiple_of=8 if self.config.fp16 else None
            )
            
            # Set up callbacks
            callbacks = []
            
            # Add early stopping if validation dataset is provided
            if val_dataset:
                callbacks.append(EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience,
                    early_stopping_threshold=self.config.early_stopping_threshold
                ))
            
            # Add custom poetry callback
            poetry_callback = PoetryTrainingCallback(
                validation_examples=val_dataset.training_examples if val_dataset else None
            )
            callbacks.append(poetry_callback)
            
            # Add any additional custom callbacks
            if custom_callbacks:
                callbacks.extend(custom_callbacks)
            
            # Create trainer
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                callbacks=callbacks,
                compute_metrics=self._compute_metrics if val_dataset else None
            )
            
            logger.info("Trainer setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup trainer: {e}")
            return False
    
    def _compute_metrics(self, eval_pred) -> Dict[str, float]:
        """
        Compute custom metrics for evaluation.
        
        Args:
            eval_pred: Evaluation predictions from trainer
            
        Returns:
            Dictionary of computed metrics
        """
        predictions, labels = eval_pred
        
        # Basic perplexity calculation
        # Note: This is a simplified version - in practice you might want more sophisticated metrics
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = predictions[..., :-1, :].contiguous()
        
        # Calculate loss manually for perplexity
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        perplexity = torch.exp(loss).item()
        
        return {
            'perplexity': perplexity,
            'eval_loss': loss.item()
        }
    
    def train(self, train_examples: List[TrainingExample],
              val_examples: Optional[List[TrainingExample]] = None,
              resume_from_checkpoint: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute the fine-tuning process.
        
        Args:
            train_examples: Training examples
            val_examples: Optional validation examples
            resume_from_checkpoint: Optional checkpoint path to resume from
            
        Returns:
            Dictionary with training results and metrics
        """
        try:
            logger.info("Starting fine-tuning process")
            
            # Prepare model and tokenizer
            if not self.prepare_model_and_tokenizer():
                raise RuntimeError("Failed to prepare model and tokenizer")
            
            # Prepare datasets
            train_dataset, val_dataset = self.prepare_datasets(train_examples, val_examples)
            
            # Setup trainer
            if not self.setup_trainer(train_dataset, val_dataset):
                raise RuntimeError("Failed to setup trainer")
            
            # Create output directory
            os.makedirs(self.config.output_dir, exist_ok=True)
            
            # Save configuration
            config_path = os.path.join(self.config.output_dir, "fine_tuning_config.json")
            with open(config_path, 'w') as f:
                json.dump(self.config.__dict__, f, indent=2)
            
            # Start training
            logger.info(f"Training with {len(train_examples)} examples")
            if val_examples:
                logger.info(f"Validating with {len(val_examples)} examples")
            
            train_result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            
            # Save the final model
            self.trainer.save_model()
            self.tokenizer.save_pretrained(self.config.output_dir)
            
            # Extract training metrics
            training_metrics = {
                'train_loss': train_result.training_loss,
                'train_runtime': train_result.metrics.get('train_runtime', 0),
                'train_samples_per_second': train_result.metrics.get('train_samples_per_second', 0),
                'train_steps_per_second': train_result.metrics.get('train_steps_per_second', 0),
                'total_flos': train_result.metrics.get('total_flos', 0),
                'train_steps': train_result.global_step,
                'epoch': train_result.metrics.get('epoch', 0)
            }
            
            # Add validation metrics if available
            if val_examples and hasattr(self.trainer.state, 'log_history'):
                eval_metrics = {}
                for log_entry in self.trainer.state.log_history:
                    if 'eval_loss' in log_entry:
                        eval_metrics.update({k: v for k, v in log_entry.items() 
                                           if k.startswith('eval_')})
                training_metrics.update(eval_metrics)
            
            # Store training history
            self.training_history.append({
                'config': self.config.__dict__,
                'metrics': training_metrics,
                'train_examples_count': len(train_examples),
                'val_examples_count': len(val_examples) if val_examples else 0
            })
            
            # Save training history
            history_path = os.path.join(self.config.output_dir, "training_history.json")
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
            
            logger.info("Fine-tuning completed successfully")
            logger.info(f"Final training loss: {training_metrics['train_loss']:.4f}")
            if 'eval_loss' in training_metrics:
                logger.info(f"Final validation loss: {training_metrics['eval_loss']:.4f}")
            
            return {
                'success': True,
                'model_path': self.config.output_dir,
                'metrics': training_metrics,
                'config': self.config.__dict__
            }
            
        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'model_path': None,
                'metrics': {},
                'config': self.config.__dict__
            }
    
    def evaluate_model(self, test_examples: List[TrainingExample]) -> Dict[str, Any]:
        """
        Evaluate the fine-tuned model on test data.
        
        Args:
            test_examples: Test examples for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.trainer:
            raise ValueError("Model not trained. Call train() first.")
        
        logger.info(f"Evaluating model on {len(test_examples)} test examples")
        
        # Prepare test dataset
        test_dataset, _ = self.prepare_datasets(test_examples)
        
        # Run evaluation
        eval_results = self.trainer.evaluate(eval_dataset=test_dataset)
        
        logger.info("Evaluation completed")
        for metric, value in eval_results.items():
            logger.info(f"{metric}: {value:.4f}")
        
        return eval_results
    
    def save_checkpoint(self, checkpoint_dir: str) -> bool:
        """
        Save a training checkpoint.
        
        Args:
            checkpoint_dir: Directory to save checkpoint
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.trainer:
                raise ValueError("Trainer not initialized")
            
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.trainer.save_model(checkpoint_dir)
            self.tokenizer.save_pretrained(checkpoint_dir)
            
            # Save additional metadata
            metadata = {
                'config': self.config.__dict__,
                'training_step': self.trainer.state.global_step if self.trainer.state else 0,
                'epoch': self.trainer.state.epoch if self.trainer.state else 0
            }
            
            with open(os.path.join(checkpoint_dir, 'checkpoint_metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Checkpoint saved to {checkpoint_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False
    
    def load_checkpoint(self, checkpoint_dir: str) -> bool:
        """
        Load a training checkpoint.
        
        Args:
            checkpoint_dir: Directory containing checkpoint
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            checkpoint_path = Path(checkpoint_dir)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
            
            # Load metadata
            metadata_path = checkpoint_path / 'checkpoint_metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"Loading checkpoint from step {metadata.get('training_step', 'unknown')}")
            
            # Load model and tokenizer
            self.model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
            
            logger.info(f"Checkpoint loaded from {checkpoint_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False


def create_fine_tuning_config(base_model: str = "gpt2",
                             output_dir: str = "./models/fine_tuned",
                             epochs: int = 3,
                             batch_size: int = 4,
                             learning_rate: float = 5e-5,
                             **kwargs) -> FineTuningConfig:
    """
    Create a fine-tuning configuration with common parameters.
    
    Args:
        base_model: Base model name
        output_dir: Output directory for trained model
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        **kwargs: Additional configuration parameters
        
    Returns:
        FineTuningConfig object
    """
    config = FineTuningConfig(
        base_model_name=base_model,
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    # Update with any additional parameters
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


def run_fine_tuning_pipeline(corpus_path: str,
                            poet_name: str,
                            base_model: str = "gpt2",
                            output_dir: str = "./models/fine_tuned",
                            **config_kwargs) -> Dict[str, Any]:
    """
    Run the complete fine-tuning pipeline from corpus to trained model.
    
    Args:
        corpus_path: Path to poetry corpus
        poet_name: Name of the poet
        base_model: Base model to fine-tune
        output_dir: Output directory for trained model
        **config_kwargs: Additional configuration parameters
        
    Returns:
        Dictionary with training results
    """
    from .training_data import PoetryCorpusLoader, TrainingDatasetFormatter
    from .poet_profile import PoetProfileBuilder
    
    try:
        logger.info(f"Starting fine-tuning pipeline for {poet_name}")
        
        # Load corpus
        corpus_loader = PoetryCorpusLoader()
        poems = corpus_loader.load_corpus_from_file(corpus_path, poet_name)
        
        # Validate corpus quality
        quality_report = corpus_loader.validate_corpus_quality(poems)
        if not quality_report['valid']:
            raise ValueError(f"Corpus quality issues: {quality_report['warnings']}")
        
        logger.info(f"Loaded {len(poems)} poems for {poet_name}")
        
        # Build poet profile
        profile_builder = PoetProfileBuilder()
        poet_profile = profile_builder.build_profile_from_poems(poems, poet_name)
        
        # Create training data
        formatter = TrainingDatasetFormatter()
        training_examples = formatter.create_instruction_output_pairs(poems, poet_profile)
        
        # Split data
        splits = formatter.create_dataset_splits(training_examples, train_ratio=0.8, val_ratio=0.2)
        
        # Create configuration
        config = create_fine_tuning_config(
            base_model=base_model,
            output_dir=output_dir,
            **config_kwargs
        )
        
        # Initialize fine-tuner
        fine_tuner = PoetryFineTuner(config)
        
        # Run training
        results = fine_tuner.train(splits['train'], splits['val'])
        
        # Add corpus information to results
        results['corpus_info'] = {
            'poet_name': poet_name,
            'total_poems': len(poems),
            'total_examples': len(training_examples),
            'quality_metrics': quality_report['metrics']
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Fine-tuning pipeline failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'corpus_info': {'poet_name': poet_name}
        }