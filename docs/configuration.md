# Configuration Guide

## Overview

The Stylistic Poetry LLM Framework uses YAML-based configuration files to manage system settings, model parameters, and deployment options. This guide covers all configuration options and best practices.

## Configuration Files

### Default Configuration

The system ships with a default configuration at `config/default.yaml`:

```yaml
# System Configuration
system:
  log_level: "INFO"
  enable_error_recovery: true
  validate_dependencies: true
  max_concurrent_requests: 10
  request_timeout: 300

# Model Configuration
model:
  default_model: "gpt2-medium"
  model_cache_dir: "./models"
  temperature: 0.8
  top_p: 0.9
  top_k: 50
  max_length: 200
  min_length: 20
  repetition_penalty: 1.1
  length_penalty: 1.0
  num_return_sequences: 1
  do_sample: true
  early_stopping: true

# Data Configuration
data:
  data_dir: "./data"
  output_dir: "./output"
  cache_dir: "./cache"
  training_data_dir: "./data/training"
  corpus_dir: "./data/corpus"
  models_dir: "./models"

# Training Configuration
training:
  batch_size: 8
  learning_rate: 5e-5
  num_epochs: 3
  warmup_steps: 100
  weight_decay: 0.01
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0
  save_steps: 500
  eval_steps: 100
  logging_steps: 50
  save_total_limit: 3
  evaluation_strategy: "steps"
  load_best_model_at_end: true
  metric_for_best_model: "eval_loss"
  greater_is_better: false

# Evaluation Configuration
evaluation:
  enable_quantitative: true
  enable_qualitative: true
  enable_pimf: true
  comparison_baseline: "original_corpus"
  metrics_to_calculate:
    - "type_token_ratio"
    - "lexical_density"
    - "readability_scores"
    - "structural_adherence"
    - "rhyme_accuracy"
    - "meter_consistency"

# Poet-Specific Settings
poets:
  emily_dickinson:
    model_path: "./models/emily_dickinson"
    features:
      dash_frequency: 0.15
      capitalization_irregularity: 0.25
      slant_rhyme_preference: 0.8
      hymn_meter_adherence: 0.7
    
  walt_whitman:
    model_path: "./models/walt_whitman"
    features:
      free_verse_preference: 0.9
      catalog_frequency: 0.3
      anaphora_usage: 0.4
      line_length_variation: 0.8
    
  edgar_allan_poe:
    model_path: "./models/edgar_allan_poe"
    features:
      end_rhyme_consistency: 0.9
      alliteration_frequency: 0.6
      refrain_usage: 0.4
      meter_regularity: 0.8

# Performance Configuration
performance:
  enable_monitoring: true
  memory_threshold: 0.8
  cpu_threshold: 0.9
  generation_timeout: 60
  analysis_timeout: 30
  cache_size: 1000
  enable_gpu: true
  mixed_precision: false

# Output Configuration
output:
  default_format: "formatted"
  include_analysis: true
  include_metadata: true
  save_intermediate_results: false
  export_formats:
    - "txt"
    - "json"
    - "html"
    - "markdown"

# Logging Configuration
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file_path: "./logs/poetry_llm.log"
  max_file_size: "10MB"
  backup_count: 5
  enable_console: true
  enable_file: true
```

### Local Configuration Override

Create `config/local.yaml` to override default settings:

```yaml
# Local development overrides
system:
  log_level: "DEBUG"

model:
  temperature: 0.9
  max_length: 150

data:
  data_dir: "/path/to/your/data"
  output_dir: "/path/to/your/output"

performance:
  enable_gpu: false  # Disable GPU for CPU-only systems
```

## Environment Variables

You can override configuration using environment variables:

```bash
# System settings
export POETRY_LLM_LOG_LEVEL="DEBUG"
export POETRY_LLM_DATA_DIR="/custom/data/path"
export POETRY_LLM_OUTPUT_DIR="/custom/output/path"

# Model settings
export POETRY_LLM_DEFAULT_MODEL="gpt2-large"
export POETRY_LLM_TEMPERATURE="0.9"
export POETRY_LLM_MAX_LENGTH="250"

# Performance settings
export POETRY_LLM_ENABLE_GPU="true"
export POETRY_LLM_BATCH_SIZE="16"
export POETRY_LLM_CACHE_SIZE="2000"
```

## Configuration Sections

### System Configuration

Controls core system behavior:

- `log_level`: Logging verbosity ("DEBUG", "INFO", "WARNING", "ERROR")
- `enable_error_recovery`: Automatic error recovery and fallback
- `validate_dependencies`: Check required packages on startup
- `max_concurrent_requests`: Maximum simultaneous requests
- `request_timeout`: Request timeout in seconds

### Model Configuration

Controls language model behavior:

- `default_model`: Default model to use for generation
- `temperature`: Randomness in generation (0.1-2.0)
- `top_p`: Nucleus sampling parameter (0.1-1.0)
- `top_k`: Top-k sampling parameter (1-100)
- `max_length`: Maximum generated text length
- `repetition_penalty`: Penalty for repetitive text

### Training Configuration

Controls fine-tuning process:

- `batch_size`: Training batch size
- `learning_rate`: Learning rate for optimization
- `num_epochs`: Number of training epochs
- `warmup_steps`: Learning rate warmup steps
- `weight_decay`: L2 regularization strength

### Poet-Specific Configuration

Each poet can have custom settings:

```yaml
poets:
  custom_poet:
    model_path: "./models/custom_poet"
    features:
      custom_feature_1: 0.5
      custom_feature_2: 0.8
    generation_config:
      temperature: 0.7
      max_length: 180
```

## Advanced Configuration

### GPU Configuration

For systems with CUDA-enabled GPUs:

```yaml
performance:
  enable_gpu: true
  gpu_device: 0  # GPU device ID
  mixed_precision: true  # Enable mixed precision training
  gpu_memory_fraction: 0.8  # Fraction of GPU memory to use
```

### Distributed Training

For multi-GPU training:

```yaml
training:
  distributed: true
  world_size: 4  # Number of GPUs
  backend: "nccl"  # Communication backend
  init_method: "env://"
```

### Custom Model Paths

Specify custom model locations:

```yaml
model:
  custom_models:
    my_poet_model: "/path/to/custom/model"
    experimental_model: "huggingface_model_name"
```

### Advanced Evaluation

Configure detailed evaluation settings:

```yaml
evaluation:
  pimf_config:
    dimensions:
      - "creative_imagination"
      - "linguistic_creativity"
      - "emotional_intensity"
      - "sonic_quality"
    weights:
      creative_imagination: 0.3
      linguistic_creativity: 0.25
      emotional_intensity: 0.25
      sonic_quality: 0.2
  
  quantitative_config:
    lexical_metrics:
      - "type_token_ratio"
      - "lexical_density"
      - "vocabulary_richness"
    structural_metrics:
      - "meter_consistency"
      - "rhyme_accuracy"
      - "stanza_regularity"
```

## Configuration Validation

The system validates configuration on startup. Common validation errors:

### Invalid Values
```yaml
# ERROR: Temperature out of range
model:
  temperature: 5.0  # Should be 0.1-2.0
```

### Missing Required Paths
```yaml
# ERROR: Data directory doesn't exist
data:
  data_dir: "/nonexistent/path"
```

### Incompatible Settings
```yaml
# ERROR: GPU enabled but CUDA not available
performance:
  enable_gpu: true  # But no CUDA installation
```

## Configuration Best Practices

### Development Environment

```yaml
system:
  log_level: "DEBUG"
  enable_error_recovery: true

model:
  default_model: "gpt2"  # Smaller model for faster iteration
  temperature: 0.8

performance:
  enable_monitoring: true
  cache_size: 500  # Smaller cache for development
```

### Production Environment

```yaml
system:
  log_level: "INFO"
  max_concurrent_requests: 50

model:
  default_model: "gpt2-large"  # Better quality
  temperature: 0.7

performance:
  enable_gpu: true
  mixed_precision: true
  cache_size: 5000
  
logging:
  enable_file: true
  backup_count: 10
```

### Training Environment

```yaml
training:
  batch_size: 16  # Larger batch for stability
  learning_rate: 3e-5  # Conservative learning rate
  num_epochs: 5
  save_steps: 100  # Frequent checkpoints
  
performance:
  enable_gpu: true
  mixed_precision: true
  memory_threshold: 0.9
```

## Configuration Loading Order

The system loads configuration in this order (later overrides earlier):

1. Default configuration (`config/default.yaml`)
2. Local configuration (`config/local.yaml`)
3. Environment variables
4. Command-line arguments
5. Runtime overrides

## Troubleshooting Configuration

### Common Issues

**Configuration File Not Found**
```bash
# Specify custom config path
export POETRY_LLM_CONFIG_PATH="/path/to/config.yaml"
```

**Invalid YAML Syntax**
```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('config/local.yaml'))"
```

**Permission Errors**
```bash
# Check file permissions
ls -la config/
chmod 644 config/local.yaml
```

**Memory Issues**
```yaml
# Reduce memory usage
training:
  batch_size: 4
  gradient_accumulation_steps: 4  # Effective batch size still 16

performance:
  cache_size: 100
  memory_threshold: 0.7
```

### Validation Commands

```bash
# Validate configuration
python -m src.cli validate-config --config config/local.yaml

# Test configuration
python -m src.cli test-config --config config/local.yaml --dry-run

# Show effective configuration
python -m src.cli show-config
```

## Configuration Templates

### Minimal Configuration

```yaml
# Minimal working configuration
model:
  default_model: "gpt2"
  temperature: 0.8

data:
  data_dir: "./data"
  output_dir: "./output"
```

### High-Performance Configuration

```yaml
# Optimized for performance
system:
  max_concurrent_requests: 100

model:
  default_model: "gpt2-large"

performance:
  enable_gpu: true
  mixed_precision: true
  cache_size: 10000

training:
  batch_size: 32
  gradient_accumulation_steps: 2
```

### Research Configuration

```yaml
# Optimized for research and experimentation
system:
  log_level: "DEBUG"

evaluation:
  enable_quantitative: true
  enable_qualitative: true
  enable_pimf: true

output:
  include_analysis: true
  include_metadata: true
  save_intermediate_results: true
  export_formats: ["json", "csv", "html"]
```