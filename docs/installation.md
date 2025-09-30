# Installation Guide

## System Requirements

### Hardware Requirements

- **Minimum**: 8GB RAM, 4 CPU cores, 10GB disk space
- **Recommended**: 16GB+ RAM, 8+ CPU cores, 50GB+ disk space
- **GPU**: Optional but recommended for fine-tuning (NVIDIA GPU with CUDA support)

### Software Requirements

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)
- Optional: CUDA toolkit for GPU acceleration

## Installation Methods

### Method 1: Standard Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-org/stylistic-poetry-llm.git
   cd stylistic-poetry-llm
   ```

2. **Create Virtual Environment**

   ```bash
   # Using venv (recommended)
   python -m venv venv

   # Activate virtual environment
   # On Linux/macOS:
   source venv/bin/activate

   # On Windows:
   venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   # Install core dependencies
   pip install -r requirements.txt

   # Install development dependencies (optional)
   pip install -r requirements-dev.txt
   ```

4. **Download Required Data**

   ```bash
   # Download NLTK data
   python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('stopwords')"

   # Download pronunciation dictionary
   python -c "import pronouncing; pronouncing.phones_for_word('test')"
   ```

5. **Verify Installation**
   ```bash
   python -m pytest tests/test_setup.py -v
   ```

### Method 2: Development Installation

For contributors and developers:

```bash
# Clone with development setup
git clone https://github.com/your-org/stylistic-poetry-llm.git
cd stylistic-poetry-llm

# Install in development mode
pip install -e .

# Install pre-commit hooks
pre-commit install

# Run full test suite
python -m pytest
```

### Method 3: Docker Installation

```bash
# Build Docker image
docker build -t stylistic-poetry-llm .

# Run container
docker run -it --rm -v $(pwd)/data:/app/data stylistic-poetry-llm
```

## Configuration

### Environment Setup

1. **Create Configuration File**

   ```bash
   cp config/default.yaml config/local.yaml
   ```

2. **Edit Configuration** (config/local.yaml)

   ```yaml
   model:
     default_model: "gpt2-medium"
     temperature: 0.8
     max_length: 200

   data:
     data_dir: "./data"
     output_dir: "./output"

   training:
     batch_size: 8
     learning_rate: 5e-5
     num_epochs: 3
   ```

3. **Set Environment Variables** (optional)
   ```bash
   export POETRY_LLM_CONFIG_PATH="./config/local.yaml"
   export POETRY_LLM_LOG_LEVEL="INFO"
   ```

## Verification

### Quick Test

```bash
# Test basic functionality
python -c "from src.stylometric import PoetryLLMSystem; system = PoetryLLMSystem(); print('Installation successful!' if system.initialize() else 'Installation failed')"
```

### Generate Sample Poetry

```bash
python poetry_cli.py generate --poet emily_dickinson --prompt "The morning light" --output sample_poem.txt
```

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'torch'**

   ```bash
   # Install PyTorch separately
   pip install torch torchvision torchaudio
   ```

2. **NLTK Data Not Found**

   ```bash
   python -c "import nltk; nltk.download('all')"
   ```

3. **Permission Errors**

   ```bash
   # Use --user flag
   pip install --user -r requirements.txt
   ```

4. **Memory Issues During Installation**
   ```bash
   # Install with no cache
   pip install --no-cache-dir -r requirements.txt
   ```

### GPU Setup (Optional)

For CUDA-enabled GPU acceleration:

```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Next Steps

After successful installation:

1. Read the [User Guide](user_guide.md) for basic usage
2. Check out [Examples](examples.md) for sample workflows
3. Review [Configuration Guide](configuration.md) for advanced settings
4. See [API Reference](api_reference.md) for programmatic usage

## Getting Help

- Check the [FAQ](faq.md) for common questions
- Review [GitHub Issues](https://github.com/your-org/stylistic-poetry-llm/issues) for known problems
- Join our [Discord community](https://discord.gg/your-invite) for support
