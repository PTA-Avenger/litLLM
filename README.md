# Stylistic Poetry LLM Framework

A sophisticated framework for generating poetry that faithfully replicates the distinctive styles of renowned poets using computational stylistics and fine-tuned language models.

## Overview

The Stylistic Poetry LLM Framework integrates advanced stylometric analysis, supervised fine-tuning methodologies, and hybrid evaluation frameworks to achieve precise creative control in AI-generated poetry. The system moves beyond simple text generation to provide controlled and artistically intentional computational creativity.

## Features

- **Stylometric Analysis**: Quantitative extraction of structural, lexical, and thematic features from poetry
- **Multi-Poet Style Modeling**: Support for Emily Dickinson, Walt Whitman, Edgar Allan Poe, and more
- **Fine-Tuning Architecture**: Supervised fine-tuning on poet-specific datasets
- **Comprehensive Evaluation**: Both quantitative metrics and qualitative assessment using PIMF framework
- **Command-Line Interface**: Intuitive CLI for poetry generation and analysis
- **Performance Monitoring**: Real-time metrics and resource utilization tracking

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd stylistic-poetry-llm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download required NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

### Basic Usage

```bash
# Generate poetry in Emily Dickinson's style
python -m src.cli.main generate --poet emily_dickinson --prompt "A bird came down the walk"

# Analyze existing poetry
python -m src.cli.main analyze --file path/to/poem.txt --poet emily_dickinson

# Train a new poet model
python -m src.cli.main train --poet custom_poet --corpus path/to/corpus/
```

## Documentation

### User Documentation
- [Installation Guide](docs/installation.md) - Complete setup instructions for all environments
- [User Guide](docs/user_guide.md) - Comprehensive usage guide with examples
- [Training Data Preparation](docs/training_data_preparation.md) - Complete guide for preparing training data
- [Configuration Guide](docs/configuration.md) - Detailed configuration options and best practices
- [Examples](docs/examples.md) - Practical examples and tutorials
- [FAQ](docs/faq.md) - Frequently asked questions and troubleshooting

### Developer Documentation
- [API Reference](docs/api_reference.md) - Complete API documentation
- [Development Guide](docs/development.md) - Contributing and development guidelines

### Deployment Documentation
- [Deployment Guide](deployment/README.md) - Docker and production deployment instructions
- [AWS Deployment Guide](docs/aws_deployment.md) - Complete AWS deployment and training guide

## Architecture

The framework consists of several key components:

- **Text Processing**: Poetry-specific text cleaning and preprocessing
- **Stylometric Analysis**: Feature extraction for structural, lexical, and thematic elements
- **Poet Profiles**: Quantitative stylistic profiles for different poets
- **Training Pipeline**: Data curation and fine-tuning workflows
- **Generation Engine**: Style-aware poetry generation
- **Evaluation Framework**: Quantitative and qualitative assessment tools

## Supported Poets

Currently supported poet styles:

- **Emily Dickinson**: Frequent dashes, slant rhyme, irregular capitalization
- **Walt Whitman**: Free verse, expansive cataloging, anaphoric repetition
- **Edgar Allan Poe**: Consistent end rhyme, alliteration, refrains

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.0+
- NLTK 3.6+
- See `requirements.txt` for complete list

## License

[License information]

## Contributing

Please read our [Contributing Guide](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{stylistic_poetry_llm,
  title={Stylistic Poetry LLM Framework},
  author={[Author Names]},
  year={2024},
  url={[Repository URL]}
}
```