# Frequently Asked Questions (FAQ)

## General Questions

### What is the Stylistic Poetry LLM Framework?

The Stylistic Poetry LLM Framework is a sophisticated system for generating poetry that faithfully replicates the distinctive styles of renowned poets. It combines computational stylistics, supervised fine-tuning methodologies, and hybrid evaluation frameworks to achieve precise creative control in AI-generated poetry.

### Which poets are currently supported?

The framework currently supports:
- **Emily Dickinson**: Known for frequent dashes, slant rhyme, and irregular capitalization
- **Walt Whitman**: Characterized by free verse, expansive cataloging, and anaphoric repetition  
- **Edgar Allan Poe**: Features consistent end rhyme, alliteration, and refrains

Additional poets can be added by training custom models on their corpus.

### What makes this different from other poetry generation tools?

Our framework goes beyond simple text generation by:
- Incorporating quantitative stylometric analysis
- Using poet-specific feature extraction
- Providing comprehensive evaluation metrics (both quantitative and qualitative)
- Supporting fine-tuning on custom poet corpora
- Offering detailed analysis of generated poetry

## Installation and Setup

### What are the system requirements?

**Minimum Requirements:**
- Python 3.8+
- 8GB RAM
- 4 CPU cores
- 10GB disk space

**Recommended Requirements:**
- Python 3.9+
- 16GB+ RAM
- 8+ CPU cores
- 50GB+ disk space
- NVIDIA GPU with CUDA support (optional but recommended)

### Why am I getting import errors during installation?

Common causes and solutions:

1. **Missing dependencies**: Run `pip install -r requirements.txt`
2. **Python version**: Ensure you're using Python 3.8 or higher
3. **Virtual environment**: Make sure you've activated your virtual environment
4. **NLTK data**: Download required data with `python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"`

### How do I enable GPU acceleration?

1. **Install CUDA toolkit** (version 11.8 recommended)
2. **Install PyTorch with CUDA support**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
3. **Verify GPU availability**:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   ```
4. **Enable in configuration**:
   ```yaml
   performance:
     enable_gpu: true
   ```

## Usage Questions

### How do I generate poetry in a specific style?

**Command Line:**
```bash
python poetry_cli.py generate --poet emily_dickinson --prompt "A bird in the garden"
```

**Python API:**
```python
from src.stylometric import PoetryLLMSystem

system = PoetryLLMSystem()
system.initialize()

result = system.generate_poetry_end_to_end(
    prompt="A bird in the garden",
    poet_style="emily_dickinson"
)
print(result['generated_text'])
```

### How can I control the quality and style of generated poetry?

1. **Adjust generation parameters**:
   - `temperature`: Lower (0.6-0.8) for consistency, higher (0.9-1.2) for creativity
   - `top_p`: Controls nucleus sampling (0.8-0.95 recommended)
   - `repetition_penalty`: Reduces repetitive text (1.1-1.3)

2. **Use stylistic constraints**:
   ```python
   constraints = {
       "rhyme_scheme": "ABAB",
       "meter": "iambic_pentameter",
       "stanza_count": 4
   }
   ```

3. **Provide better prompts**: Use thematically relevant and specific prompts

### Why is the generated poetry not matching the expected style?

Common issues and solutions:

1. **Insufficient training data**: Ensure the poet model has been properly trained
2. **Inappropriate parameters**: Adjust temperature and sampling parameters
3. **Poor prompt quality**: Use prompts that align with the poet's typical themes
4. **Model limitations**: Consider fine-tuning on additional data

### How do I interpret the evaluation scores?

**Overall Score (0-1)**: Combined metric indicating stylistic similarity
- 0.8-1.0: Excellent match
- 0.6-0.8: Good match
- 0.4-0.6: Fair match
- Below 0.4: Poor match

**Lexical Metrics**:
- `type_token_ratio`: Vocabulary richness (higher = more diverse)
- `lexical_density`: Content word ratio (0.4-0.6 typical for poetry)

**Structural Metrics**:
- `meter_consistency`: Adherence to metrical patterns (0-1)
- `rhyme_accuracy`: Correctness of rhyme schemes (0-1)

## Training and Customization

### How do I train a model for a new poet?

1. **Prepare corpus**: Collect clean text files of the poet's work
2. **Process data**:
   ```python
   from src.stylometric.training_data import TrainingDataProcessor
   
   processor = TrainingDataProcessor()
   training_data = processor.process_corpus(
       corpus_path=Path("./data/corpus/new_poet/"),
       poet_name="new_poet"
   )
   ```
3. **Fine-tune model**:
   ```python
   from src.stylometric.fine_tuning import FineTuningManager
   
   trainer = FineTuningManager()
   model = trainer.prepare_model_for_training("gpt2-medium", "new_poet")
   result = trainer.train_model(model, training_data['examples'], config)
   ```

### What format should the training corpus be in?

- **File format**: Plain text (.txt) files
- **Structure**: One poem per file or poems separated by blank lines
- **Encoding**: UTF-8
- **Size**: Minimum 50 poems, recommended 200+ poems
- **Quality**: Clean, properly formatted text without metadata

### How long does training take?

Training time depends on:
- **Corpus size**: 50 poems (~30 minutes), 500 poems (~3 hours)
- **Model size**: GPT-2 small (~1x), GPT-2 medium (~3x), GPT-2 large (~8x)
- **Hardware**: GPU acceleration reduces time by 5-10x
- **Parameters**: More epochs increase training time proportionally

### Can I use my own base model?

Yes, the framework supports:
- **Hugging Face models**: Any GPT-2 compatible model
- **Custom models**: Models following the transformers library interface
- **Local models**: Pre-downloaded models stored locally

Specify in configuration:
```yaml
model:
  custom_models:
    my_model: "path/to/local/model"
    # or
    my_model: "huggingface_model_name"
```

## Performance and Optimization

### Why is poetry generation slow?

Common causes and solutions:

1. **Large model size**: Use smaller models (gpt2 vs gpt2-large) for faster generation
2. **CPU-only inference**: Enable GPU acceleration if available
3. **High max_length**: Reduce maximum generation length
4. **No caching**: Enable result caching in configuration

### How can I improve generation speed?

1. **Use GPU acceleration**:
   ```yaml
   performance:
     enable_gpu: true
   ```

2. **Optimize model parameters**:
   ```yaml
   model:
     max_length: 150  # Reduce from default 200
     batch_size: 16   # Increase for batch processing
   ```

3. **Enable caching**:
   ```yaml
   performance:
     cache_size: 5000
   ```

4. **Use smaller models for development**:
   ```yaml
   model:
     default_model: "gpt2"  # Instead of gpt2-large
   ```

### How much memory does the system use?

Memory usage varies by configuration:
- **GPT-2 small**: ~500MB
- **GPT-2 medium**: ~1.5GB  
- **GPT-2 large**: ~3GB
- **Additional overhead**: ~500MB for analysis components

For multiple concurrent requests, multiply by the number of active requests.

### Can I run this on a low-resource system?

Yes, with optimizations:

1. **Use smallest model**:
   ```yaml
   model:
     default_model: "distilgpt2"
   ```

2. **Reduce cache size**:
   ```yaml
   performance:
     cache_size: 100
   ```

3. **Limit concurrent requests**:
   ```yaml
   system:
     max_concurrent_requests: 1
   ```

4. **Disable GPU if causing issues**:
   ```yaml
   performance:
     enable_gpu: false
   ```

## Troubleshooting

### The system fails to initialize. What should I check?

1. **Dependencies**: Verify all packages are installed correctly
2. **NLTK data**: Ensure required NLTK datasets are downloaded
3. **File permissions**: Check read/write access to data directories
4. **Configuration**: Validate YAML syntax in configuration files
5. **Logs**: Check logs for specific error messages

### I'm getting CUDA out of memory errors. How do I fix this?

1. **Reduce batch size**:
   ```yaml
   training:
     batch_size: 4  # Reduce from default 8
   ```

2. **Use gradient accumulation**:
   ```yaml
   training:
     gradient_accumulation_steps: 4
   ```

3. **Enable mixed precision**:
   ```yaml
   performance:
     mixed_precision: true
   ```

4. **Clear GPU cache**:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### Generated poetry contains repetitive text. How do I fix this?

1. **Increase repetition penalty**:
   ```yaml
   model:
     repetition_penalty: 1.3  # Increase from default 1.1
   ```

2. **Adjust sampling parameters**:
   ```yaml
   model:
     top_p: 0.9
     top_k: 50
   ```

3. **Use different prompts**: Try more specific or varied prompts
4. **Retrain model**: Consider additional training data or different parameters

### The evaluation scores seem inconsistent. Why?

1. **Small sample size**: Evaluate multiple poems for reliable metrics
2. **Prompt quality**: Poor prompts can lead to inconsistent results
3. **Model variance**: Different random seeds produce different outputs
4. **Reference corpus**: Ensure comparison corpus is representative

## Integration and Development

### How do I integrate this with my existing application?

The framework provides multiple integration options:

1. **Python API**: Direct integration using the PoetryLLMSystem class
2. **REST API**: Use the Flask-based web service
3. **Command line**: Call CLI commands from your application
4. **Docker**: Deploy as a containerized service

### Can I modify the evaluation metrics?

Yes, you can:

1. **Extend existing evaluators**:
   ```python
   from src.stylometric.evaluation_metrics import QuantitativeEvaluator
   
   class CustomEvaluator(QuantitativeEvaluator):
       def custom_metric(self, text):
           # Your custom logic here
           pass
   ```

2. **Add new metrics to configuration**:
   ```yaml
   evaluation:
     custom_metrics:
       - "my_custom_metric"
   ```

3. **Create plugin-style evaluators**: Follow the evaluator interface

### How do I contribute to the project?

1. **Fork the repository** on GitHub
2. **Create a feature branch**: `git checkout -b feature/new-feature`
3. **Follow coding standards**: Use the provided linting configuration
4. **Add tests**: Ensure new features have appropriate test coverage
5. **Submit a pull request**: Include description of changes and tests

### Is there an API for batch processing?

Yes, several options:

1. **Python batch processing**:
   ```python
   results = []
   for prompt in prompts:
       result = system.generate_poetry_end_to_end(prompt, poet_style)
       results.append(result)
   ```

2. **CLI batch processing**:
   ```bash
   python poetry_cli.py batch-generate --input prompts.csv --output results.csv
   ```

3. **REST API batch endpoint**:
   ```bash
   curl -X POST /batch-generate -d @batch_request.json
   ```

## Licensing and Usage

### What license does this project use?

Please refer to the LICENSE file in the repository for current licensing information.

### Can I use this for commercial purposes?

License terms vary. Please review the LICENSE file and any third-party dependencies for commercial usage rights and restrictions.

### How do I cite this work in academic research?

```bibtex
@software{stylistic_poetry_llm,
  title={Stylistic Poetry LLM Framework},
  author={[Author Names]},
  year={2024},
  url={[Repository URL]},
  version={[Version Number]}
}
```

## Getting Help

### Where can I get additional support?

1. **Documentation**: Check the complete documentation in the `docs/` directory
2. **GitHub Issues**: Report bugs and request features
3. **Community**: Join our Discord server for community support
4. **Examples**: Review the `examples/` directory for sample code

### How do I report a bug?

1. **Check existing issues** on GitHub
2. **Provide detailed information**:
   - System configuration
   - Steps to reproduce
   - Expected vs actual behavior
   - Error messages and logs
3. **Include minimal reproduction case** if possible
4. **Use appropriate issue templates**

### How do I request a new feature?

1. **Search existing feature requests** on GitHub
2. **Create a detailed feature request** including:
   - Use case description
   - Proposed implementation approach
   - Benefits and impact
   - Potential alternatives
3. **Engage with the community** for feedback and discussion