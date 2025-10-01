# User Guide

## Getting Started

The Stylistic Poetry LLM Framework provides multiple ways to generate and analyze poetry. This guide covers the most common use cases and workflows.

## Command Line Interface

### Basic Poetry Generation

Generate poetry in a specific poet's style:

```bash
# Generate Emily Dickinson style poetry
python poetry_cli.py generate --poet emily_dickinson --prompt "A bird came down the walk"

# Generate Walt Whitman style poetry
python poetry_cli.py generate --poet walt_whitman --prompt "I celebrate myself"

# Generate Edgar Allan Poe style poetry
python poetry_cli.py generate --poet edgar_allan_poe --prompt "Once upon a midnight dreary"
```

### Advanced Generation Options

```bash
# Control generation parameters
python poetry_cli.py generate \
    --poet emily_dickinson \
    --prompt "The morning light" \
    --temperature 0.8 \
    --max_length 150 \
    --num_poems 3 \
    --output poems.txt

# Generate with specific stylistic constraints
python poetry_cli.py generate \
    --poet emily_dickinson \
    --prompt "Nature's beauty" \
    --rhyme_scheme "ABAB" \
    --meter "iambic_tetrameter" \
    --stanza_count 4
```

### Poetry Analysis

Analyze existing poetry for stylistic features:

```bash
# Analyze a single poem
python poetry_cli.py analyze --file poem.txt --poet emily_dickinson

# Analyze multiple poems
python poetry_cli.py analyze --directory poems/ --poet walt_whitman --output analysis_report.json

# Compare generated poetry to original
python poetry_cli.py compare --generated generated_poem.txt --original original_poem.txt --poet emily_dickinson
```

### Training New Models

Train a model on a custom poet's corpus:

```bash
# Quick training with the training script
python train_poet_model.py \
    --poet robert_frost \
    --corpus data/corpus/frost_poems.txt \
    --base-model gpt2-medium \
    --epochs 3 \
    --batch-size 8

# Advanced training with evaluation
python train_poet_model.py \
    --poet custom_poet \
    --corpus data/corpus/custom_poet/ \
    --base-model gpt2-large \
    --epochs 5 \
    --batch-size 16 \
    --learning-rate 3e-5 \
    --use-gpu \
    --evaluate
```

For detailed training instructions including Kaggle and AWS deployment, see the [Training Guide](training_guide.md).

## Python API

### Basic Usage

```python
from src.stylometric import PoetryLLMSystem
from src.stylometric.model_interface import PoetryGenerationRequest, GenerationConfig

# Initialize system
system = PoetryLLMSystem()
system.initialize()

# Generate poetry
config = GenerationConfig(
    temperature=0.8,
    max_length=200,
    top_p=0.9
)

request = PoetryGenerationRequest(
    prompt="The autumn leaves",
    poet_style="emily_dickinson",
    generation_config=config
)

result = system.generate_poetry_end_to_end(
    prompt="The autumn leaves",
    poet_style="emily_dickinson",
    model_name="gpt2-medium"
)

print(result['formatted_output'])
```

### Advanced API Usage

```python
# Analyze poetry stylistically
analysis = system.analyze_existing_poetry(
    text="Hope is the thing with feathers",
    compare_with="emily_dickinson"
)

print(f"Stylistic similarity: {analysis['overall_score']}")
print(f"Lexical metrics: {analysis['analysis_results']['lexical_metrics']}")

# Create custom poet profile
from src.stylometric.poet_profile import PoetProfile

profile = PoetProfile(
    name="custom_poet",
    structural_features={
        "avg_line_length": 8.5,
        "stanza_patterns": ["quatrain", "tercet"],
        "rhyme_schemes": ["ABAB", "AABA"]
    },
    lexical_features={
        "type_token_ratio": 0.65,
        "avg_word_length": 4.2,
        "pos_frequencies": {"NOUN": 0.25, "VERB": 0.18, "ADJ": 0.15}
    }
)

# Save profile for later use
system.profile_manager.save_profile(profile)
```

## Supported Poets and Styles

### Emily Dickinson
- **Characteristics**: Frequent dashes, slant rhyme, irregular capitalization
- **Typical Forms**: Short lyrics, hymn meter, unconventional punctuation
- **Example Usage**: `--poet emily_dickinson`

### Walt Whitman
- **Characteristics**: Free verse, long lines, cataloging, anaphora
- **Typical Forms**: Expansive verses, parallel structure, democratic themes
- **Example Usage**: `--poet walt_whitman`

### Edgar Allan Poe
- **Characteristics**: Regular meter, internal rhyme, dark themes, refrains
- **Typical Forms**: Ballad meter, trochaic octameter, narrative poetry
- **Example Usage**: `--poet edgar_allan_poe`

## Configuration Options

### Generation Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `temperature` | Creativity/randomness | 0.8 | 0.1-2.0 |
| `top_p` | Nucleus sampling | 0.9 | 0.1-1.0 |
| `top_k` | Top-k sampling | 50 | 1-100 |
| `max_length` | Maximum poem length | 200 | 50-500 |
| `repetition_penalty` | Avoid repetition | 1.1 | 1.0-2.0 |

### Stylistic Constraints

```python
constraints = {
    "rhyme_scheme": "ABAB",  # Specific rhyme pattern
    "meter": "iambic_pentameter",  # Metrical pattern
    "stanza_count": 4,  # Number of stanzas
    "line_count_per_stanza": 4,  # Lines per stanza
    "syllable_count": [10, 10, 10, 10],  # Syllables per line
    "theme": "nature",  # Thematic constraint
    "mood": "contemplative"  # Emotional tone
}
```

## Evaluation and Quality Assessment

### Quantitative Metrics

The system provides comprehensive quantitative evaluation:

```python
# Get detailed metrics
metrics = result['analysis_results']

print(f"Lexical richness (TTR): {metrics['lexical_metrics']['type_token_ratio']}")
print(f"Structural adherence: {metrics['structural_metrics']['meter_consistency']}")
print(f"Readability score: {metrics['readability_metrics']['flesch_reading_ease']}")
```

### Qualitative Assessment

Using the PIMF (Poetic Intensity Measurement Framework):

```python
# PIMF evaluation dimensions
pimf_scores = result['poet_specific_analysis']['pimf_scores']

print(f"Creative imagination: {pimf_scores['creative_imagination']}")
print(f"Linguistic creativity: {pimf_scores['linguistic_creativity']}")
print(f"Emotional intensity: {pimf_scores['emotional_intensity']}")
```

## Best Practices

### For High-Quality Generation

1. **Use Appropriate Prompts**: Start with thematically relevant prompts
2. **Adjust Temperature**: Lower (0.6-0.8) for consistency, higher (0.9-1.2) for creativity
3. **Iterate and Refine**: Generate multiple versions and select the best
4. **Validate Style**: Use analysis tools to verify stylistic adherence

### For Training Custom Models

1. **Curate Quality Data**: Ensure clean, well-formatted training corpus
2. **Balanced Dataset**: Include diverse examples of the poet's work
3. **Proper Preprocessing**: Use the built-in data preparation tools
4. **Monitor Training**: Watch for overfitting and adjust parameters

### For Analysis and Evaluation

1. **Use Multiple Metrics**: Combine quantitative and qualitative assessment
2. **Compare Baselines**: Evaluate against known poet examples
3. **Consider Context**: Account for poem length and complexity
4. **Validate Results**: Cross-check with literary expertise

## Troubleshooting

### Common Issues

**Poor Quality Output**
- Adjust temperature and sampling parameters
- Check prompt relevance and clarity
- Verify poet profile accuracy

**Slow Generation**
- Reduce max_length parameter
- Use smaller base models for faster inference
- Enable GPU acceleration if available

**Style Inconsistency**
- Increase training data quality and quantity
- Fine-tune model parameters
- Use stricter stylistic constraints

**Memory Issues**
- Reduce batch size during training
- Use gradient checkpointing
- Process poems in smaller chunks

## Examples and Tutorials

See the [Examples](examples.md) documentation for:
- Complete workflow tutorials
- Sample code snippets
- Common use case implementations
- Integration examples

## Advanced Features

### Custom Poet Profiles
Learn how to create and train models for new poets not included in the default set.

### Batch Processing
Process multiple poems or large corpora efficiently using the batch processing features.

### Integration with Other Tools
Connect the framework with external poetry analysis tools and writing applications.

### Performance Optimization
Optimize generation speed and quality for production deployments.