# Development Guide

## Getting Started with Development

### Development Environment Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-org/stylistic-poetry-llm.git
   cd stylistic-poetry-llm
   ```

2. **Set Up Development Environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install development dependencies
   pip install -r requirements-dev.txt
   pip install -e .  # Install in development mode
   
   # Install pre-commit hooks
   pre-commit install
   ```

3. **Verify Installation**
   ```bash
   # Run tests
   python -m pytest
   
   # Check code style
   flake8 src/ tests/
   black --check src/ tests/
   
   # Type checking
   mypy src/
   ```

### Project Structure

```
stylistic-poetry-llm/
├── src/
│   ├── stylometric/           # Core framework modules
│   │   ├── __init__.py
│   │   ├── system_integration.py    # Main system class
│   │   ├── text_processing.py       # Text preprocessing
│   │   ├── lexical_analysis.py      # Lexical feature extraction
│   │   ├── structural_analysis.py   # Structural analysis
│   │   ├── poet_profile.py          # Poet profile management
│   │   ├── training_data.py         # Training data processing
│   │   ├── fine_tuning.py           # Model fine-tuning
│   │   ├── model_interface.py       # Model interaction
│   │   ├── evaluation_metrics.py    # Quantitative evaluation
│   │   ├── evaluation_comparison.py # Comparative evaluation
│   │   └── dickinson_features.py    # Poet-specific features
│   ├── utils/                 # Utility modules
│   ├── cli.py                # Command-line interface
│   └── main.py               # Main application entry
├── tests/                    # Test suite
├── docs/                     # Documentation
├── config/                   # Configuration files
├── data/                     # Data directory
├── examples/                 # Example scripts
├── deployment/               # Deployment configurations
└── requirements.txt          # Dependencies
```

## Development Workflow

### Code Style and Standards

We follow PEP 8 with some modifications:

```python
# Use type hints for all functions
def analyze_poetry(text: str, poet_style: Optional[str] = None) -> Dict[str, Any]:
    """Analyze poetry text for stylistic features.
    
    Args:
        text: The poetry text to analyze
        poet_style: Optional poet style for comparison
        
    Returns:
        Dictionary containing analysis results
        
    Raises:
        ValidationError: If text is invalid
    """
    pass

# Use dataclasses for structured data
@dataclass
class PoetProfile:
    name: str
    structural_features: Dict[str, Any]
    lexical_features: Dict[str, Any]
    
# Use enums for constants
class PoetStyle(Enum):
    EMILY_DICKINSON = "emily_dickinson"
    WALT_WHITMAN = "walt_whitman"
    EDGAR_ALLAN_POE = "edgar_allan_poe"
```

### Testing Guidelines

#### Test Structure

```python
import unittest
from unittest.mock import Mock, patch
from src.stylometric import PoetryLLMSystem

class TestPoetryGeneration(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.system = PoetryLLMSystem()
        self.test_prompt = "The morning sun"
        
    def tearDown(self):
        """Clean up after each test method."""
        self.system.cleanup()
    
    def test_basic_generation(self):
        """Test basic poetry generation functionality."""
        result = self.system.generate_poetry_end_to_end(
            prompt=self.test_prompt,
            poet_style="emily_dickinson"
        )
        
        self.assertTrue(result['success'])
        self.assertIn('generated_text', result)
        self.assertGreater(len(result['generated_text']), 0)
    
    @patch('src.stylometric.model_interface.ModelInterface.generate')
    def test_generation_with_mock(self, mock_generate):
        """Test generation with mocked model interface."""
        mock_generate.return_value = "Mocked poem text"
        
        result = self.system.generate_poetry_end_to_end(
            prompt=self.test_prompt,
            poet_style="emily_dickinson"
        )
        
        mock_generate.assert_called_once()
        self.assertEqual(result['generated_text'], "Mocked poem text")
```

#### Test Categories

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete workflows
4. **Performance Tests**: Test system performance and resource usage

#### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_poet_profile.py

# Run with coverage
python -m pytest --cov=src --cov-report=html

# Run performance tests
python -m pytest tests/test_performance.py -m performance

# Run tests in parallel
python -m pytest -n auto
```

### Adding New Features

#### 1. Adding a New Poet Style

**Step 1: Create Poet Profile**
```python
# In src/stylometric/poet_profiles/shakespeare.py
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ShakespeareProfile:
    """Profile for Shakespearean poetry style."""
    
    @classmethod
    def get_profile(cls) -> Dict[str, Any]:
        return {
            "name": "william_shakespeare",
            "structural_features": {
                "sonnet_form": True,
                "iambic_pentameter": 0.9,
                "rhyme_scheme": "ABABCDCDEFEFGG",
                "volta_position": 9  # Typical volta at line 9
            },
            "lexical_features": {
                "archaic_language": 0.3,
                "metaphor_frequency": 0.4,
                "type_token_ratio": 0.7
            },
            "thematic_features": {
                "love_themes": 0.6,
                "nature_imagery": 0.4,
                "mortality_themes": 0.3
            }
        }
```

**Step 2: Create Feature Extractor**
```python
# In src/stylometric/shakespeare_features.py
from typing import Dict, Any, List
import re

class ShakespeareFeatureExtractor:
    """Extract Shakespeare-specific stylistic features."""
    
    def extract_sonnet_structure(self, text: str) -> Dict[str, Any]:
        """Analyze sonnet structure."""
        lines = text.strip().split('\n')
        
        return {
            "line_count": len(lines),
            "is_sonnet": len(lines) == 14,
            "quatrain_structure": self._analyze_quatrains(lines),
            "couplet_present": self._has_final_couplet(lines)
        }
    
    def detect_iambic_pentameter(self, text: str) -> Dict[str, Any]:
        """Detect iambic pentameter patterns."""
        # Implementation for meter detection
        pass
    
    def extract_archaic_language(self, text: str) -> Dict[str, Any]:
        """Identify archaic language patterns."""
        archaic_words = ['thou', 'thee', 'thy', 'thine', 'doth', 'hath']
        words = text.lower().split()
        
        archaic_count = sum(1 for word in words if word in archaic_words)
        
        return {
            "archaic_word_count": archaic_count,
            "archaic_ratio": archaic_count / len(words) if words else 0,
            "archaic_words_found": [w for w in words if w in archaic_words]
        }
```

**Step 3: Register the New Poet**
```python
# In src/stylometric/poet_profile.py
from .shakespeare_features import ShakespeareFeatureExtractor

class PoetProfileManager:
    def __init__(self):
        self.extractors = {
            "emily_dickinson": DickinsonFeatureExtractor(),
            "walt_whitman": WhitmanFeatureExtractor(),
            "edgar_allan_poe": PoeFeatureExtractor(),
            "william_shakespeare": ShakespeareFeatureExtractor(),  # Add new extractor
        }
```

**Step 4: Add Tests**
```python
# In tests/test_shakespeare_features.py
import unittest
from src.stylometric.shakespeare_features import ShakespeareFeatureExtractor

class TestShakespeareFeatures(unittest.TestCase):
    
    def setUp(self):
        self.extractor = ShakespeareFeatureExtractor()
        self.sonnet_text = """
        Shall I compare thee to a summer's day?
        Thou art more lovely and more temperate:
        Rough winds do shake the darling buds of May,
        And summer's lease hath all too short a date:
        """
    
    def test_archaic_language_detection(self):
        """Test detection of archaic language."""
        result = self.extractor.extract_archaic_language(self.sonnet_text)
        
        self.assertGreater(result['archaic_count'], 0)
        self.assertIn('thee', result['archaic_words_found'])
        self.assertIn('thou', result['archaic_words_found'])
```

#### 2. Adding New Evaluation Metrics

**Step 1: Create Metric Class**
```python
# In src/stylometric/metrics/semantic_coherence.py
from typing import Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer

class SemanticCoherenceMetric:
    """Evaluate semantic coherence of generated poetry."""
    
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def calculate_coherence(self, text: str) -> Dict[str, float]:
        """Calculate semantic coherence score."""
        sentences = text.split('\n')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return {"coherence_score": 1.0, "sentence_count": len(sentences)}
        
        # Get sentence embeddings
        embeddings = self.model.encode(sentences)
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i + 1])
            similarities.append(sim)
        
        coherence_score = np.mean(similarities)
        
        return {
            "coherence_score": float(coherence_score),
            "sentence_count": len(sentences),
            "min_similarity": float(np.min(similarities)),
            "max_similarity": float(np.max(similarities))
        }
```

**Step 2: Integrate with Evaluation System**
```python
# In src/stylometric/evaluation_metrics.py
from .metrics.semantic_coherence import SemanticCoherenceMetric

class QuantitativeEvaluator:
    def __init__(self):
        # ... existing initialization ...
        self.semantic_metric = SemanticCoherenceMetric()
    
    def evaluate_poetry(self, text: str, reference_style: Optional[str] = None) -> EvaluationResult:
        """Enhanced evaluation with semantic coherence."""
        # ... existing evaluation logic ...
        
        # Add semantic coherence
        semantic_results = self.semantic_metric.calculate_coherence(text)
        
        # Include in results
        result.semantic_metrics = semantic_results
        
        return result
```

### Performance Optimization

#### Profiling and Benchmarking

```python
# In scripts/profile_generation.py
import cProfile
import pstats
from src.stylometric import PoetryLLMSystem

def profile_generation():
    """Profile poetry generation performance."""
    system = PoetryLLMSystem()
    system.initialize()
    
    def generate_poems():
        for i in range(10):
            result = system.generate_poetry_end_to_end(
                prompt=f"Test prompt {i}",
                poet_style="emily_dickinson"
            )
    
    # Profile the function
    profiler = cProfile.Profile()
    profiler.enable()
    generate_poems()
    profiler.disable()
    
    # Analyze results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)

if __name__ == "__main__":
    profile_generation()
```

#### Memory Optimization

```python
# Memory-efficient batch processing
class BatchProcessor:
    def __init__(self, batch_size: int = 8):
        self.batch_size = batch_size
    
    def process_batch(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """Process prompts in batches to manage memory."""
        results = []
        
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i:i + self.batch_size]
            batch_results = self._process_single_batch(batch)
            results.extend(batch_results)
            
            # Clear GPU cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results
```

### Documentation Standards

#### Docstring Format

```python
def analyze_meter(self, text: str, expected_meter: Optional[str] = None) -> Dict[str, Any]:
    """Analyze the metrical structure of poetry text.
    
    This function analyzes the syllabic patterns and stress patterns in poetry
    to determine adherence to specific metrical forms like iambic pentameter.
    
    Args:
        text: The poetry text to analyze. Should be properly formatted with
            line breaks preserved.
        expected_meter: Optional expected meter pattern for comparison.
            Supported values: 'iambic_pentameter', 'trochaic_tetrameter', etc.
    
    Returns:
        A dictionary containing:
            - meter_pattern: Detected metrical pattern
            - consistency_score: Float between 0-1 indicating adherence
            - syllable_counts: List of syllable counts per line
            - stress_patterns: List of stress patterns per line
    
    Raises:
        ValueError: If text is empty or contains no valid lines
        UnsupportedMeterError: If expected_meter is not supported
    
    Example:
        >>> analyzer = StructuralAnalyzer()
        >>> result = analyzer.analyze_meter("Shall I compare thee to a summer's day?")
        >>> print(result['meter_pattern'])
        'iambic_pentameter'
    
    Note:
        This function uses the CMU Pronouncing Dictionary for syllable counting
        and may not be accurate for all words, especially proper nouns.
    """
```

#### README Updates

When adding features, update the main README.md:

```markdown
## New Features in v2.1.0

### Shakespeare Style Support
- Added William Shakespeare poet profile
- Sonnet structure analysis
- Iambic pentameter detection
- Archaic language pattern recognition

### Enhanced Evaluation
- Semantic coherence metrics
- Cross-poet style comparison
- Improved PIMF framework integration

### Performance Improvements
- 40% faster generation with optimized caching
- Reduced memory usage for batch processing
- GPU memory management improvements
```

### Release Process

#### Version Management

```bash
# Update version in setup.py and __init__.py
# Create release branch
git checkout -b release/v2.1.0

# Update CHANGELOG.md
# Run full test suite
python -m pytest

# Create release commit
git add .
git commit -m "Release v2.1.0"

# Create tag
git tag -a v2.1.0 -m "Release version 2.1.0"

# Merge to main
git checkout main
git merge release/v2.1.0

# Push changes and tags
git push origin main
git push origin v2.1.0
```

#### Continuous Integration

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', 3.11]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-dev.txt
        pip install -e .
    
    - name: Run tests
      run: |
        python -m pytest --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### Debugging and Troubleshooting

#### Logging Configuration

```python
# In src/stylometric/utils/logging.py
import logging
import sys
from pathlib import Path

def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None):
    """Set up comprehensive logging configuration."""
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # File handler (if specified)
    handlers = [console_handler]
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        handlers=handlers
    )
    
    # Set specific logger levels
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
```

#### Debug Mode

```python
# Enable debug mode in configuration
debug_config = {
    "system": {
        "log_level": "DEBUG",
        "enable_profiling": True,
        "save_intermediate_results": True
    },
    "model": {
        "temperature": 0.0,  # Deterministic generation for debugging
        "do_sample": False
    }
}
```

### Contributing Guidelines

#### Pull Request Process

1. **Fork and Clone**: Fork the repository and clone your fork
2. **Create Branch**: Create a feature branch from `main`
3. **Implement Changes**: Make your changes with appropriate tests
4. **Run Tests**: Ensure all tests pass and coverage is maintained
5. **Update Documentation**: Update relevant documentation
6. **Submit PR**: Create a pull request with clear description

#### Code Review Checklist

- [ ] Code follows project style guidelines
- [ ] All tests pass and coverage is maintained
- [ ] Documentation is updated
- [ ] No breaking changes (or properly documented)
- [ ] Performance impact is considered
- [ ] Security implications are reviewed

#### Issue Templates

```markdown
## Bug Report

**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Initialize system with '...'
2. Call method '....'
3. See error

**Expected behavior**
What you expected to happen.

**Environment:**
- OS: [e.g. Ubuntu 20.04]
- Python version: [e.g. 3.9.7]
- Package version: [e.g. 2.1.0]

**Additional context**
Any other context about the problem.
```

This development guide provides a comprehensive foundation for contributing to and extending the Stylistic Poetry LLM Framework. Follow these guidelines to maintain code quality and project consistency.