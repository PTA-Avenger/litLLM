# API Reference

## Core Classes

### PoetryLLMSystem

Main system integration class that coordinates all components.

```python
class PoetryLLMSystem:
    def __init__(self, config_path: Optional[Path] = None)
    def initialize(self, log_level: str = "INFO", enable_error_recovery: bool = True, validate_dependencies: bool = True) -> bool
    def generate_poetry_end_to_end(self, prompt: str, poet_style: Optional[str] = None, model_name: str = "gpt2", **kwargs) -> Dict[str, Any]
    def analyze_existing_poetry(self, text: str, compare_with: Optional[str] = None) -> Dict[str, Any]
    def get_system_status(self) -> SystemStatus
    def cleanup(self) -> None
```

#### Methods

**`initialize(log_level, enable_error_recovery, validate_dependencies)`**
- Initializes the complete system with all components
- **Parameters:**
  - `log_level` (str): Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
  - `enable_error_recovery` (bool): Enable automatic error recovery
  - `validate_dependencies` (bool): Validate required packages
- **Returns:** bool - True if successful

**`generate_poetry_end_to_end(prompt, poet_style, model_name, **kwargs)`**
- Generates poetry with complete analysis and formatting
- **Parameters:**
  - `prompt` (str): Input prompt for generation
  - `poet_style` (str, optional): Target poet style
  - `model_name` (str): Model to use for generation
  - `**kwargs`: Additional generation parameters
- **Returns:** Dict with generated poetry and analysis

**`analyze_existing_poetry(text, compare_with)`**
- Analyzes existing poetry for stylistic features
- **Parameters:**
  - `text` (str): Poetry text to analyze
  - `compare_with` (str, optional): Poet style to compare against
- **Returns:** Dict with comprehensive analysis results

## Text Processing

### PoetryTextProcessor

Handles poetry-specific text processing and cleaning.

```python
class PoetryTextProcessor:
    def clean_text(self, text: str) -> str
    def tokenize_poetry(self, text: str) -> List[str]
    def extract_lines(self, text: str) -> List[str]
    def extract_stanzas(self, text: str) -> List[List[str]]
    def normalize_punctuation(self, text: str) -> str
    def preserve_poetic_formatting(self, text: str) -> str
```

#### Methods

**`clean_text(text)`**
- Cleans and normalizes poetry text while preserving poetic elements
- **Parameters:** `text` (str) - Raw poetry text
- **Returns:** str - Cleaned text

**`tokenize_poetry(text)`**
- Tokenizes poetry with awareness of poetic conventions
- **Parameters:** `text` (str) - Poetry text
- **Returns:** List[str] - Tokens

## Analysis Components

### LexicalAnalyzer

Performs lexical analysis on poetry text.

```python
class LexicalAnalyzer:
    def calculate_type_token_ratio(self, tokens: List[str]) -> float
    def calculate_lexical_density(self, tokens: List[str]) -> float
    def analyze_word_frequencies(self, tokens: List[str]) -> Dict[str, int]
    def calculate_readability_scores(self, text: str) -> Dict[str, float]
    def analyze_pos_distribution(self, tokens: List[str]) -> Dict[str, float]
    def get_vocabulary_richness(self, tokens: List[str]) -> Dict[str, float]
```

### StructuralAnalyzer

Analyzes structural elements of poetry.

```python
class StructuralAnalyzer:
    def analyze_meter(self, text: str) -> Dict[str, Any]
    def analyze_rhyme_scheme(self, lines: List[str]) -> str
    def count_syllables(self, text: str) -> int
    def analyze_stanza_structure(self, text: str) -> Dict[str, Any]
    def detect_poetic_devices(self, text: str) -> Dict[str, List[str]]
    def analyze_line_structure(self, lines: List[str]) -> Dict[str, Any]
```

## Model Interface

### PoetryGenerationRequest

Request object for poetry generation.

```python
@dataclass
class PoetryGenerationRequest:
    prompt: str
    poet_style: Optional[str] = None
    generation_config: Optional[GenerationConfig] = None
    constraints: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
```

### GenerationConfig

Configuration for poetry generation parameters.

```python
@dataclass
class GenerationConfig:
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    max_length: int = 200
    min_length: int = 20
    repetition_penalty: float = 1.1
    length_penalty: float = 1.0
    num_return_sequences: int = 1
    do_sample: bool = True
    early_stopping: bool = True
```

### PoetryGenerationResponse

Response object containing generated poetry and metadata.

```python
@dataclass
class PoetryGenerationResponse:
    generated_text: str
    success: bool
    error_message: Optional[str] = None
    generation_metadata: Optional[Dict[str, Any]] = None
    stylistic_analysis: Optional[Dict[str, Any]] = None
    confidence_score: Optional[float] = None
```

## Poet Profiles

### PoetProfile

Represents a poet's stylistic profile.

```python
@dataclass
class PoetProfile:
    name: str
    structural_features: Dict[str, Any]
    lexical_features: Dict[str, Any]
    thematic_features: Dict[str, Any]
    historical_context: Optional[Dict[str, Any]] = None
    example_poems: Optional[List[str]] = None
    training_corpus_info: Optional[Dict[str, Any]] = None
```

### PoetProfileManager

Manages poet profiles and their operations.

```python
class PoetProfileManager:
    def load_profile(self, poet_name: str) -> Optional[PoetProfile]
    def save_profile(self, profile: PoetProfile) -> bool
    def list_available_poets(self) -> List[str]
    def create_profile_from_corpus(self, poet_name: str, corpus_path: Path) -> PoetProfile
    def compare_profiles(self, profile1: PoetProfile, profile2: PoetProfile) -> Dict[str, float]
    def validate_profile(self, profile: PoetProfile) -> bool
```

## Training Components

### TrainingDataProcessor

Processes and prepares training data for fine-tuning.

```python
class TrainingDataProcessor:
    def process_corpus(self, corpus_path: Path, poet_name: str) -> Dict[str, Any]
    def create_training_examples(self, poems: List[str], poet_style: str) -> List[Dict[str, str]]
    def augment_training_data(self, examples: List[Dict[str, str]]) -> List[Dict[str, str]]
    def validate_training_data(self, data: List[Dict[str, str]]) -> bool
    def split_dataset(self, data: List[Dict[str, str]], train_ratio: float = 0.8) -> Tuple[List, List]
```

### FineTuningManager

Manages the fine-tuning process for poet-specific models.

```python
class FineTuningManager:
    def prepare_model_for_training(self, base_model: str, poet_style: str) -> Any
    def train_model(self, model: Any, training_data: List[Dict[str, str]], config: TrainingConfig) -> TrainingResult
    def evaluate_model(self, model: Any, test_data: List[Dict[str, str]]) -> Dict[str, float]
    def save_trained_model(self, model: Any, save_path: Path) -> bool
    def load_trained_model(self, model_path: Path) -> Any
```

## Evaluation Framework

### QuantitativeEvaluator

Provides quantitative evaluation metrics for generated poetry.

```python
class QuantitativeEvaluator:
    def evaluate_poetry(self, text: str, reference_style: Optional[str] = None) -> EvaluationResult
    def calculate_lexical_metrics(self, text: str) -> Dict[str, float]
    def calculate_structural_metrics(self, text: str) -> Dict[str, float]
    def calculate_readability_metrics(self, text: str) -> Dict[str, float]
    def compare_with_reference(self, generated: str, reference: str) -> Dict[str, float]
```

### EvaluationComparator

Compares poetry across different dimensions and provides comparative analysis.

```python
class EvaluationComparator:
    def compare_poetry_side_by_side(self, text1: str, text2: str, poet_style: str) -> ComparisonResult
    def batch_compare(self, poems: List[str], reference_poet: str) -> List[ComparisonResult]
    def generate_comparison_report(self, comparisons: List[ComparisonResult]) -> str
    def visualize_comparison(self, comparison: ComparisonResult) -> Any
```

## Specialized Components

### DickinsonFeatureExtractor

Extracts Emily Dickinson-specific stylistic features.

```python
class DickinsonFeatureExtractor:
    def extract_dash_usage(self, text: str) -> Dict[str, Any]
    def analyze_capitalization_patterns(self, text: str) -> Dict[str, Any]
    def detect_slant_rhymes(self, lines: List[str]) -> List[Tuple[str, str]]
    def analyze_hymn_meter(self, text: str) -> Dict[str, Any]
    def extract_nature_imagery(self, text: str) -> List[str]
```

### DickinsonStyleGenerator

Generates poetry in Emily Dickinson's specific style.

```python
class DickinsonStyleGenerator:
    def generate_with_dickinson_features(self, prompt: str, config: GenerationConfig) -> str
    def apply_dickinson_formatting(self, text: str) -> str
    def ensure_dickinson_constraints(self, text: str) -> str
    def validate_dickinson_style(self, text: str) -> bool
```

## Output Formatting

### PoetryOutputFormatter

Formats poetry output with analysis and metadata.

```python
class PoetryOutputFormatter:
    def create_comprehensive_output(self, poem_text: str, analysis_results: Dict, **kwargs) -> str
    def format_for_display(self, poem_text: str) -> str
    def format_for_export(self, poem_text: str, format_type: str) -> str
    def create_analysis_report(self, analysis_results: Dict) -> str
    def format_comparison_results(self, comparison: ComparisonResult) -> str
```

## Utility Functions

### Configuration Management

```python
def get_config() -> SystemConfig
def load_config(config_path: Path) -> SystemConfig
def validate_config(config: SystemConfig) -> bool
def update_config(config: SystemConfig, updates: Dict[str, Any]) -> SystemConfig
```

### Error Handling

```python
class PoetryLLMError(Exception): pass
class ConfigurationError(PoetryLLMError): pass
class ValidationError(PoetryLLMError): pass
class ModelError(PoetryLLMError): pass
class TrainingError(PoetryLLMError): pass
```

### Performance Monitoring

```python
def create_performance_monitor() -> PerformanceMonitor
def track_generation_time(func: Callable) -> Callable
def monitor_memory_usage() -> Dict[str, float]
def log_performance_metrics(metrics: Dict[str, Any]) -> None
```

## Data Structures

### EvaluationResult

```python
@dataclass
class EvaluationResult:
    overall_score: float
    lexical_metrics: Dict[str, float]
    structural_metrics: Dict[str, float]
    readability_metrics: Dict[str, float]
    poet_specific_metrics: Optional[Dict[str, float]] = None
    detailed_analysis: Optional[Dict[str, Any]] = None
```

### ComparisonResult

```python
@dataclass
class ComparisonResult:
    similarity_score: float
    differences: Dict[str, Any]
    recommendations: List[str]
    detailed_comparison: Dict[str, Any]
```

### TrainingConfig

```python
@dataclass
class TrainingConfig:
    learning_rate: float = 5e-5
    batch_size: int = 8
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    save_steps: int = 500
    eval_steps: int = 100
```

## Usage Examples

### Basic Generation

```python
from src.stylometric import PoetryLLMSystem

# Initialize system
system = PoetryLLMSystem()
system.initialize()

# Generate poetry
result = system.generate_poetry_end_to_end(
    prompt="The morning sun",
    poet_style="emily_dickinson"
)

print(result['formatted_output'])
```

### Custom Analysis

```python
from src.stylometric.lexical_analysis import LexicalAnalyzer
from src.stylometric.structural_analysis import StructuralAnalyzer

# Analyze custom text
lexical = LexicalAnalyzer()
structural = StructuralAnalyzer()

text = "Hope is the thing with feathers..."
lexical_metrics = lexical.calculate_type_token_ratio(text.split())
rhyme_scheme = structural.analyze_rhyme_scheme(text.split('\n'))
```

### Training Custom Model

```python
from src.stylometric.fine_tuning import FineTuningManager
from src.stylometric.training_data import TrainingDataProcessor

# Prepare training data
processor = TrainingDataProcessor()
training_data = processor.process_corpus(
    corpus_path=Path("./data/custom_poet/"),
    poet_name="custom_poet"
)

# Train model
trainer = FineTuningManager()
model = trainer.prepare_model_for_training("gpt2", "custom_poet")
result = trainer.train_model(model, training_data['examples'], TrainingConfig())
```