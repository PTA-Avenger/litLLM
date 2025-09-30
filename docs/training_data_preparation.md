# Training Data Preparation Guide

## Overview

This guide provides comprehensive instructions for preparing high-quality training data for the Stylistic Poetry LLM Framework. Proper data preparation is crucial for training models that can accurately replicate specific poets' styles.

## Table of Contents

1. [Data Requirements](#data-requirements)
2. [Corpus Collection](#corpus-collection)
3. [Data Formats](#data-formats)
4. [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
5. [Quality Validation](#quality-validation)
6. [Feature Encoding](#feature-encoding)
7. [Dataset Formatting](#dataset-formatting)
8. [Data Augmentation](#data-augmentation)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

## Data Requirements

### Minimum Requirements

- **Corpus Size**: At least 50 poems per poet (recommended: 200+ poems)
- **Text Quality**: Clean, properly formatted text without OCR errors
- **Completeness**: Complete poems, not fragments or excerpts
- **Authenticity**: Verified works by the target poet
- **Diversity**: Representative sample of the poet's different periods and themes

### Recommended Specifications

```python
# Ideal corpus characteristics
CORPUS_SPECS = {
    "minimum_poems": 50,
    "recommended_poems": 200,
    "minimum_lines_per_poem": 4,
    "maximum_lines_per_poem": 100,
    "minimum_words_per_poem": 20,
    "encoding": "UTF-8",
    "line_ending": "\\n",
    "poem_separator": "\\n\\n"
}
```

## Corpus Collection

### 1. Public Domain Sources

#### Project Gutenberg
```bash
# Download from Project Gutenberg
wget https://www.gutenberg.org/files/12242/12242-0.txt -O emily_dickinson_complete.txt
```

#### Poetry Foundation
- High-quality, curated collections
- Proper attribution and metadata
- Modern formatting standards

#### Archive.org
- Historical poetry collections
- Rare and out-of-print works
- Multiple format options

### 2. Academic Sources

#### Digital Libraries
- JSTOR Digital Collections
- HathiTrust Digital Library
- Internet Archive Scholar

#### University Collections
- Many universities provide digitized poetry collections
- Often include scholarly annotations
- High-quality transcriptions

### 3. Specialized Poetry Databases

#### Representative Poets Database
- Comprehensive collections by poet
- Standardized formatting
- Metadata included

## Data Formats

### Supported Input Formats

#### 1. Simple Text Format (.txt)

**Structure:**
```
Poem Title 1

First line of poem
Second line of poem
Third line of poem

Poem Title 2

Another first line
Another second line
```

**Example:**
```
Hope is the Thing with Feathers

Hope is the thing with feathers
That perches in the soul,
And sings the tune without the words,
And never stops at all,

And sweetest in the gale is heard;
And sore must be the storm
That could abash the little bird
That kept so many warm.

I've heard it in the chillest land,
And on the strangest sea;
Yet, never, in extremity,
It asked a crumb of me.

Because I Could Not Stop for Death

Because I could not stop for Death,
He kindly stopped for me;
The carriage held but just ourselves
And Immortality.
```

#### 2. JSON Format (.json)

**Structure:**
```json
{
  "poet": "Emily Dickinson",
  "poems": [
    {
      "title": "Hope is the Thing with Feathers",
      "text": "Hope is the thing with feathers\nThat perches in the soul,\nAnd sings the tune without the words,\nAnd never stops at all,",
      "year": 1861,
      "collection": "Poems by Emily Dickinson"
    }
  ]
}
```

**Alternative JSON Structure:**
```json
[
  {
    "title": "Hope is the Thing with Feathers",
    "content": "Hope is the thing with feathers...",
    "author": "Emily Dickinson",
    "metadata": {
      "year": 1861,
      "collection": "Poems by Emily Dickinson"
    }
  }
]
```

#### 3. CSV Format (.csv)

```csv
title,text,poet,year,collection
"Hope is the Thing with Feathers","Hope is the thing with feathers\nThat perches in the soul...","Emily Dickinson",1861,"Poems by Emily Dickinson"
"Because I Could Not Stop for Death","Because I could not stop for Death\nHe kindly stopped for me...","Emily Dickinson",1863,"Poems by Emily Dickinson"
```

## Data Cleaning and Preprocessing

### 1. Automated Cleaning Script

```python
# scripts/clean_corpus.py
import re
from pathlib import Path
from typing import List, Dict, Any

class CorpusCleaner:
    """Clean and preprocess poetry corpus data."""
    
    def __init__(self):
        self.cleaning_rules = {
            'remove_page_numbers': True,
            'remove_headers_footers': True,
            'normalize_punctuation': True,
            'preserve_line_breaks': True,
            'remove_extra_whitespace': True
        }
    
    def clean_text_file(self, input_path: Path, output_path: Path, poet_name: str):
        """Clean a single text file."""
        
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Apply cleaning rules
        cleaned_content = self.apply_cleaning_rules(content, poet_name)
        
        # Validate cleaned content
        validation_results = self.validate_cleaned_content(cleaned_content)
        
        if validation_results['valid']:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            print(f"✓ Cleaned corpus saved to {output_path}")
        else:
            print(f"✗ Validation failed: {validation_results['errors']}")
    
    def apply_cleaning_rules(self, content: str, poet_name: str) -> str:
        """Apply comprehensive cleaning rules."""
        
        # Remove Project Gutenberg headers/footers
        content = self.remove_gutenberg_metadata(content)
        
        # Remove page numbers and running headers
        content = self.remove_page_artifacts(content)
        
        # Normalize line endings
        content = re.sub(r'\r\n|\r', '\n', content)
        
        # Remove excessive whitespace while preserving poem structure
        content = self.normalize_whitespace(content)
        
        # Fix common OCR errors
        content = self.fix_ocr_errors(content)
        
        # Normalize punctuation
        content = self.normalize_punctuation(content)
        
        return content.strip()
    
    def remove_gutenberg_metadata(self, content: str) -> str:
        """Remove Project Gutenberg metadata."""
        # Remove start marker and everything before it
        start_pattern = r'\*\*\* START OF .*? \*\*\*.*?\n'
        content = re.sub(start_pattern, '', content, flags=re.DOTALL)
        
        # Remove end marker and everything after it
        end_pattern = r'\*\*\* END OF .*? \*\*\*.*'
        content = re.sub(end_pattern, '', content, flags=re.DOTALL)
        
        return content
    
    def remove_page_artifacts(self, content: str) -> str:
        """Remove page numbers, headers, and footers."""
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip page numbers (standalone numbers)
            if re.match(r'^\d+$', line):
                continue
            
            # Skip running headers (repeated short lines)
            if len(line) < 50 and line.isupper():
                continue
            
            # Skip lines that are just punctuation or special characters
            if re.match(r'^[^\w\s]*$', line):
                continue
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def normalize_whitespace(self, content: str) -> str:
        """Normalize whitespace while preserving poem structure."""
        # Split into potential poems (separated by multiple blank lines)
        poems = re.split(r'\n\s*\n\s*\n+', content)
        
        normalized_poems = []
        for poem in poems:
            if not poem.strip():
                continue
            
            # Normalize within each poem
            lines = [line.strip() for line in poem.split('\n')]
            lines = [line for line in lines if line]  # Remove empty lines
            
            if lines:
                normalized_poems.append('\n'.join(lines))
        
        return '\n\n'.join(normalized_poems)
    
    def fix_ocr_errors(self, content: str) -> str:
        """Fix common OCR errors in digitized texts."""
        ocr_fixes = {
            # Common character substitutions
            r'\bm\b': 'in',  # 'm' often misread as 'in'
            r'\btlie\b': 'the',
            r'\btliat\b': 'that',
            r'\bwlien\b': 'when',
            r'\bwliich\b': 'which',
            r'\bwliere\b': 'where',
            r'\bwlio\b': 'who',
            
            # Fix spacing issues
            r'([a-z])([A-Z])': r'\1 \2',  # Missing spaces between words
            r'([.!?])([A-Z])': r'\1 \2',  # Missing spaces after punctuation
            
            # Fix punctuation
            r',,': ',',  # Double commas
            r'\.\.': '.',  # Double periods
            r'\s+([,.!?;:])': r'\1',  # Remove spaces before punctuation
        }
        
        for pattern, replacement in ocr_fixes.items():
            content = re.sub(pattern, replacement, content)
        
        return content
    
    def normalize_punctuation(self, content: str) -> str:
        """Normalize punctuation marks."""
        # Normalize quotes
        content = re.sub(r'["""]', '"', content)
        content = re.sub(r'[''']', "'", content)
        
        # Normalize dashes
        content = re.sub(r'—|–', '—', content)  # Use em dash consistently
        
        # Normalize ellipses
        content = re.sub(r'\.{3,}', '...', content)
        
        return content
    
    def validate_cleaned_content(self, content: str) -> Dict[str, Any]:
        """Validate the cleaned content quality."""
        lines = content.split('\n')
        poems = content.split('\n\n')
        
        errors = []
        warnings = []
        
        # Check for minimum content
        if len(content.strip()) < 100:
            errors.append("Content too short after cleaning")
        
        # Check for reasonable poem count
        if len(poems) < 5:
            warnings.append(f"Only {len(poems)} poems found - may be insufficient")
        
        # Check for excessively long lines (possible formatting issues)
        long_lines = [line for line in lines if len(line) > 200]
        if long_lines:
            warnings.append(f"{len(long_lines)} very long lines found - check formatting")
        
        # Check for repeated content (possible duplication)
        unique_lines = set(lines)
        if len(unique_lines) < len(lines) * 0.8:
            warnings.append("High proportion of repeated lines - check for duplicates")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'stats': {
                'total_lines': len(lines),
                'total_poems': len(poems),
                'total_characters': len(content),
                'unique_lines': len(unique_lines)
            }
        }

# Usage example
if __name__ == "__main__":
    cleaner = CorpusCleaner()
    
    input_file = Path("raw_corpus/emily_dickinson_raw.txt")
    output_file = Path("cleaned_corpus/emily_dickinson_clean.txt")
    
    cleaner.clean_text_file(input_file, output_file, "Emily Dickinson")
```

### 2. Manual Cleaning Checklist

#### Pre-Processing Checklist
- [ ] Remove copyright notices and metadata
- [ ] Remove page numbers and headers
- [ ] Check for OCR errors and typos
- [ ] Verify poem boundaries
- [ ] Ensure consistent formatting
- [ ] Remove duplicate poems
- [ ] Verify poet attribution

#### Quality Checks
- [ ] All poems are complete (no fragments)
- [ ] Line breaks are preserved correctly
- [ ] Punctuation is accurate to original
- [ ] Special characters (dashes, quotes) are consistent
- [ ] No extraneous text or annotations

## Quality Validation

### 1. Automated Validation Script

```python
# scripts/validate_corpus.py
from pathlib import Path
from src.stylometric.training_data import PoetryCorpusLoader

def validate_corpus_quality(corpus_path: Path, poet_name: str):
    """Comprehensive corpus quality validation."""
    
    loader = PoetryCorpusLoader()
    
    try:
        # Load corpus
        poems = loader.load_corpus_from_file(corpus_path, poet_name)
        
        # Run quality validation
        quality_report = loader.validate_corpus_quality(poems)
        
        print(f"\n=== Corpus Quality Report for {poet_name} ===")
        print(f"Valid: {quality_report['valid']}")
        print(f"\nMetrics:")
        for key, value in quality_report['metrics'].items():
            print(f"  {key}: {value}")
        
        if quality_report['warnings']:
            print(f"\nWarnings:")
            for warning in quality_report['warnings']:
                print(f"  ⚠️  {warning}")
        
        # Additional detailed analysis
        detailed_analysis = analyze_corpus_details(poems)
        print(f"\n=== Detailed Analysis ===")
        for key, value in detailed_analysis.items():
            print(f"  {key}: {value}")
        
        return quality_report['valid']
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

def analyze_corpus_details(poems: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Perform detailed corpus analysis."""
    
    analysis = {
        'poem_length_distribution': {},
        'vocabulary_size': 0,
        'unique_first_lines': 0,
        'average_stanza_count': 0,
        'rhyme_scheme_variety': 0
    }
    
    all_words = set()
    first_lines = set()
    total_stanzas = 0
    
    length_buckets = {'short': 0, 'medium': 0, 'long': 0, 'very_long': 0}
    
    for poem in poems:
        text = poem.get('text', '')
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Poem length distribution
        line_count = len(lines)
        if line_count <= 8:
            length_buckets['short'] += 1
        elif line_count <= 16:
            length_buckets['medium'] += 1
        elif line_count <= 32:
            length_buckets['long'] += 1
        else:
            length_buckets['very_long'] += 1
        
        # Vocabulary analysis
        words = text.lower().split()
        all_words.update(words)
        
        # First line uniqueness
        if lines:
            first_lines.add(lines[0].lower())
        
        # Stanza count
        stanzas = text.split('\n\n')
        total_stanzas += len(stanzas)
    
    analysis['poem_length_distribution'] = length_buckets
    analysis['vocabulary_size'] = len(all_words)
    analysis['unique_first_lines'] = len(first_lines)
    analysis['average_stanza_count'] = total_stanzas / len(poems) if poems else 0
    
    return analysis

# Usage
if __name__ == "__main__":
    corpus_file = Path("data/corpus/emily_dickinson.txt")
    is_valid = validate_corpus_quality(corpus_file, "Emily Dickinson")
    
    if is_valid:
        print("✅ Corpus validation passed!")
    else:
        print("❌ Corpus validation failed!")
```

### 2. Quality Metrics

#### Essential Metrics
- **Completeness**: All poems are complete works
- **Consistency**: Uniform formatting and structure
- **Authenticity**: Verified attribution to the poet
- **Diversity**: Representative sample of poet's work
- **Size**: Sufficient quantity for training

#### Quality Thresholds
```python
QUALITY_THRESHOLDS = {
    'minimum_poems': 50,
    'minimum_avg_lines': 4,
    'maximum_duplicate_ratio': 0.05,
    'minimum_vocabulary_size': 500,
    'maximum_ocr_error_rate': 0.02
}
```

## Feature Encoding

### 1. Stylometric Feature Extraction

The framework automatically extracts stylometric features during data preparation:

```python
# Example of feature encoding process
from src.stylometric.training_data import TrainingDataProcessor

processor = TrainingDataProcessor()

# Process corpus and extract features
training_data = processor.process_corpus(
    corpus_path=Path("data/corpus/emily_dickinson.txt"),
    poet_name="emily_dickinson"
)

# Features are automatically encoded for each poem and line
print("Sample features:")
for example in training_data['examples'][:2]:
    print(f"Poem: {example['poem_title']}")
    print(f"Features: {example['stylometric_features']}")
    print("---")
```

### 2. Custom Feature Encoding

You can add custom features for specific poets:

```python
# custom_features.py
from src.stylometric.training_data import StylemetricFeatureEncoder

class CustomFeatureEncoder(StylemetricFeatureEncoder):
    """Extended feature encoder with custom features."""
    
    def encode_dickinson_features(self, line: str, poem_context: str) -> Dict[str, Any]:
        """Extract Dickinson-specific features."""
        features = {}
        
        # Dash frequency (Dickinson's signature)
        features['dash_count'] = line.count('—') + line.count('-')
        features['dash_ratio'] = features['dash_count'] / len(line.split()) if line.split() else 0
        
        # Irregular capitalization
        words = line.split()
        if words:
            cap_words = [w for w in words if w and w[0].isupper()]
            features['irregular_caps'] = len(cap_words) > 1 and not words[0][0].isupper()
        
        # Slant rhyme indicators
        features['potential_slant_rhyme'] = self.detect_slant_rhyme_patterns(line)
        
        return features
    
    def detect_slant_rhyme_patterns(self, line: str) -> bool:
        """Detect patterns that might indicate slant rhyme."""
        # Simplified detection - look for near-rhyme endings
        words = line.strip().split()
        if not words:
            return False
        
        last_word = words[-1].lower().strip('.,!?;:')
        
        # Common slant rhyme patterns in Dickinson
        slant_patterns = ['ight', 'ound', 'ead', 'ome', 'ull']
        return any(last_word.endswith(pattern) for pattern in slant_patterns)

# Usage
encoder = CustomFeatureEncoder()
features = encoder.encode_dickinson_features(
    "I heard a Fly buzz — when I died —",
    "Full poem context here..."
)
```## Datas
et Formatting

### 1. Training Example Generation

The framework creates multiple types of training examples from each poem:

```python
# Example of training data generation
from src.stylometric.training_data import TrainingDataProcessor

processor = TrainingDataProcessor()

# Generate training examples
poems = [
    {
        'text': 'Hope is the thing with feathers\nThat perches in the soul...',
        'title': 'Hope is the Thing with Feathers',
        'poet': 'emily_dickinson'
    }
]

training_examples = processor.create_training_examples(poems)

# Types of examples generated:
# 1. Complete poem generation
# 2. Line completion
# 3. Style-specific generation
# 4. Structural continuation

for example in training_examples[:3]:
    print(f"Instruction: {example['instruction']}")
    print(f"Input: {example['input_text']}")
    print(f"Output: {example['output_text'][:100]}...")
    print("---")
```

### 2. Instruction Templates

The framework uses various instruction templates:

```python
INSTRUCTION_TEMPLATES = {
    'complete_generation': [
        "Write a poem in the style of {poet_name}.",
        "Compose a {poet_name}-style poem.",
        "Generate poetry that captures {poet_name}'s distinctive voice."
    ],
    
    'continuation': [
        "Continue this poem in the style of {poet_name}:",
        "Complete this {poet_name} poem:",
        "Add the next lines in {poet_name}'s style:"
    ],
    
    'style_specific': [
        "Write a {line_count}-line poem like {poet_name}.",
        "Create a poem with {poet_name}'s characteristic {feature}.",
        "Compose in {poet_name}'s style with {constraint}."
    ]
}
```

### 3. Output Format

Training examples are formatted for different model architectures:

#### For GPT-style Models
```json
{
    "instruction": "Write a poem in the style of Emily Dickinson.",
    "input": "",
    "output": "Hope is the thing with feathers\nThat perches in the soul,\nAnd sings the tune without the words,\nAnd never stops at all,",
    "metadata": {
        "poet": "emily_dickinson",
        "features": {
            "line_count": 4,
            "syllable_pattern": [8, 6, 8, 6],
            "rhyme_scheme": "ABCB"
        }
    }
}
```

#### For Instruction-Following Models
```json
{
    "messages": [
        {
            "role": "system",
            "content": "You are a poetry generator that writes in the style of specific poets."
        },
        {
            "role": "user", 
            "content": "Write a poem in the style of Emily Dickinson about hope."
        },
        {
            "role": "assistant",
            "content": "Hope is the thing with feathers\nThat perches in the soul..."
        }
    ]
}
```

## Data Augmentation

### 1. Stylistic Augmentation

```python
# augmentation.py
from typing import List, Dict, Any
import random

class PoetryDataAugmenter:
    """Augment training data with stylistic variations."""
    
    def __init__(self):
        self.augmentation_strategies = {
            'prompt_variation': self.vary_prompts,
            'length_variation': self.vary_length_constraints,
            'thematic_variation': self.vary_themes,
            'structural_variation': self.vary_structure_constraints
        }
    
    def augment_training_data(self, examples: List[Dict[str, Any]], 
                            augmentation_factor: int = 2) -> List[Dict[str, Any]]:
        """Augment training examples with variations."""
        
        augmented = []
        
        for example in examples:
            # Keep original
            augmented.append(example)
            
            # Generate variations
            for _ in range(augmentation_factor):
                strategy = random.choice(list(self.augmentation_strategies.keys()))
                augmented_example = self.augmentation_strategies[strategy](example)
                augmented.append(augmented_example)
        
        return augmented
    
    def vary_prompts(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Create prompt variations."""
        poet_name = example['poet_name']
        
        prompt_variations = [
            f"Compose a poem in {poet_name}'s distinctive style.",
            f"Write poetry that captures the essence of {poet_name}.",
            f"Create a {poet_name}-inspired poem.",
            f"Generate verse in the manner of {poet_name}."
        ]
        
        new_example = example.copy()
        new_example['instruction'] = random.choice(prompt_variations)
        return new_example
    
    def vary_length_constraints(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Add length constraints to prompts."""
        poet_name = example['poet_name']
        output_lines = len(example['output_text'].split('\n'))
        
        length_prompts = [
            f"Write a {output_lines}-line poem in the style of {poet_name}.",
            f"Compose a short poem like {poet_name} with exactly {output_lines} lines.",
            f"Create a brief {poet_name}-style verse of {output_lines} lines."
        ]
        
        new_example = example.copy()
        new_example['instruction'] = random.choice(length_prompts)
        return new_example
    
    def vary_themes(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Add thematic constraints."""
        poet_name = example['poet_name']
        
        # Extract potential themes from the poem
        themes = self.extract_themes(example['output_text'])
        
        if themes:
            theme = random.choice(themes)
            thematic_prompts = [
                f"Write a {poet_name} poem about {theme}.",
                f"Compose in {poet_name}'s style on the theme of {theme}.",
                f"Create a {poet_name}-inspired poem exploring {theme}."
            ]
            
            new_example = example.copy()
            new_example['instruction'] = random.choice(thematic_prompts)
            return new_example
        
        return example
    
    def vary_structure_constraints(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Add structural constraints."""
        poet_name = example['poet_name']
        
        # Analyze structure
        structure = self.analyze_structure(example['output_text'])
        
        structural_prompts = [
            f"Write a {poet_name} poem with {structure['stanza_count']} stanzas.",
            f"Compose in {poet_name}'s style using {structure['rhyme_scheme']} rhyme scheme.",
            f"Create a {poet_name} poem in {structure['meter']} meter."
        ]
        
        new_example = example.copy()
        new_example['instruction'] = random.choice(structural_prompts)
        return new_example
    
    def extract_themes(self, poem_text: str) -> List[str]:
        """Extract potential themes from poem text."""
        # Simplified theme extraction
        theme_keywords = {
            'nature': ['bird', 'tree', 'flower', 'sky', 'wind', 'sun', 'moon'],
            'death': ['death', 'grave', 'dying', 'mortal', 'eternal'],
            'love': ['love', 'heart', 'beloved', 'passion', 'romance'],
            'hope': ['hope', 'faith', 'trust', 'believe', 'dream'],
            'time': ['time', 'moment', 'hour', 'day', 'year', 'forever']
        }
        
        poem_lower = poem_text.lower()
        found_themes = []
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in poem_lower for keyword in keywords):
                found_themes.append(theme)
        
        return found_themes
    
    def analyze_structure(self, poem_text: str) -> Dict[str, Any]:
        """Analyze poem structure for constraints."""
        lines = [line.strip() for line in poem_text.split('\n') if line.strip()]
        stanzas = poem_text.split('\n\n')
        
        return {
            'line_count': len(lines),
            'stanza_count': len(stanzas),
            'rhyme_scheme': 'ABAB',  # Simplified
            'meter': 'iambic'  # Simplified
        }

# Usage
augmenter = PoetryDataAugmenter()
augmented_data = augmenter.augment_training_data(training_examples, augmentation_factor=3)
```

### 2. Contextual Augmentation

```python
class ContextualAugmenter:
    """Add contextual information to training examples."""
    
    def add_historical_context(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Add historical context to prompts."""
        poet_name = example['poet_name']
        
        historical_contexts = {
            'emily_dickinson': [
                "Write a poem in Emily Dickinson's 19th-century New England style.",
                "Compose like Emily Dickinson from her reclusive Amherst period.",
                "Create a poem in Dickinson's Civil War era voice."
            ],
            'walt_whitman': [
                "Write in Walt Whitman's expansive American democratic style.",
                "Compose like Whitman celebrating the common person.",
                "Create a poem in Whitman's Civil War nurse perspective."
            ]
        }
        
        if poet_name in historical_contexts:
            new_example = example.copy()
            new_example['instruction'] = random.choice(historical_contexts[poet_name])
            return new_example
        
        return example
    
    def add_stylistic_context(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Add specific stylistic instructions."""
        poet_name = example['poet_name']
        
        stylistic_contexts = {
            'emily_dickinson': [
                "Write with Dickinson's characteristic dashes and slant rhymes.",
                "Use Dickinson's hymn meter and irregular capitalization.",
                "Employ Dickinson's compressed, enigmatic style."
            ],
            'edgar_allan_poe': [
                "Write with Poe's dark, atmospheric imagery.",
                "Use Poe's internal rhyme and repetitive refrains.",
                "Employ Poe's Gothic, melancholic tone."
            ]
        }
        
        if poet_name in stylistic_contexts:
            new_example = example.copy()
            new_example['instruction'] = random.choice(stylistic_contexts[poet_name])
            return new_example
        
        return example
```

## Best Practices

### 1. Corpus Curation Guidelines

#### Quality Over Quantity
- **Prefer authentic, complete poems** over fragments
- **Verify attributions** using scholarly sources
- **Include diverse examples** from different periods
- **Maintain consistent formatting** throughout

#### Balanced Representation
```python
CORPUS_BALANCE_GUIDELINES = {
    'early_period': 0.3,    # 30% early works
    'middle_period': 0.4,   # 40% mature works  
    'late_period': 0.3,     # 30% late works
    
    'short_poems': 0.4,     # 40% short poems (< 16 lines)
    'medium_poems': 0.4,    # 40% medium poems (16-32 lines)
    'long_poems': 0.2,      # 20% long poems (> 32 lines)
    
    'major_themes': 0.6,    # 60% major themes
    'minor_themes': 0.4     # 40% minor themes
}
```

### 2. Data Processing Pipeline

```python
# Complete data processing pipeline
def process_poet_corpus(poet_name: str, corpus_path: Path) -> Dict[str, Any]:
    """Complete pipeline for processing poet corpus."""
    
    print(f"Processing corpus for {poet_name}...")
    
    # Step 1: Load and clean corpus
    loader = PoetryCorpusLoader()
    poems = loader.load_corpus_from_file(corpus_path, poet_name)
    print(f"✓ Loaded {len(poems)} poems")
    
    # Step 2: Validate quality
    quality_report = loader.validate_corpus_quality(poems)
    if not quality_report['valid']:
        raise ValueError(f"Corpus quality validation failed: {quality_report['warnings']}")
    print("✓ Quality validation passed")
    
    # Step 3: Create training examples
    processor = TrainingDataProcessor()
    training_examples = processor.create_training_examples(poems)
    print(f"✓ Generated {len(training_examples)} training examples")
    
    # Step 4: Augment data
    augmenter = PoetryDataAugmenter()
    augmented_examples = augmenter.augment_training_data(training_examples, augmentation_factor=2)
    print(f"✓ Augmented to {len(augmented_examples)} examples")
    
    # Step 5: Split dataset
    train_data, val_data = processor.split_dataset(augmented_examples, train_ratio=0.8)
    print(f"✓ Split into {len(train_data)} train / {len(val_data)} validation examples")
    
    # Step 6: Save processed data
    output_dir = Path(f"data/processed/{poet_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processor.save_training_data(train_data, output_dir / "train.json")
    processor.save_training_data(val_data, output_dir / "validation.json")
    
    print(f"✓ Saved processed data to {output_dir}")
    
    return {
        'train_examples': len(train_data),
        'val_examples': len(val_data),
        'total_poems': len(poems),
        'quality_metrics': quality_report['metrics']
    }

# Usage
result = process_poet_corpus("emily_dickinson", Path("data/raw/emily_dickinson.txt"))
print(f"Processing complete: {result}")
```

### 3. Quality Assurance Checklist

#### Pre-Training Validation
- [ ] Corpus size meets minimum requirements
- [ ] All poems are complete and properly formatted
- [ ] No duplicate or near-duplicate poems
- [ ] Consistent poet attribution
- [ ] Balanced representation across themes/periods
- [ ] Training examples cover diverse instruction types
- [ ] Validation set is representative

#### Post-Processing Verification
- [ ] Training data format is correct
- [ ] Features are properly encoded
- [ ] No data leakage between train/validation sets
- [ ] Augmented examples maintain quality
- [ ] File formats are compatible with training pipeline

## Troubleshooting

### Common Issues and Solutions

#### 1. Encoding Problems
```python
# Fix encoding issues
def fix_encoding_issues(file_path: Path) -> str:
    """Try multiple encodings to read file."""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            print(f"Successfully read with {encoding}")
            return content
        except UnicodeDecodeError:
            continue
    
    raise ValueError("Could not decode file with any supported encoding")
```

#### 2. Parsing Failures
```python
# Debug parsing issues
def debug_parsing_failure(content: str, poet_name: str):
    """Debug why corpus parsing failed."""
    
    print(f"Content length: {len(content)}")
    print(f"First 200 characters: {content[:200]}")
    print(f"Last 200 characters: {content[-200:]}")
    
    # Check for common issues
    if len(content) < 100:
        print("❌ Content too short")
    
    if not re.search(r'\n\s*\n', content):
        print("❌ No poem separators found")
    
    if content.count('\n') < 10:
        print("❌ Very few line breaks")
    
    # Try different splitting strategies
    poems_double_newline = content.split('\n\n')
    poems_triple_newline = content.split('\n\n\n')
    
    print(f"Poems with \\n\\n split: {len(poems_double_newline)}")
    print(f"Poems with \\n\\n\\n split: {len(poems_triple_newline)}")
```

#### 3. Quality Validation Failures
```python
# Address quality issues
def fix_quality_issues(poems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fix common quality issues in parsed poems."""
    
    fixed_poems = []
    
    for poem in poems:
        text = poem.get('text', '').strip()
        
        # Skip empty poems
        if not text:
            continue
        
        # Fix line break issues
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if line]  # Remove empty lines
        
        # Skip very short poems
        if len(lines) < 2:
            continue
        
        # Reconstruct poem
        fixed_text = '\n'.join(lines)
        
        fixed_poem = poem.copy()
        fixed_poem['text'] = fixed_text
        fixed_poems.append(fixed_poem)
    
    return fixed_poems
```

#### 4. Memory Issues with Large Corpora
```python
# Process large corpora in chunks
def process_large_corpus(corpus_path: Path, poet_name: str, chunk_size: int = 100):
    """Process large corpus in manageable chunks."""
    
    loader = PoetryCorpusLoader()
    poems = loader.load_corpus_from_file(corpus_path, poet_name)
    
    all_examples = []
    
    # Process in chunks
    for i in range(0, len(poems), chunk_size):
        chunk = poems[i:i + chunk_size]
        print(f"Processing chunk {i//chunk_size + 1}/{(len(poems) + chunk_size - 1)//chunk_size}")
        
        processor = TrainingDataProcessor()
        chunk_examples = processor.create_training_examples(chunk)
        all_examples.extend(chunk_examples)
        
        # Clear memory
        del chunk_examples
        import gc
        gc.collect()
    
    return all_examples
```

### Performance Optimization

#### 1. Parallel Processing
```python
# Process multiple poets in parallel
from multiprocessing import Pool
from functools import partial

def process_multiple_poets(poet_configs: List[Dict[str, Any]], num_processes: int = 4):
    """Process multiple poets in parallel."""
    
    process_func = partial(process_single_poet_config)
    
    with Pool(num_processes) as pool:
        results = pool.map(process_func, poet_configs)
    
    return results

def process_single_poet_config(config: Dict[str, Any]):
    """Process a single poet configuration."""
    return process_poet_corpus(config['name'], Path(config['corpus_path']))

# Usage
poet_configs = [
    {'name': 'emily_dickinson', 'corpus_path': 'data/raw/dickinson.txt'},
    {'name': 'walt_whitman', 'corpus_path': 'data/raw/whitman.txt'},
    {'name': 'edgar_allan_poe', 'corpus_path': 'data/raw/poe.txt'}
]

results = process_multiple_poets(poet_configs)
```

#### 2. Caching Intermediate Results
```python
# Cache processed data
import pickle
from pathlib import Path

def cache_processed_data(data: Any, cache_path: Path):
    """Cache processed data for reuse."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)

def load_cached_data(cache_path: Path) -> Any:
    """Load cached processed data."""
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    return None

# Usage in processing pipeline
def process_with_cache(poet_name: str, corpus_path: Path):
    """Process with caching for efficiency."""
    
    cache_path = Path(f"cache/{poet_name}_processed.pkl")
    
    # Try to load from cache
    cached_data = load_cached_data(cache_path)
    if cached_data:
        print(f"✓ Loaded cached data for {poet_name}")
        return cached_data
    
    # Process and cache
    result = process_poet_corpus(poet_name, corpus_path)
    cache_processed_data(result, cache_path)
    
    return result
```

This comprehensive guide provides everything needed to prepare high-quality training data for the Stylistic Poetry LLM Framework, from initial corpus collection through final dataset formatting and validation.