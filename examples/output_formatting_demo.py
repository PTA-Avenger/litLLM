#!/usr/bin/env python3
"""
Demonstration of enhanced output formatting and analysis display functionality.

This script showcases the new output formatting capabilities including:
- Enhanced poem display with decorative formatting
- Comprehensive stylistic analysis display
- Comparison analysis with visual progress bars
- Multiple output format options (simple, enhanced, detailed)
- Structured saving with metadata
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from stylometric.output_formatter import PoetryOutputFormatter, format_poetry_output
from stylometric.evaluation_comparison import ComparisonResult
import tempfile
import json


def demo_poem_formatting():
    """Demonstrate poem display formatting."""
    print("=" * 80)
    print("DEMO: Enhanced Poem Display Formatting")
    print("=" * 80)
    
    formatter = PoetryOutputFormatter()
    
    sample_poem = """Hope is the thing with feathers
That perches in the soul,
And sings the tune without the words,
And never stops at all."""
    
    # Basic formatting
    formatted = formatter.format_poem_display(sample_poem, "Emily Dickinson Style")
    print(formatted)
    print()


def demo_analysis_formatting():
    """Demonstrate stylistic analysis formatting."""
    print("=" * 80)
    print("DEMO: Stylistic Analysis Display")
    print("=" * 80)
    
    formatter = PoetryOutputFormatter()
    
    sample_analysis = {
        'word_count': 20,
        'line_count': 4,
        'stanza_count': 1,
        'avg_words_per_line': 5.0,
        'ttr': 0.85,
        'lexical_density': 0.75,
        'avg_word_length': 4.2,
        'syllable_count': 28,
        'avg_syllables_per_line': 7.0,
        'rhyme_scheme': 'ABAB',
        'meter_pattern': 'Common Meter'
    }
    
    # Detailed analysis formatting
    formatted = formatter.format_stylistic_analysis(sample_analysis, detailed=True)
    print(formatted)
    print()


def demo_comparison_formatting():
    """Demonstrate comparison analysis formatting."""
    print("=" * 80)
    print("DEMO: Style Comparison Analysis")
    print("=" * 80)
    
    formatter = PoetryOutputFormatter()
    
    # Create mock comparison result
    class MockComparisonResult:
        def __init__(self):
            self.overall_comparison_score = 0.78
            self.similarity_scores = {
                'lexical_similarity': 0.82,
                'structural_similarity': 0.75,
                'readability_similarity': 0.77,
                'overall_similarity': 0.78
            }
            self.metric_differences = {
                'ttr': 0.05,
                'lexical_density': -0.03,
                'avg_syllables_per_line': 1.2,
                'word_count': -5,
                'line_count': 0
            }
    
    mock_result = MockComparisonResult()
    formatted = formatter.format_comparison_analysis(mock_result)
    print(formatted)
    print()


def demo_comprehensive_output():
    """Demonstrate comprehensive output creation."""
    print("=" * 80)
    print("DEMO: Comprehensive Output Generation")
    print("=" * 80)
    
    sample_poem = """I dwell in Possibility—
A fairer House than Prose—
More numerous of Windows—
Superior—for Doors—"""
    
    sample_analysis = {
        'word_count': 16,
        'line_count': 4,
        'stanza_count': 1,
        'avg_words_per_line': 4.0,
        'ttr': 0.94,
        'lexical_density': 0.81,
        'syllable_count': 20,
        'avg_syllables_per_line': 5.0
    }
    
    sample_metadata = {
        'model_name': 'gpt2-medium',
        'temperature': 0.9,
        'max_length': 150,
        'generation_time_ms': 2300,
        'tokens_generated': 25
    }
    
    # Create comprehensive output
    output = format_poetry_output(
        poem_text=sample_poem,
        analysis_results=sample_analysis,
        generation_metadata=sample_metadata,
        poet_style="emily_dickinson",
        prompt="dwelling in possibility"
    )
    
    print(output)
    print()


def demo_saving_functionality():
    """Demonstrate saving results with different formats."""
    print("=" * 80)
    print("DEMO: Saving Results with Enhanced Formatting")
    print("=" * 80)
    
    formatter = PoetryOutputFormatter()
    
    sample_poem = """Because I could not stop for Death—
He kindly stopped for me—
The Carriage held but just Ourselves—
And Immortality."""
    
    sample_analysis = {
        'word_count': 18,
        'line_count': 4,
        'ttr': 0.89,
        'lexical_density': 0.72
    }
    
    # Save as JSON
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json_path = Path(f.name)
    
    formatter.save_results_with_formatting(
        output_path=json_path,
        poem_text=sample_poem,
        analysis_results=sample_analysis,
        generation_config={'temperature': 0.8, 'max_length': 200},
        generation_metadata={'model': 'gpt2', 'generation_time_ms': 1800},
        prompt="death and immortality",
        poet_style="emily_dickinson"
    )
    
    print(f"JSON saved to: {json_path}")
    
    # Display JSON content
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("JSON structure:")
    print(f"  - metadata: {list(data['metadata'].keys())}")
    print(f"  - results: {list(data['results'].keys())}")
    print(f"  - prompt: {data['metadata']['prompt']}")
    print(f"  - poet_style: {data['metadata']['poet_style']}")
    
    # Save as formatted text
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        txt_path = Path(f.name)
    
    formatter.save_results_with_formatting(
        output_path=txt_path,
        poem_text=sample_poem,
        analysis_results=sample_analysis,
        generation_config={'temperature': 0.8, 'max_length': 200},
        generation_metadata={'model': 'gpt2', 'generation_time_ms': 1800},
        prompt="death and immortality",
        poet_style="emily_dickinson"
    )
    
    print(f"Formatted text saved to: {txt_path}")
    
    # Clean up
    json_path.unlink()
    txt_path.unlink()
    print("Temporary files cleaned up")
    print()


def demo_progress_bars():
    """Demonstrate visual progress bars for similarity scores."""
    print("=" * 80)
    print("DEMO: Visual Progress Bars")
    print("=" * 80)
    
    formatter = PoetryOutputFormatter()
    
    print("Similarity Score Visualizations:")
    print()
    
    scores = [0.95, 0.78, 0.62, 0.45, 0.23, 0.0]
    labels = ["Excellent", "Good", "Fair", "Poor", "Very Poor", "No Match"]
    
    for score, label in zip(scores, labels):
        bar = formatter._create_progress_bar(score, width=30)
        print(f"{label:<12} {score:.2f} {bar}")
    
    print()


def main():
    """Run all demonstrations."""
    print("STYLISTIC POETRY LLM - OUTPUT FORMATTING DEMONSTRATIONS")
    print("=" * 80)
    print()
    
    demo_poem_formatting()
    demo_analysis_formatting()
    demo_comparison_formatting()
    demo_comprehensive_output()
    demo_saving_functionality()
    demo_progress_bars()
    
    print("=" * 80)
    print("All demonstrations completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()