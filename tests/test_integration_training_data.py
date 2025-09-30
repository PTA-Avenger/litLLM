"""Integration test for the complete training data processing pipeline."""

import tempfile
import json
from pathlib import Path

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.stylometric.training_data import (
    PoetryCorpusLoader, TrainingDatasetFormatter
)


def test_complete_training_data_pipeline():
    """Test the complete pipeline from corpus loading to training data generation."""
    
    # Create a sample corpus file
    sample_corpus = """Emily's First Poem
I dwell in Possibility—
A fairer House than Prose—
More numerous of Windows—
Superior—for Doors—


Another Dickinson Poem
Because I could not stop for Death—
He kindly stopped for me—
The Carriage held but just Ourselves—
And Immortality."""
    
    # Create temporary corpus file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(sample_corpus)
        corpus_path = f.name
    
    try:
        # Step 1: Load corpus
        loader = PoetryCorpusLoader()
        poems = loader.load_corpus_from_file(corpus_path, "Emily Dickinson")
        
        # Verify corpus loading
        assert len(poems) == 2
        assert poems[0]['poet'] == "Emily Dickinson"
        assert "I dwell in Possibility" in poems[0]['text']
        
        # Step 2: Validate corpus quality
        quality_report = loader.validate_corpus_quality(poems)
        assert quality_report['valid']
        assert quality_report['metrics']['total_poems'] == 2
        
        # Step 3: Create training examples
        formatter = TrainingDatasetFormatter()
        training_examples = formatter.create_instruction_output_pairs(poems)
        
        # Verify training examples were created
        assert len(training_examples) > 0
        
        # Check that examples have the right structure
        example = training_examples[0]
        assert example.poet_name == "Emily Dickinson"
        assert "Emily Dickinson" in example.instruction
        assert len(example.output_text) > 0
        assert isinstance(example.stylometric_features, dict)
        
        # Step 4: Format for HuggingFace
        hf_formatted = formatter.format_for_huggingface(training_examples)
        
        # Verify HuggingFace formatting
        assert len(hf_formatted) == len(training_examples)
        assert 'prompt' in hf_formatted[0]
        assert 'completion' in hf_formatted[0]
        assert hf_formatted[0]['poet_name'] == "Emily Dickinson"
        
        # Step 5: Save training data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            output_path = f.name
        
        formatter.save_training_data(training_examples, output_path, "jsonl")
        
        # Verify saved data
        assert Path(output_path).exists()
        
        # Read and verify the saved data
        with open(output_path, 'r') as f:
            lines = f.readlines()
            assert len(lines) > 0
            
            # Parse first line to verify structure
            first_example = json.loads(lines[0])
            assert 'instruction' in first_example
            assert 'output_text' in first_example
            assert first_example['poet_name'] == "Emily Dickinson"
        
        print(f"✓ Successfully processed {len(poems)} poems into {len(training_examples)} training examples")
        print(f"✓ Saved training data to {output_path}")
        
        # Clean up output file
        Path(output_path).unlink()
        
    finally:
        # Clean up corpus file
        Path(corpus_path).unlink()


if __name__ == "__main__":
    test_complete_training_data_pipeline()
    print("Integration test passed!")