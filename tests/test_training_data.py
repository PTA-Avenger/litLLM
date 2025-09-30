"""Unit tests for training data preparation system."""

import json
import tempfile
from pathlib import Path
import pytest

from src.stylometric.training_data import (
    TrainingExample, PoetryCorpusLoader, StylemetricFeatureEncoder, 
    TrainingDatasetFormatter
)
from src.stylometric.poet_profile import PoetProfile


class TestTrainingExample:
    """Test TrainingExample data model."""
    
    def test_training_example_creation(self):
        """Test creating a TrainingExample."""
        example = TrainingExample(
            instruction="Write a poem",
            input_text="",
            output_text="Roses are red\nViolets are blue",
            stylometric_features={"lines": 2},
            poet_name="Test Poet"
        )
        
        assert example.instruction == "Write a poem"
        assert example.output_text == "Roses are red\nViolets are blue"
        assert example.stylometric_features == {"lines": 2}
        assert example.poet_name == "Test Poet"
    
    def test_training_example_serialization(self):
        """Test TrainingExample serialization."""
        example = TrainingExample(
            instruction="Write a poem",
            input_text="",
            output_text="Test poem",
            stylometric_features={"lines": 1},
            poet_name="Test Poet"
        )
        
        # Test to_dict
        data = example.to_dict()
        assert isinstance(data, dict)
        assert data['instruction'] == "Write a poem"
        assert data['poet_name'] == "Test Poet"
        
        # Test from_dict
        reconstructed = TrainingExample.from_dict(data)
        assert reconstructed.instruction == example.instruction
        assert reconstructed.poet_name == example.poet_name


class TestPoetryCorpusLoader:
    """Test PoetryCorpusLoader functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.loader = PoetryCorpusLoader()
    
    def test_detect_format(self):
        """Test format detection."""
        # Test JSON format
        json_content = '{"poems": ["test poem"]}'
        assert self.loader._detect_format(json_content) == "json"
        
        # Test Gutenberg format
        gutenberg_content = "*** START OF PROJECT *** content *** END OF PROJECT ***"
        assert self.loader._detect_format(gutenberg_content) == "gutenberg"
        
        # Test simple format
        simple_content = "This is a simple poem"
        assert self.loader._detect_format(simple_content) == "simple"
    
    def test_parse_simple_format(self):
        """Test parsing simple text format."""
        content = """Poem Title 1
Line 1 of poem 1
Line 2 of poem 1


Poem Title 2
Line 1 of poem 2
Line 2 of poem 2"""
        
        poems = self.loader._parse_simple_format(content, "Test Poet")
        
        assert len(poems) == 2
        assert poems[0]['title'] == "Poem Title 1"
        assert poems[0]['poet'] == "Test Poet"
        assert "Line 1 of poem 1" in poems[0]['text']
        assert poems[1]['title'] == "Poem Title 2"
    
    def test_parse_json_format(self):
        """Test parsing JSON format."""
        content = json.dumps({
            "poems": [
                {"title": "Test Poem", "text": "Line 1\nLine 2"},
                "Simple poem text"
            ]
        })
        
        poems = self.loader._parse_json_format(content, "Test Poet")
        
        assert len(poems) == 2
        assert poems[0]['title'] == "Test Poem"
        assert poems[0]['text'] == "Line 1\nLine 2"
        assert poems[1]['title'] == "Poem 2"
        assert poems[1]['text'] == "Simple poem text"
    
    def test_load_corpus_from_file(self):
        """Test loading corpus from file."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test Poem A\nLine 1\nLine 2\n\n\nAnother Poem B\nLine 1\nLine 2")
            temp_path = f.name
        
        try:
            poems = self.loader.load_corpus_from_file(temp_path, "Test Poet")
            assert len(poems) >= 1
            assert poems[0]['poet'] == "Test Poet"
        finally:
            Path(temp_path).unlink()
    
    def test_validate_corpus_quality(self):
        """Test corpus quality validation."""
        # Test empty corpus
        result = self.loader.validate_corpus_quality([])
        assert not result['valid']
        assert 'No poems found' in result['warnings'][0]
        
        # Test corpus with quality issues
        poor_poems = [
            {'text': '', 'poet': 'Test'},  # Empty poem
            {'text': 'Short', 'poet': 'Test'},  # Very short poem
            {'text': 'Line 1\nLine 2\nLine 3\nLine 4\nWith multiple lines', 'poet': 'Test'}
        ]
        
        result = self.loader.validate_corpus_quality(poor_poems)
        assert len(result['warnings']) > 0
        assert result['metrics']['total_poems'] == 3
        assert result['metrics']['empty_poems'] == 1


class TestStylemetricFeatureEncoder:
    """Test StylemetricFeatureEncoder functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.encoder = StylemetricFeatureEncoder()
    
    def test_encode_line_features(self):
        """Test encoding features for a single line."""
        line = "The quick brown fox jumps over the lazy dog"
        poem_context = "The quick brown fox jumps over the lazy dog\nThis is line two"
        
        features = self.encoder.encode_line_features(line, poem_context, 0)
        
        assert 'line_length' in features
        assert 'syllable_count' in features
        assert 'avg_word_length' in features
        assert features['line_length'] == 9  # 9 words
        assert features['syllable_count'] > 0
    
    def test_encode_poem_features(self):
        """Test encoding features for entire poem."""
        poem_text = """Line one of the poem
Line two with more words
Line three continues
Final line here"""
        
        features = self.encoder.encode_poem_features(poem_text)
        
        assert 'total_lines' in features
        assert 'total_stanzas' in features
        assert 'poem_ttr' in features
        assert features['total_lines'] == 4
        assert features['total_stanzas'] >= 1


class TestTrainingDatasetFormatter:
    """Test TrainingDatasetFormatter functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = TrainingDatasetFormatter()
    
    def test_create_instruction_output_pairs(self):
        """Test creating instruction-output pairs."""
        poems = [
            {
                'text': 'Line one\nLine two\nLine three',
                'poet': 'Test Poet',
                'title': 'Test Poem'
            }
        ]
        
        examples = self.formatter.create_instruction_output_pairs(poems)
        
        assert len(examples) > 0
        assert all(isinstance(ex, TrainingExample) for ex in examples)
        assert examples[0].poet_name == 'Test Poet'
        assert 'Test Poet' in examples[0].instruction
    
    def test_format_for_huggingface(self):
        """Test formatting for HuggingFace."""
        examples = [
            TrainingExample(
                instruction="Write a poem",
                input_text="",
                output_text="Test poem",
                stylometric_features={"lines": 1},
                poet_name="Test Poet"
            )
        ]
        
        # Test default instruction_following format
        formatted = self.formatter.format_for_huggingface(examples)
        
        assert len(formatted) == 1
        assert 'instruction' in formatted[0]
        assert 'output' in formatted[0]
        assert formatted[0]['poet_name'] == 'Test Poet'
        
        # Test completion format specifically
        completion_formatted = self.formatter.format_for_huggingface(examples, 'completion')
        assert 'prompt' in completion_formatted[0]
        assert 'completion' in completion_formatted[0]
    
    def test_save_training_data(self):
        """Test saving training data."""
        examples = [
            TrainingExample(
                instruction="Write a poem",
                input_text="",
                output_text="Test poem",
                stylometric_features={"lines": 1},
                poet_name="Test Poet"
            )
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_path = f.name
        
        try:
            self.formatter.save_training_data(examples, temp_path, "jsonl")
            
            # Verify file was created and has content
            assert Path(temp_path).exists()
            with open(temp_path, 'r') as f:
                content = f.read()
                assert len(content) > 0
                assert 'Test Poet' in content
        finally:
            Path(temp_path).unlink()


class TestDataProcessingPipeline:
    """Integration tests for complete data processing pipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.loader = PoetryCorpusLoader()
        self.formatter = TrainingDatasetFormatter()
    
    def test_complete_pipeline_integration(self):
        """Test complete pipeline from corpus loading to formatted output."""
        # Create sample corpus data
        sample_poems = [
            {
                'text': 'Line one of first poem\nLine two continues\nLine three ends here',
                'poet': 'Test Poet',
                'title': 'First Poem'
            },
            {
                'text': 'Second poem begins\nWith different structure\nAnd varied content\nEnding with fourth line',
                'poet': 'Test Poet', 
                'title': 'Second Poem'
            }
        ]
        
        # Test corpus quality validation
        quality_result = self.loader.validate_corpus_quality(sample_poems)
        assert quality_result['valid']
        assert quality_result['metrics']['total_poems'] == 2
        
        # Test training example creation
        training_examples = self.formatter.create_instruction_output_pairs(sample_poems)
        assert len(training_examples) > 0
        assert all(isinstance(ex, TrainingExample) for ex in training_examples)
        
        # Test data augmentation
        augmented_examples = self.formatter.apply_data_augmentation(training_examples, 1.5)
        assert len(augmented_examples) >= len(training_examples)
        
        # Test HuggingFace formatting
        hf_formatted = self.formatter.format_for_huggingface(augmented_examples)
        assert len(hf_formatted) == len(augmented_examples)
        assert all('instruction' in ex for ex in hf_formatted)
        
        # Test dataset splits
        splits = self.formatter.create_dataset_splits(augmented_examples)
        assert 'train' in splits and 'val' in splits and 'test' in splits
        total_split_size = len(splits['train']) + len(splits['val']) + len(splits['test'])
        assert total_split_size == len(augmented_examples)
    
    def test_data_augmentation_strategies(self):
        """Test different data augmentation strategies."""
        base_example = TrainingExample(
            instruction="Write a poem",
            input_text="",
            output_text="Sample poem\nWith two lines",
            stylometric_features={
                'total_lines': 2,
                'rhyme_scheme': 'AA',
                'dominant_meter': 'iambic'
            },
            poet_name="Test Poet"
        )
        
        # Test style variation
        style_aug = self.formatter._augment_style_variation(base_example)
        assert style_aug is not None
        assert style_aug.poet_name == base_example.poet_name
        assert style_aug.output_text == base_example.output_text
        assert 'Test Poet' in style_aug.instruction
        
        # Test length variation
        length_aug = self.formatter._augment_length_variation(base_example)
        assert length_aug is not None
        assert 'short' in length_aug.instruction.lower()
        
        # Test structural variation
        struct_aug = self.formatter._augment_structural_variation(base_example)
        assert struct_aug is not None
        assert ('AA' in struct_aug.instruction or 'iambic' in struct_aug.instruction)
    
    def test_dataset_balancing(self):
        """Test dataset balancing functionality."""
        # Create unbalanced dataset
        examples = []
        for i in range(10):
            poet = "Poet A" if i < 7 else "Poet B"
            examples.append(TrainingExample(
                instruction=f"Write poem {i}",
                input_text="",
                output_text=f"Poem {i}",
                stylometric_features={'total_lines': i % 5 + 1},
                poet_name=poet
            ))
        
        # Test balancing by poet
        balanced = self.formatter.create_balanced_dataset(examples, 'poet_name')
        poet_counts = {}
        for ex in balanced:
            poet_counts[ex.poet_name] = poet_counts.get(ex.poet_name, 0) + 1
        
        # Should have equal representation
        assert len(set(poet_counts.values())) <= 1  # All counts should be equal
    
    def test_multiple_format_styles(self):
        """Test different HuggingFace format styles."""
        example = TrainingExample(
            instruction="Write a poem",
            input_text="Starting line",
            output_text="Complete poem",
            stylometric_features={},
            poet_name="Test Poet"
        )
        
        # Test instruction following format
        inst_format = self.formatter.format_for_huggingface([example], 'instruction_following')
        assert 'instruction' in inst_format[0]
        assert 'input' in inst_format[0]
        assert 'output' in inst_format[0]
        
        # Test chat format
        chat_format = self.formatter.format_for_huggingface([example], 'chat')
        assert 'messages' in chat_format[0]
        assert isinstance(chat_format[0]['messages'], list)
        
        # Test completion format
        comp_format = self.formatter.format_for_huggingface([example], 'completion')
        assert 'prompt' in comp_format[0]
        assert 'completion' in comp_format[0]
    
    def test_error_handling_and_edge_cases(self):
        """Test error handling and edge cases."""
        # Test empty dataset
        empty_examples = []
        augmented = self.formatter.apply_data_augmentation(empty_examples)
        assert len(augmented) == 0
        
        # Test invalid split ratios
        with pytest.raises(ValueError):
            self.formatter.create_dataset_splits([], 0.5, 0.5, 0.5)  # Sum > 1.0
        
        # Test unknown format style
        example = TrainingExample(
            instruction="Test",
            input_text="",
            output_text="Test",
            stylometric_features={},
            poet_name="Test"
        )
        
        with pytest.raises(ValueError):
            self.formatter.format_for_huggingface([example], 'unknown_format')
    
    def test_serialization_and_file_operations(self):
        """Test serialization and file I/O operations."""
        example = TrainingExample(
            instruction="Write a poem",
            input_text="",
            output_text="Test poem content",
            stylometric_features={'lines': 1},
            poet_name="Test Poet"
        )
        
        # Test serialization
        data_dict = example.to_dict()
        assert isinstance(data_dict, dict)
        assert data_dict['poet_name'] == "Test Poet"
        
        # Test deserialization
        reconstructed = TrainingExample.from_dict(data_dict)
        assert reconstructed.poet_name == example.poet_name
        assert reconstructed.output_text == example.output_text
        
        # Test file saving with different formats
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_path = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            jsonl_path = f.name
        
        try:
            # Test JSON format
            self.formatter.save_training_data([example], json_path, "json")
            assert Path(json_path).exists()
            
            # Test JSONL format
            self.formatter.save_training_data([example], jsonl_path, "jsonl")
            assert Path(jsonl_path).exists()
            
            # Verify content
            with open(jsonl_path, 'r') as f:
                content = f.read()
                assert 'Test Poet' in content
                
        finally:
            Path(json_path).unlink(missing_ok=True)
            Path(jsonl_path).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__])