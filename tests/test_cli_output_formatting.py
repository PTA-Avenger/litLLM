"""
Integration tests for CLI output formatting and analysis display.

Tests the complete user interface functionality including formatted output,
stylistic analysis display, and result saving with evaluation metrics.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from click.testing import CliRunner

from src.cli import cli
from src.stylometric.output_formatter import PoetryOutputFormatter, format_poetry_output
from src.stylometric.evaluation_comparison import ComparisonResult


class TestCLIOutputFormatting:
    """Test CLI output formatting functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.sample_poem = """The quiet forest whispers low,
Of secrets that the old trees know,
While shadows dance on mossy ground,
And peace in nature can be found."""
        
        self.sample_analysis = {
            'word_count': 24,
            'line_count': 4,
            'stanza_count': 1,
            'avg_words_per_line': 6.0,
            'ttr': 0.875,
            'lexical_density': 0.667,
            'syllable_count': 32,
            'avg_syllables_per_line': 8.0
        }
        
        self.sample_metadata = {
            'model_name': 'gpt2',
            'temperature': 0.8,
            'max_length': 200,
            'generation_time_ms': 1500,
            'tokens_generated': 30
        }
    
    @patch('src.cli.create_poetry_model')
    @patch('src.cli.QuantitativeEvaluator')
    def test_enhanced_output_formatting(self, mock_evaluator, mock_model):
        """Test enhanced output formatting in CLI."""
        # Mock model and evaluator
        mock_model_instance = Mock()
        mock_model_instance.load_model.return_value = True
        mock_model_instance.generate_poetry.return_value = Mock(
            success=True,
            generated_text=self.sample_poem,
            generation_metadata=self.sample_metadata
        )
        mock_model_instance.unload_model.return_value = None
        mock_model.return_value = mock_model_instance
        
        mock_evaluator_instance = Mock()
        mock_evaluator_instance.calculate_lexical_metrics.return_value = self.sample_analysis
        mock_evaluator.return_value = mock_evaluator_instance
        
        # Test enhanced formatting
        result = self.runner.invoke(cli, [
            'generate', 'test prompt',
            '--poet', 'emily_dickinson',
            '--format', 'enhanced'
        ])
        
        assert result.exit_code == 0
        assert 'POETRY GENERATION RESULTS' in result.output
        assert 'STYLISTIC ANALYSIS' in result.output
        assert 'Emily Dickinson' in result.output
        assert 'The quiet forest whispers low' in result.output
    
    @patch('src.cli.create_poetry_model')
    @patch('src.cli.QuantitativeEvaluator')
    def test_save_results_json_format(self, mock_evaluator, mock_model):
        """Test saving results in JSON format with comprehensive metadata."""
        # Mock model
        mock_model_instance = Mock()
        mock_model_instance.load_model.return_value = True
        mock_model_instance.generate_poetry.return_value = Mock(
            success=True,
            generated_text=self.sample_poem,
            generation_metadata=self.sample_metadata
        )
        mock_model_instance.unload_model.return_value = None
        mock_model.return_value = mock_model_instance
        
        # Mock evaluator
        mock_evaluator_instance = Mock()
        mock_evaluator_instance.calculate_lexical_metrics.return_value = self.sample_analysis
        mock_evaluator.return_value = mock_evaluator_instance
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_file = f.name
        
        try:
            result = self.runner.invoke(cli, [
                'generate', 'test prompt',
                '--poet', 'walt_whitman',
                '--theme', 'nature',
                '--output', output_file,
                '--format', 'enhanced'
            ])
            
            assert result.exit_code == 0
            assert f"Output saved to: {output_file}" in result.output
            
            # Verify JSON content
            with open(output_file, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
            
            assert 'metadata' in saved_data
            assert 'results' in saved_data
            assert saved_data['metadata']['prompt'] == 'test prompt'
            assert saved_data['metadata']['poet_style'] == 'walt_whitman'
            assert saved_data['metadata']['theme'] == 'nature'
            assert 'The quiet forest whispers low' in saved_data['results']['generated_text']
            assert saved_data['results']['analysis_results'] == self.sample_analysis
        
        finally:
            Path(output_file).unlink()


class TestPoetryOutputFormatter:
    """Test the PoetryOutputFormatter class directly."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = PoetryOutputFormatter()
        self.sample_poem = """Roses are red,
Violets are blue,
Poetry is art,
And so are you."""
        
        self.sample_analysis = {
            'word_count': 12,
            'line_count': 4,
            'ttr': 0.917,
            'lexical_density': 0.75,
            'syllable_count': 16,
            'avg_syllables_per_line': 4.0
        }
    
    def test_format_poem_display(self):
        """Test poem display formatting."""
        formatted = self.formatter.format_poem_display(self.sample_poem, "Test Poem")
        
        assert "TEST POEM" in formatted
        assert "═" in formatted
        assert "Roses are red" in formatted
        assert "Violets are blue" in formatted
    
    def test_format_stylistic_analysis(self):
        """Test stylistic analysis formatting."""
        formatted = self.formatter.format_stylistic_analysis(self.sample_analysis, detailed=True)
        
        assert "STYLISTIC ANALYSIS" in formatted
        assert "BASIC METRICS" in formatted
        assert "LEXICAL METRICS" in formatted
        assert "Word count" in formatted
        assert "12" in formatted
        assert "0.917" in formatted
    
    def test_create_comprehensive_output(self):
        """Test comprehensive output creation."""
        output = self.formatter.create_comprehensive_output(
            poem_text=self.sample_poem,
            analysis_results=self.sample_analysis,
            poet_style="emily_dickinson",
            prompt="test prompt"
        )
        
        assert "POETRY GENERATION RESULTS" in output
        assert "Emily Dickinson" in output
        assert "test prompt" in output
        assert "Roses are red" in output
        assert "Violets are blue" in output
        assert "STYLISTIC ANALYSIS" in output
    
    def test_save_results_json(self):
        """Test saving results in JSON format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = Path(f.name)
        
        try:
            self.formatter.save_results_with_formatting(
                output_path=output_path,
                poem_text=self.sample_poem,
                analysis_results=self.sample_analysis,
                generation_config={'temperature': 0.8},
                generation_metadata={'model': 'gpt2'},
                prompt="test prompt"
            )
            
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            assert 'metadata' in data
            assert 'results' in data
            assert data['metadata']['prompt'] == 'test prompt'
            assert data['results']['generated_text'] == self.sample_poem
        
        finally:
            output_path.unlink()
    
    def test_format_number(self):
        """Test number formatting utility."""
        assert self.formatter._format_number(3.14159, 2) == "3.14"
        assert self.formatter._format_number(42) == "42"
        assert self.formatter._format_number("N/A") == "N/A"
    
    def test_create_progress_bar(self):
        """Test progress bar creation."""
        bar = self.formatter._create_progress_bar(0.75, width=10)
        assert len(bar) == 10
        assert "█" in bar
        assert "░" in bar
        
        # Test edge cases
        empty_bar = self.formatter._create_progress_bar(0.0, width=5)
        assert empty_bar == "░░░░░"
        
        full_bar = self.formatter._create_progress_bar(1.0, width=5)
        assert full_bar == "█████"


if __name__ == '__main__':
    pytest.main([__file__])