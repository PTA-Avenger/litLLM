"""
Integration tests for CLI with actual model interface.

These tests verify that the CLI integrates properly with the model interface
and other system components.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock
from click.testing import CliRunner

# Import CLI module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from cli import cli


class TestCLIIntegration:
    """Integration tests for CLI functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    @patch('cli.create_poetry_model')
    @patch('cli.config_manager')
    def test_end_to_end_generation_workflow(self, mock_config, mock_create_model):
        """Test complete end-to-end generation workflow."""
        # Setup configuration mock
        mock_config.get_config.return_value = Mock(
            model=Mock(base_model_name='gpt2'),
            device='cpu',
            data=Mock(data_dir='./data')
        )
        
        # Setup model mock
        mock_model = Mock()
        mock_model.load_model.return_value = True
        mock_model.unload_model.return_value = True
        
        # Mock successful generation
        from stylometric.model_interface import PoetryGenerationResponse
        mock_response = PoetryGenerationResponse(
            generated_text="The morning dew glistens\nOn petals soft and bright\nNature's gentle whisper",
            prompt="morning garden",
            poet_style="emily_dickinson",
            generation_metadata={
                'model_name': 'gpt2',
                'device': 'cpu',
                'full_prompt': 'Write a poem in the style of Emily Dickinson with dashes, slant rhyme, and contemplative themes: morning garden'
            },
            success=True
        )
        mock_model.generate_poetry.return_value = mock_response
        mock_create_model.return_value = mock_model
        
        # Test the complete workflow
        result = self.runner.invoke(cli, [
            'generate', 'morning garden',
            '--poet', 'emily_dickinson',
            '--form', 'free_verse',
            '--theme', 'nature',
            '--temperature', '0.7',
            '--max-length', '100'
        ])
        
        # Verify successful execution
        assert result.exit_code == 0
        
        # Verify output contains expected elements
        assert 'Poetry Generation' in result.output
        assert 'Prompt: morning garden' in result.output
        assert 'emily_dickinson' in result.output
        assert 'Form: free_verse' in result.output
        assert 'Theme: nature' in result.output
        assert 'Temperature: 0.7' in result.output
        assert 'Loading model...' in result.output
        assert 'Generating poetry...' in result.output
        assert 'GENERATED POEM' in result.output
        assert 'The morning dew glistens' in result.output
        
        # Verify model interactions
        mock_model.load_model.assert_called_once()
        mock_model.generate_poetry.assert_called_once()
        mock_model.unload_model.assert_called_once()
        
        # Verify generation request parameters
        call_args = mock_model.generate_poetry.call_args[0][0]
        assert call_args.prompt == 'morning garden'
        assert call_args.poet_style == 'emily_dickinson'
        assert call_args.form == 'free_verse'
        assert call_args.theme == 'nature'
        assert call_args.generation_config.temperature == 0.7
        assert call_args.generation_config.max_length == 100
    
    @patch('cli.QuantitativeEvaluator')
    @patch('cli.create_poetry_model')
    @patch('cli.config_manager')
    def test_generation_with_analysis_integration(self, mock_config, mock_create_model, mock_evaluator_class):
        """Test generation with stylistic analysis integration."""
        # Setup mocks
        mock_config.get_config.return_value = Mock()
        mock_model = Mock()
        mock_model.load_model.return_value = True
        mock_model.unload_model.return_value = True
        
        from stylometric.model_interface import PoetryGenerationResponse
        mock_response = PoetryGenerationResponse(
            generated_text="Roses are red\nViolets are blue\nPoetry is art\nAnd so are you",
            prompt="simple poem",
            success=True
        )
        mock_model.generate_poetry.return_value = mock_response
        mock_create_model.return_value = mock_model
        
        # Setup evaluator mock
        mock_evaluator = Mock()
        mock_evaluator.calculate_lexical_metrics.return_value = {
            'ttr': 0.85,
            'mattr': 0.78,
            'lexical_density': 0.65,
            'word_count': 12,
            'unique_words': 10,
            'avg_word_length': 4.2
        }
        mock_evaluator_class.return_value = mock_evaluator
        
        # Test generation with analysis
        result = self.runner.invoke(cli, [
            'generate', 'simple poem',
            '--analyze'
        ])
        
        # Verify successful execution
        assert result.exit_code == 0
        assert 'Stylistic Analysis' in result.output
        assert 'Type-Token Ratio: 0.850' in result.output
        
        # Verify evaluator was called
        mock_evaluator.calculate_lexical_metrics.assert_called_once_with(mock_response.generated_text)
    
    @patch('cli.create_poetry_model')
    @patch('cli.config_manager')
    def test_output_file_integration(self, mock_config, mock_create_model):
        """Test output file saving integration."""
        # Setup mocks
        mock_config.get_config.return_value = Mock()
        mock_model = Mock()
        mock_model.load_model.return_value = True
        mock_model.unload_model.return_value = True
        
        from stylometric.model_interface import PoetryGenerationResponse
        mock_response = PoetryGenerationResponse(
            generated_text="Test poem content\nSecond line here",
            prompt="test prompt",
            poet_style="general",
            success=True
        )
        mock_model.generate_poetry.return_value = mock_response
        mock_create_model.return_value = mock_model
        
        # Test with temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            result = self.runner.invoke(cli, [
                'generate', 'test prompt',
                '--output', tmp_path
            ])
            
            # Verify successful execution
            assert result.exit_code == 0
            assert f'Output saved to: {tmp_path}' in result.output
            
            # Verify file contents
            with open(tmp_path, 'r') as f:
                content = f.read()
                assert 'Prompt: test prompt' in content
                assert 'Test poem content' in content
                assert 'Second line here' in content
        
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def test_command_discovery(self):
        """Test that all expected commands are available."""
        result = self.runner.invoke(cli, ['--help'])
        
        assert result.exit_code == 0
        
        # Verify all expected commands are listed
        expected_commands = ['generate', 'list-poets', 'list-forms', 'examples', 'info']
        for command in expected_commands:
            assert command in result.output
    
    def test_parameter_validation_integration(self):
        """Test parameter validation across different commands."""
        # Test invalid poet style
        result = self.runner.invoke(cli, [
            'generate', 'test',
            '--poet', 'nonexistent_poet'
        ])
        assert result.exit_code == 2
        assert 'Invalid poet style' in result.output
        
        # Test invalid form
        result = self.runner.invoke(cli, [
            'generate', 'test',
            '--form', 'nonexistent_form'
        ])
        assert result.exit_code == 2
        assert 'Invalid form' in result.output
        
        # Test invalid temperature
        result = self.runner.invoke(cli, [
            'generate', 'test',
            '--temperature', '3.0'
        ])
        assert result.exit_code == 2
        assert 'Temperature must be between 0.0 and 2.0' in result.output


if __name__ == '__main__':
    pytest.main([__file__])