"""
Tests for the CLI poetry generation interface.

This module contains comprehensive tests for the command-line interface,
including user experience tests, input validation, and integration tests.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner

# Import CLI module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from cli import cli, validate_poet_style, validate_form, validate_temperature
from stylometric.model_interface import PoetryGenerationResponse, GenerationConfig


class TestCLIValidation:
    """Test input validation functions."""
    
    def test_validate_poet_style_valid(self):
        """Test validation of valid poet styles."""
        ctx = Mock()
        param = Mock()
        
        # Test valid poet styles
        assert validate_poet_style(ctx, param, 'emily_dickinson') == 'emily_dickinson'
        assert validate_poet_style(ctx, param, 'walt_whitman') == 'walt_whitman'
        assert validate_poet_style(ctx, param, 'edgar_allan_poe') == 'edgar_allan_poe'
        assert validate_poet_style(ctx, param, 'general') == 'general'
        assert validate_poet_style(ctx, param, None) is None
    
    def test_validate_poet_style_invalid(self):
        """Test validation of invalid poet styles."""
        from click import BadParameter
        
        ctx = Mock()
        param = Mock()
        
        with pytest.raises(BadParameter):
            validate_poet_style(ctx, param, 'invalid_poet')
        
        with pytest.raises(BadParameter):
            validate_poet_style(ctx, param, 'shakespeare')
    
    def test_validate_form_valid(self):
        """Test validation of valid poetic forms."""
        ctx = Mock()
        param = Mock()
        
        assert validate_form(ctx, param, 'sonnet') == 'sonnet'
        assert validate_form(ctx, param, 'haiku') == 'haiku'
        assert validate_form(ctx, param, 'free_verse') == 'free_verse'
        assert validate_form(ctx, param, None) is None
    
    def test_validate_form_invalid(self):
        """Test validation of invalid poetic forms."""
        from click import BadParameter
        
        ctx = Mock()
        param = Mock()
        
        with pytest.raises(BadParameter):
            validate_form(ctx, param, 'invalid_form')
    
    def test_validate_temperature_valid(self):
        """Test validation of valid temperature values."""
        ctx = Mock()
        param = Mock()
        
        assert validate_temperature(ctx, param, 0.0) == 0.0
        assert validate_temperature(ctx, param, 0.8) == 0.8
        assert validate_temperature(ctx, param, 2.0) == 2.0
        assert validate_temperature(ctx, param, None) is None
    
    def test_validate_temperature_invalid(self):
        """Test validation of invalid temperature values."""
        from click import BadParameter
        
        ctx = Mock()
        param = Mock()
        
        with pytest.raises(BadParameter):
            validate_temperature(ctx, param, -0.1)
        
        with pytest.raises(BadParameter):
            validate_temperature(ctx, param, 2.1)


class TestCLICommands:
    """Test CLI command functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_list_poets_command(self):
        """Test the list-poets command."""
        result = self.runner.invoke(cli, ['list-poets'])
        
        assert result.exit_code == 0
        assert 'Available Poet Styles' in result.output
        assert 'emily_dickinson' in result.output
        assert 'walt_whitman' in result.output
        assert 'edgar_allan_poe' in result.output
        assert 'general' in result.output
    
    def test_list_forms_command(self):
        """Test the list-forms command."""
        result = self.runner.invoke(cli, ['list-forms'])
        
        assert result.exit_code == 0
        assert 'Available Poetic Forms' in result.output
        assert 'sonnet' in result.output
        assert 'haiku' in result.output
        assert 'free_verse' in result.output
    
    def test_examples_command_general(self):
        """Test the examples command without specific poet."""
        result = self.runner.invoke(cli, ['examples'])
        
        assert result.exit_code == 0
        assert 'Usage Examples' in result.output
        assert 'poetry-cli generate' in result.output
        assert '--poet' in result.output
        assert '--form' in result.output
    
    def test_examples_command_specific_poet(self):
        """Test the examples command with specific poet."""
        result = self.runner.invoke(cli, ['examples', '--poet', 'emily_dickinson'])
        
        assert result.exit_code == 0
        assert 'emily_dickinson' in result.output
        assert 'poetry-cli generate' in result.output
    
    def test_info_command(self):
        """Test the info command."""
        with patch('cli.config_manager') as mock_config:
            mock_config.get_config.return_value = Mock(
                model=Mock(base_model_name='gpt2'),
                device='cpu',
                data=Mock(data_dir='./data')
            )
            
            result = self.runner.invoke(cli, ['info'])
            
            assert result.exit_code == 0
            assert 'Stylistic Poetry LLM Framework' in result.output
            assert 'Available Poet Styles' in result.output
            assert 'Available Forms' in result.output
            assert 'Configuration' in result.output


class TestGenerateCommand:
    """Test the generate command functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        
        # Mock successful generation response
        self.mock_response = PoetryGenerationResponse(
            generated_text="A beautiful poem\nAbout nature's grace\nIn morning light",
            prompt="nature",
            poet_style="emily_dickinson",
            generation_metadata={'model_name': 'gpt2'},
            success=True
        )
    
    @patch('cli.create_poetry_model')
    @patch('cli.config_manager')
    def test_generate_basic_success(self, mock_config, mock_create_model):
        """Test basic successful poetry generation."""
        # Setup mocks
        mock_config.get_config.return_value = Mock()
        mock_model = Mock()
        mock_model.load_model.return_value = True
        mock_model.generate_poetry.return_value = self.mock_response
        mock_model.unload_model.return_value = True
        mock_create_model.return_value = mock_model
        
        result = self.runner.invoke(cli, ['generate', 'nature'])
        
        assert result.exit_code == 0
        assert 'Poetry Generation' in result.output
        assert 'Loading model' in result.output
        assert 'Generating poetry' in result.output
        assert 'GENERATED POEM' in result.output
        assert 'A beautiful poem' in result.output
        
        # Verify model methods were called
        mock_model.load_model.assert_called_once()
        mock_model.generate_poetry.assert_called_once()
        mock_model.unload_model.assert_called_once()
    
    @patch('cli.create_poetry_model')
    @patch('cli.config_manager')
    def test_generate_with_poet_style(self, mock_config, mock_create_model):
        """Test generation with specific poet style."""
        mock_config.get_config.return_value = Mock()
        mock_model = Mock()
        mock_model.load_model.return_value = True
        mock_model.generate_poetry.return_value = self.mock_response
        mock_model.unload_model.return_value = True
        mock_create_model.return_value = mock_model
        
        result = self.runner.invoke(cli, [
            'generate', 'nature', 
            '--poet', 'emily_dickinson'
        ])
        
        assert result.exit_code == 0
        assert 'emily_dickinson' in result.output
        assert 'Emily Dickinson' in result.output
        
        # Check that the request was made with correct poet style
        call_args = mock_model.generate_poetry.call_args[0][0]
        assert call_args.poet_style == 'emily_dickinson'
    
    @patch('cli.create_poetry_model')
    @patch('cli.config_manager')
    def test_generate_model_load_failure(self, mock_config, mock_create_model):
        """Test handling of model loading failure."""
        mock_config.get_config.return_value = Mock()
        mock_model = Mock()
        mock_model.load_model.return_value = False
        mock_create_model.return_value = mock_model
        
        result = self.runner.invoke(cli, ['generate', 'nature'])
        
        assert result.exit_code == 1
        assert 'Failed to load model' in result.output
    
    def test_generate_empty_prompt(self):
        """Test handling of empty prompt."""
        result = self.runner.invoke(cli, ['generate', ''])
        
        assert result.exit_code == 1  # CLI error exit code
        assert 'Prompt cannot be empty' in result.output
    
    def test_generate_invalid_poet(self):
        """Test handling of invalid poet style."""
        result = self.runner.invoke(cli, [
            'generate', 'nature',
            '--poet', 'invalid_poet'
        ])
        
        assert result.exit_code == 2
        assert 'Invalid poet style' in result.output


class TestCLIUserExperience:
    """Test user experience aspects of the CLI."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_help_messages(self):
        """Test that help messages are informative and clear."""
        # Test main help
        result = self.runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'Stylistic Poetry LLM Framework' in result.output
        assert 'Generate poetry in the style of renowned' in result.output
        
        # Test generate command help
        result = self.runner.invoke(cli, ['generate', '--help'])
        assert result.exit_code == 0
        assert 'Generate poetry based on a prompt' in result.output
        assert 'Examples:' in result.output
        assert '--poet' in result.output
        assert '--form' in result.output
    
    def test_error_messages_are_clear(self):
        """Test that error messages provide clear guidance."""
        # Test invalid poet
        result = self.runner.invoke(cli, [
            'generate', 'nature', '--poet', 'shakespeare'
        ])
        assert 'Invalid poet style' in result.output
        assert 'Available options:' in result.output


if __name__ == '__main__':
    pytest.main([__file__])