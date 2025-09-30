"""
Unit tests for the poetry generation model interface.

Tests cover the abstract base class, GPT implementation, and factory function.
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from src.stylometric.model_interface import (
    PoetryGenerationModel,
    GPTPoetryModel,
    PoetryGenerationRequest,
    PoetryGenerationResponse,
    GenerationConfig,
    create_poetry_model
)


class TestPoetryGenerationModel:
    """Test the abstract base class."""
    
    def test_abstract_class_cannot_be_instantiated(self):
        """Test that the abstract base class cannot be instantiated directly."""
        with pytest.raises(TypeError):
            PoetryGenerationModel("test-model")


class MockPoetryModel(PoetryGenerationModel):
    """Mock implementation for testing the abstract base class."""
    
    def load_model(self):
        self.is_loaded = True
        return True
    
    def generate_poetry(self, request):
        return PoetryGenerationResponse(
            generated_text="Mock generated poetry",
            prompt=request.prompt,
            poet_style=request.poet_style
        )
    
    def unload_model(self):
        self.is_loaded = False
        return True


class TestAbstractBaseClass:
    """Test the abstract base class functionality."""
    
    def test_mock_implementation(self):
        """Test that mock implementation works correctly."""
        model = MockPoetryModel("test-model")
        assert model.model_name == "test-model"
        assert not model.is_model_loaded()
        
        # Test loading
        assert model.load_model()
        assert model.is_model_loaded()
        
        # Test generation
        request = PoetryGenerationRequest(prompt="test prompt", poet_style="test_style")
        response = model.generate_poetry(request)
        assert response.success
        assert response.generated_text == "Mock generated poetry"
        assert response.prompt == "test prompt"
        assert response.poet_style == "test_style"
        
        # Test unloading
        assert model.unload_model()
        assert not model.is_model_loaded()


class TestGenerationConfig:
    """Test the GenerationConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = GenerationConfig()
        assert config.max_length == 200
        assert config.temperature == 0.8
        assert config.top_p == 0.9
        assert config.top_k == 50
        assert config.do_sample is True
        assert config.num_return_sequences == 1
        assert config.pad_token_id is None
        assert config.eos_token_id is None
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = GenerationConfig(
            max_length=100,
            temperature=0.5,
            top_p=0.8,
            top_k=40,
            do_sample=False,
            num_return_sequences=2
        )
        assert config.max_length == 100
        assert config.temperature == 0.5
        assert config.top_p == 0.8
        assert config.top_k == 40
        assert config.do_sample is False
        assert config.num_return_sequences == 2


class TestPoetryGenerationRequest:
    """Test the PoetryGenerationRequest dataclass."""
    
    def test_minimal_request(self):
        """Test request with only required fields."""
        request = PoetryGenerationRequest(prompt="test prompt")
        assert request.prompt == "test prompt"
        assert request.poet_style is None
        assert request.theme is None
        assert request.form is None
        assert request.generation_config is None
    
    def test_full_request(self):
        """Test request with all fields."""
        config = GenerationConfig(max_length=100)
        request = PoetryGenerationRequest(
            prompt="test prompt",
            poet_style="emily_dickinson",
            theme="nature",
            form="sonnet",
            generation_config=config
        )
        assert request.prompt == "test prompt"
        assert request.poet_style == "emily_dickinson"
        assert request.theme == "nature"
        assert request.form == "sonnet"
        assert request.generation_config == config


class TestPoetryGenerationResponse:
    """Test the PoetryGenerationResponse dataclass."""
    
    def test_successful_response(self):
        """Test successful response creation."""
        response = PoetryGenerationResponse(
            generated_text="Generated poetry",
            prompt="test prompt",
            poet_style="test_style"
        )
        assert response.generated_text == "Generated poetry"
        assert response.prompt == "test prompt"
        assert response.poet_style == "test_style"
        assert response.success is True
        assert response.error_message is None
        assert response.generation_metadata is None
    
    def test_error_response(self):
        """Test error response creation."""
        response = PoetryGenerationResponse(
            generated_text="",
            prompt="test prompt",
            success=False,
            error_message="Test error"
        )
        assert response.generated_text == ""
        assert response.prompt == "test prompt"
        assert response.success is False
        assert response.error_message == "Test error"


class TestGPTPoetryModel:
    """Test the GPT-based poetry model implementation."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = GPTPoetryModel()
        assert model.model_name == "gpt2"
        assert not model.is_model_loaded()
        assert model.device in ["cuda", "cpu"]
        assert model.model is None
        assert model.tokenizer is None
        assert model.generation_pipeline is None
    
    def test_initialization_with_custom_params(self):
        """Test model initialization with custom parameters."""
        model = GPTPoetryModel(model_name="gpt2-medium", device="cpu")
        assert model.model_name == "gpt2-medium"
        assert model.device == "cpu"
    
    def test_style_prompts(self):
        """Test that style prompts are properly defined."""
        model = GPTPoetryModel()
        assert "emily_dickinson" in model.style_prompts
        assert "walt_whitman" in model.style_prompts
        assert "edgar_allan_poe" in model.style_prompts
        assert "general" in model.style_prompts
        
        # Check that prompts contain expected style indicators
        assert "Emily Dickinson" in model.style_prompts["emily_dickinson"]
        assert "dashes" in model.style_prompts["emily_dickinson"]
        assert "Walt Whitman" in model.style_prompts["walt_whitman"]
        assert "free verse" in model.style_prompts["walt_whitman"]
        assert "Edgar Allan Poe" in model.style_prompts["edgar_allan_poe"]
        assert "dark themes" in model.style_prompts["edgar_allan_poe"]
    
    @patch('src.stylometric.model_interface.AutoTokenizer')
    @patch('src.stylometric.model_interface.AutoModelForCausalLM')
    @patch('src.stylometric.model_interface.pipeline')
    def test_load_model_success(self, mock_pipeline, mock_model, mock_tokenizer):
        """Test successful model loading."""
        # Setup mocks
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer_instance.pad_token_id = 50256
        mock_tokenizer_instance.eos_token_id = 50256
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = Mock()
        mock_model_instance.to.return_value = mock_model_instance  # Mock the .to() method
        mock_model.from_pretrained.return_value = mock_model_instance
        
        mock_pipeline_instance = Mock()
        mock_pipeline.return_value = mock_pipeline_instance
        
        # Test loading
        model = GPTPoetryModel(device="cpu")
        result = model.load_model()
        
        assert result is True
        assert model.is_model_loaded()
        assert model.tokenizer == mock_tokenizer_instance
        assert model.model == mock_model_instance
        assert model.generation_pipeline == mock_pipeline_instance
        
        # Verify pad token was set
        assert mock_tokenizer_instance.pad_token == "<eos>"
    
    @patch('src.stylometric.model_interface.AutoTokenizer')
    def test_load_model_failure(self, mock_tokenizer):
        """Test model loading failure."""
        mock_tokenizer.from_pretrained.side_effect = Exception("Loading failed")
        
        model = GPTPoetryModel()
        result = model.load_model()
        
        assert result is False
        assert not model.is_model_loaded()
    
    def test_build_style_aware_prompt(self):
        """Test style-aware prompt building."""
        model = GPTPoetryModel()
        
        # Test with Emily Dickinson style
        request = PoetryGenerationRequest(
            prompt="nature's beauty",
            poet_style="emily_dickinson"
        )
        prompt = model._build_style_aware_prompt(request)
        assert "Emily Dickinson" in prompt
        assert "nature's beauty" in prompt
        assert "dashes" in prompt
        
        # Test with form and theme
        request = PoetryGenerationRequest(
            prompt="love",
            poet_style="walt_whitman",
            form="free verse",
            theme="democracy"
        )
        prompt = model._build_style_aware_prompt(request)
        assert "Walt Whitman" in prompt
        assert "love" in prompt
        assert "free verse" in prompt
        assert "democracy" in prompt
        
        # Test with unknown style (should use general)
        request = PoetryGenerationRequest(
            prompt="test",
            poet_style="unknown_poet"
        )
        prompt = model._build_style_aware_prompt(request)
        assert "thoughtful poem" in prompt
        assert "test" in prompt
    
    def test_generate_poetry_model_not_loaded(self):
        """Test poetry generation when model is not loaded."""
        model = GPTPoetryModel()
        request = PoetryGenerationRequest(prompt="test prompt")
        
        response = model.generate_poetry(request)
        
        assert not response.success
        assert "Model not loaded" in response.error_message
        assert response.generated_text == ""
    
    @patch('src.stylometric.model_interface.AutoTokenizer')
    @patch('src.stylometric.model_interface.AutoModelForCausalLM')
    @patch('src.stylometric.model_interface.pipeline')
    def test_generate_poetry_success(self, mock_pipeline, mock_model, mock_tokenizer):
        """Test successful poetry generation."""
        # Setup mocks
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer_instance.pad_token_id = 50256
        mock_tokenizer_instance.eos_token_id = 50256
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.return_value = [{"generated_text": "  Generated poetry text  "}]
        mock_pipeline.return_value = mock_pipeline_instance
        
        # Load model and generate
        model = GPTPoetryModel(device="cpu")
        model.load_model()
        
        request = PoetryGenerationRequest(
            prompt="nature",
            poet_style="emily_dickinson"
        )
        response = model.generate_poetry(request)
        
        assert response.success
        assert response.generated_text == "Generated poetry text"
        assert response.prompt == "nature"
        assert response.poet_style == "emily_dickinson"
        assert response.generation_metadata is not None
        assert "model_name" in response.generation_metadata
    
    @patch('src.stylometric.model_interface.AutoTokenizer')
    @patch('src.stylometric.model_interface.AutoModelForCausalLM')
    @patch('src.stylometric.model_interface.pipeline')
    def test_generate_poetry_with_custom_config(self, mock_pipeline, mock_model, mock_tokenizer):
        """Test poetry generation with custom generation config."""
        # Setup mocks
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = "<pad>"
        mock_tokenizer_instance.pad_token_id = 50256
        mock_tokenizer_instance.eos_token_id = 50256
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.return_value = [{"generated_text": "Custom generated poetry"}]
        mock_pipeline.return_value = mock_pipeline_instance
        
        # Load model
        model = GPTPoetryModel(device="cpu")
        model.load_model()
        
        # Generate with custom config
        custom_config = GenerationConfig(
            max_length=100,
            temperature=0.5,
            top_p=0.8
        )
        request = PoetryGenerationRequest(
            prompt="test",
            generation_config=custom_config
        )
        response = model.generate_poetry(request)
        
        assert response.success
        assert response.generated_text == "Custom generated poetry"
        
        # Verify pipeline was called with custom config
        mock_pipeline_instance.assert_called_once()
        call_args = mock_pipeline_instance.call_args
        assert call_args[1]["max_length"] == 100
        assert call_args[1]["temperature"] == 0.5
        assert call_args[1]["top_p"] == 0.8
    
    def test_unload_model(self):
        """Test model unloading."""
        model = GPTPoetryModel()
        
        # Set some mock objects
        model.model = Mock()
        model.tokenizer = Mock()
        model.generation_pipeline = Mock()
        model.is_loaded = True
        
        result = model.unload_model()
        
        assert result is True
        assert not model.is_model_loaded()
        assert model.model is None
        assert model.tokenizer is None
        assert model.generation_pipeline is None


class TestCreatePoetryModel:
    """Test the factory function for creating poetry models."""
    
    def test_create_gpt_model(self):
        """Test creating a GPT model."""
        model = create_poetry_model("gpt", "gpt2")
        assert isinstance(model, GPTPoetryModel)
        assert model.model_name == "gpt2"
    
    def test_create_gpt_model_with_kwargs(self):
        """Test creating a GPT model with additional kwargs."""
        model = create_poetry_model("gpt", "gpt2-medium", device="cpu")
        assert isinstance(model, GPTPoetryModel)
        assert model.model_name == "gpt2-medium"
        assert model.device == "cpu"
    
    def test_create_unsupported_model_type(self):
        """Test creating an unsupported model type."""
        with pytest.raises(ValueError, match="Unsupported model type"):
            create_poetry_model("unsupported_type")
    
    def test_case_insensitive_model_type(self):
        """Test that model type is case insensitive."""
        model = create_poetry_model("GPT", "gpt2")
        assert isinstance(model, GPTPoetryModel)


if __name__ == "__main__":
    pytest.main([__file__])