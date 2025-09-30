"""Tests for Emily Dickinson-specific poetry generation."""

import pytest
from unittest.mock import Mock, patch
import logging

from src.stylometric.dickinson_generation import (
    DickinsonGenerationConfig,
    DickinsonPromptEngineer,
    CommonMeterSubverter,
    DickinsonStyleValidator,
    DickinsonPoetryGenerator,
    create_dickinson_generator,
    generate_dickinson_poem
)
from src.stylometric.model_interface import (
    PoetryGenerationModel,
    PoetryGenerationRequest,
    PoetryGenerationResponse,
    GenerationConfig
)


class TestDickinsonGenerationConfig:
    """Test class for Dickinson generation configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = DickinsonGenerationConfig()
        
        assert config.enforce_dashes is True
        assert config.target_dash_frequency == 1.2
        assert config.enforce_slant_rhyme is True
        assert config.target_slant_rhyme_ratio == 0.6
        assert config.enforce_irregular_caps is True
        assert config.target_cap_frequency == 0.2
        assert config.enforce_common_meter is True
        assert config.allow_meter_subversion is True
        assert config.subversion_probability == 0.3
        assert config.max_attempts == 5
        assert config.style_threshold == 0.6
        assert config.use_style_prompts is True
        assert config.include_examples is True
        assert config.use_thematic_guidance is True


class TestDickinsonPromptEngineer:
    """Test class for Dickinson prompt engineering."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.prompt_engineer = DickinsonPromptEngineer()
    
    def test_initialization(self):
        """Test prompt engineer initialization."""
        assert len(self.prompt_engineer.dickinson_themes) > 0
        assert len(self.prompt_engineer.dickinson_vocabulary) > 0
        assert len(self.prompt_engineer.style_instructions) > 0
        
        # Check specific themes
        assert "death and immortality" in self.prompt_engineer.dickinson_themes
        assert "nature and seasons" in self.prompt_engineer.dickinson_themes
        
        # Check specific vocabulary
        assert "eternity" in self.prompt_engineer.dickinson_vocabulary
        assert "solitude" in self.prompt_engineer.dickinson_vocabulary
    
    def test_create_dickinson_prompt_basic(self):
        """Test basic Dickinson prompt creation."""
        base_prompt = "a summer day"
        prompt = self.prompt_engineer.create_dickinson_prompt(base_prompt)
        
        assert "Emily Dickinson" in prompt
        assert base_prompt in prompt
        assert "Style requirements:" in prompt


class TestCommonMeterSubverter:
    """Test class for common meter subversion."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.subverter = CommonMeterSubverter()
    
    def test_initialization(self):
        """Test subverter initialization."""
        assert self.subverter.base_pattern == [8, 6, 8, 6]
        assert len(self.subverter.dickinson_variations) > 0
        assert self.subverter.text_processor is not None
    
    def test_suggest_meter_variation_random(self):
        """Test random meter variation suggestion."""
        variation = self.subverter.suggest_meter_variation(subversion_type="random")
        
        assert isinstance(variation, list)
        assert len(variation) == 4
        assert all(isinstance(x, int) for x in variation)
        assert variation in self.subverter.dickinson_variations


class MockPoetryModel(PoetryGenerationModel):
    """Mock poetry model for testing."""
    
    def __init__(self, model_name: str = "mock_model"):
        super().__init__(model_name)
        self.is_loaded = True
        self.generation_responses = []
        self.current_response_index = 0
    
    def load_model(self) -> bool:
        self.is_loaded = True
        return True
    
    def generate_poetry(self, request: PoetryGenerationRequest) -> PoetryGenerationResponse:
        if self.current_response_index < len(self.generation_responses):
            response = self.generation_responses[self.current_response_index]
            self.current_response_index += 1
            return response
        else:
            # Default response
            return PoetryGenerationResponse(
                generated_text="I dwell in Possibility —\nA fairer House than Prose —",
                prompt=request.prompt,
                poet_style=request.poet_style,
                success=True
            )
    
    def unload_model(self) -> bool:
        self.is_loaded = False
        return True
    
    def set_responses(self, responses):
        """Set predefined responses for testing."""
        self.generation_responses = responses
        self.current_response_index = 0


class TestDickinsonStyleValidator:
    """Test class for Dickinson style validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = DickinsonStyleValidator()
        
        # Sample Dickinson-style poem for testing
        self.dickinson_style_poem = """I dwell in Possibility —
A fairer House than Prose —
More numerous of Windows —
Superior — for Doors —"""
        
        # Sample non-Dickinson poem
        self.regular_poem = """The sun shines bright today,
The birds sing in the trees.
A gentle breeze blows through the leaves,
And puts my mind at ease."""
    
    def test_initialization(self):
        """Test validator initialization."""
        assert self.validator.feature_detector is not None
        assert self.validator.meter_subverter is not None
    
    def test_validate_style_consistency_dickinson_style(self):
        """Test validation with Dickinson-style poem."""
        config = DickinsonGenerationConfig()
        results = self.validator.validate_style_consistency(
            self.dickinson_style_poem, config
        )
        
        assert "overall_similarity" in results
        assert "meets_threshold" in results
        assert "feature_validations" in results
        assert "is_valid" in results
        assert isinstance(results["overall_similarity"], float)
        assert 0.0 <= results["overall_similarity"] <= 1.0


class TestDickinsonPoetryGenerator:
    """Test class for Dickinson poetry generator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_model = MockPoetryModel()
        self.generator = DickinsonPoetryGenerator(self.mock_model)
        
        # Sample successful response
        self.good_response = PoetryGenerationResponse(
            generated_text="""I dwell in Possibility —
A fairer House than Prose —
More numerous of Windows —
Superior — for Doors —""",
            prompt="test prompt",
            poet_style="emily_dickinson",
            success=True
        )
    
    def test_initialization(self):
        """Test generator initialization."""
        assert self.generator.base_model is not None
        assert self.generator.prompt_engineer is not None
        assert self.generator.style_validator is not None
        assert self.generator.meter_subverter is not None
    
    def test_generate_dickinson_poetry_success(self):
        """Test successful Dickinson poetry generation."""
        self.mock_model.set_responses([self.good_response])
        
        result = self.generator.generate_dickinson_poetry("nature's beauty")
        
        assert result["success"] is True
        assert "generated_poem" in result
        assert "validation_results" in result
        assert "similarity_score" in result
        assert "is_style_valid" in result
        assert "attempts_made" in result
        assert result["attempts_made"] >= 1


class TestEndToEndDickinsonGeneration:
    """End-to-end integration tests for Dickinson style generation."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.mock_model = MockPoetryModel()
        
        # Create high-quality Dickinson-style response
        self.quality_response = PoetryGenerationResponse(
            generated_text="""Because I could not stop for Death —
He kindly stopped for me —
The Carriage held but just Ourselves —
And Immortality.""",
            prompt="death and time",
            poet_style="emily_dickinson",
            success=True
        )
    
    def test_complete_dickinson_generation_pipeline(self):
        """Test the complete Dickinson generation pipeline."""
        self.mock_model.set_responses([self.quality_response])
        
        # Create generator with custom config
        config = DickinsonGenerationConfig(
            enforce_dashes=True,
            enforce_slant_rhyme=True,
            enforce_irregular_caps=True,
            enforce_common_meter=True,
            max_attempts=3,
            style_threshold=0.5
        )
        
        generator = DickinsonPoetryGenerator(self.mock_model)
        
        # Generate poetry with theme
        result = generator.generate_dickinson_poetry(
            prompt="mortality and eternity",
            theme="death and immortality",
            config=config
        )
        
        # Verify successful generation
        assert result["success"] is True
        assert "generated_poem" in result
        assert "validation_results" in result
        assert "similarity_score" in result
        
        # Verify poem content
        poem = result["generated_poem"]
        assert len(poem) > 0
        assert "Death" in poem or "death" in poem
        
        # Verify validation results structure
        validation = result["validation_results"]
        assert "overall_similarity" in validation
        assert "feature_validations" in validation
        assert "is_valid" in validation
    
    def test_dickinson_style_consistency_validation(self):
        """Test style consistency validation across multiple generations."""
        # Set up multiple quality responses
        responses = [self.quality_response] * 3
        self.mock_model.set_responses(responses)
        
        generator = DickinsonPoetryGenerator(self.mock_model)
        
        prompts = ["hope and faith", "nature's mystery", "solitude and thought"]
        results = generator.batch_generate_dickinson_poetry(
            prompts, 
            theme="contemplation and wonder"
        )
        
        # Verify all generations succeeded
        assert len(results) == 3
        for result in results:
            assert result["success"] is True
            assert "validation_results" in result
            
            # Check that validation includes required features
            validation = result["validation_results"]
            assert "overall_similarity" in validation
            assert isinstance(validation["overall_similarity"], float)
    
    def test_prompt_engineering_effectiveness(self):
        """Test that prompt engineering produces Dickinson-appropriate prompts."""
        prompt_engineer = DickinsonPromptEngineer()
        
        # Test various prompt configurations
        base_prompt = "a bird's song"
        
        # Test with full configuration
        full_config = DickinsonGenerationConfig(
            use_style_prompts=True,
            include_examples=True,
            use_thematic_guidance=True
        )
        
        enhanced_prompt = prompt_engineer.create_dickinson_prompt(
            base_prompt, 
            theme="nature and transcendence",
            config=full_config
        )
        
        # Verify prompt contains Dickinson-specific elements
        assert "Emily Dickinson" in enhanced_prompt
        assert base_prompt in enhanced_prompt
        assert "Style requirements:" in enhanced_prompt
        assert "dashes" in enhanced_prompt.lower()
        assert "nature and transcendence" in enhanced_prompt
        
        # Test meter guidance
        meter_enhanced = prompt_engineer.enhance_prompt_with_meter_guidance(
            enhanced_prompt, [8, 6, 8, 6]
        )
        
        assert "Meter guidance:" in meter_enhanced
        assert "8-6-8-6" in meter_enhanced
    
    def test_common_meter_subversion_modeling(self):
        """Test common meter subversion modeling and application."""
        subverter = CommonMeterSubverter()
        
        # Test various subversion types
        random_variation = subverter.suggest_meter_variation(subversion_type="random")
        compression_variation = subverter.suggest_meter_variation(subversion_type="compression")
        expansion_variation = subverter.suggest_meter_variation(subversion_type="expansion")
        
        # Verify variations are valid
        assert len(random_variation) == 4
        assert len(compression_variation) == 4
        assert len(expansion_variation) == 4
        
        # Verify compression/expansion logic
        base_total = sum(subverter.base_pattern)
        assert sum(compression_variation) <= base_total
        assert sum(expansion_variation) >= base_total
        
        # Test meter adherence analysis
        test_poem = """The summer day was bright and clear
With birds in flight
The flowers bloomed throughout the year
A wondrous sight"""
        
        analysis = subverter.analyze_meter_adherence(test_poem)
        assert "adherence_score" in analysis
        assert 0.0 <= analysis["adherence_score"] <= 1.0


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_dickinson_generator(self):
        """Test Dickinson generator creation."""
        mock_model = MockPoetryModel()
        generator = create_dickinson_generator(mock_model)
        
        assert isinstance(generator, DickinsonPoetryGenerator)
        assert generator.base_model is mock_model
    
    def test_generate_dickinson_poem(self):
        """Test convenience function for single poem generation."""
        mock_model = MockPoetryModel()
        good_response = PoetryGenerationResponse(
            generated_text="I dwell in Possibility —\nA fairer House than Prose —",
            prompt="test",
            success=True
        )
        mock_model.set_responses([good_response])
        
        result = generate_dickinson_poem(
            mock_model,
            "hope",
            theme="faith and doubt",
            max_attempts=2
        )
        
        assert "success" in result
        assert "generated_poem" in result