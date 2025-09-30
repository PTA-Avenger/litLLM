"""
Abstract base class and concrete implementations for poetry generation models.

This module provides the interface for poetry generation models and includes
a GPT-based implementation using the transformers library.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    GenerationConfig,
    pipeline
)

# Import error handling utilities
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.exceptions import (
    PoetryLLMError, ModelError, GenerationFailureError, ResourceConstraintError,
    ErrorContext, create_user_friendly_error_message
)
from utils.error_handlers import GenerationErrorHandler

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for poetry generation parameters."""
    max_length: int = 200
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    num_return_sequences: int = 1
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None


@dataclass
class PoetryGenerationRequest:
    """Request object for poetry generation."""
    prompt: str
    poet_style: Optional[str] = None
    theme: Optional[str] = None
    form: Optional[str] = None  # sonnet, haiku, free verse, etc.
    generation_config: Optional[GenerationConfig] = None


@dataclass
class PoetryGenerationResponse:
    """Response object containing generated poetry and metadata."""
    generated_text: str
    prompt: str
    poet_style: Optional[str] = None
    generation_metadata: Dict[str, Any] = None
    success: bool = True
    error_message: Optional[str] = None


class PoetryGenerationModel(ABC):
    """Abstract base class for poetry generation models."""
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the poetry generation model.
        
        Args:
            model_name: Name or path of the model to load
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.is_loaded = False
        
    @abstractmethod
    def load_model(self) -> bool:
        """
        Load the model and tokenizer.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def generate_poetry(self, request: PoetryGenerationRequest) -> PoetryGenerationResponse:
        """
        Generate poetry based on the given request.
        
        Args:
            request: PoetryGenerationRequest containing prompt and parameters
            
        Returns:
            PoetryGenerationResponse containing generated poetry and metadata
        """
        pass
    
    @abstractmethod
    def unload_model(self) -> bool:
        """
        Unload the model to free memory.
        
        Returns:
            bool: True if model unloaded successfully, False otherwise
        """
        pass
    
    def is_model_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        return self.is_loaded


class GPTPoetryModel(PoetryGenerationModel):
    """GPT-based poetry generation model using transformers library."""
    
    def __init__(self, model_name: str = "gpt2", device: Optional[str] = None, **kwargs):
        """
        Initialize GPT poetry model.
        
        Args:
            model_name: Name of the GPT model to use (default: gpt2)
            device: Device to run the model on (cuda/cpu, auto-detected if None)
            **kwargs: Additional parameters
        """
        super().__init__(model_name, **kwargs)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.generation_pipeline = None
        
        # Style-specific prompt templates
        self.style_prompts = {
            "emily_dickinson": "Write a poem in the style of Emily Dickinson with dashes, slant rhyme, and contemplative themes:",
            "walt_whitman": "Write a poem in the style of Walt Whitman with free verse, cataloging, and expansive themes:",
            "edgar_allan_poe": "Write a poem in the style of Edgar Allan Poe with dark themes, consistent rhyme, and haunting atmosphere:",
            "general": "Write a thoughtful poem about:"
        }
    
    def load_model(self) -> bool:
        """
        Load the GPT model and tokenizer with comprehensive error handling.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        with ErrorContext(logger, f"loading model {self.model_name}", reraise=False) as ctx:
            logger.info(f"Loading model: {self.model_name}")
            
            # Check for resource constraints first
            if self.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                self.device = "cpu"
            
            # Load tokenizer with error handling
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                
                # Set pad token if not present
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    
            except Exception as e:
                error_msg = create_user_friendly_error_message(e, "tokenizer loading")
                logger.error(f"Tokenizer loading failed: {error_msg}")
                raise ModelError(
                    f"Failed to load tokenizer for {self.model_name}: {error_msg}",
                    error_code="TOKENIZER_LOAD_FAILED",
                    details={"model_name": self.model_name, "original_error": str(e)}
                )
            
            # Load model with memory management
            try:
                # Determine appropriate dtype and device settings
                torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
                device_map = "auto" if self.device == "cuda" else None
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    low_cpu_mem_usage=True  # Enable memory optimization
                )
                
                if self.device == "cpu":
                    self.model = self.model.to(self.device)
                    
            except torch.cuda.OutOfMemoryError as e:
                logger.error("GPU out of memory during model loading")
                raise ResourceConstraintError(
                    f"Insufficient GPU memory to load {self.model_name}",
                    error_code="GPU_MEMORY_INSUFFICIENT",
                    details={
                        "model_name": self.model_name,
                        "device": self.device,
                        "suggestions": [
                            "Try using a smaller model",
                            "Use CPU instead of GPU",
                            "Clear GPU cache and retry"
                        ]
                    }
                )
            except Exception as e:
                error_msg = create_user_friendly_error_message(e, "model loading")
                logger.error(f"Model loading failed: {error_msg}")
                raise ModelError(
                    f"Failed to load model {self.model_name}: {error_msg}",
                    error_code="MODEL_LOAD_FAILED",
                    details={"model_name": self.model_name, "device": self.device, "original_error": str(e)}
                )
            
            # Create generation pipeline with error handling
            try:
                self.generation_pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if self.device == "cuda" else -1
                )
                
            except Exception as e:
                error_msg = create_user_friendly_error_message(e, "pipeline creation")
                logger.error(f"Pipeline creation failed: {error_msg}")
                raise ModelError(
                    f"Failed to create generation pipeline: {error_msg}",
                    error_code="PIPELINE_CREATION_FAILED",
                    details={"model_name": self.model_name, "original_error": str(e)}
                )
            
            self.is_loaded = True
            logger.info(f"Model {self.model_name} loaded successfully on {self.device}")
            return True
        
        # If we reach here, an error occurred and was handled by ErrorContext
        self.is_loaded = False
        return False
    
    def _build_style_aware_prompt(self, request: PoetryGenerationRequest) -> str:
        """
        Build a style-aware prompt based on the request.
        
        Args:
            request: PoetryGenerationRequest containing prompt and style info
            
        Returns:
            str: Formatted prompt for generation
        """
        base_prompt = request.prompt.strip()
        
        # Get style-specific prompt template
        style_key = request.poet_style or "general"
        style_template = self.style_prompts.get(style_key, self.style_prompts["general"])
        
        # Build the full prompt
        full_prompt = f"{style_template} {base_prompt}"
        
        # Add form specification if provided
        if request.form:
            full_prompt += f" (in {request.form} form)"
        
        # Add theme specification if provided
        if request.theme:
            full_prompt += f" focusing on themes of {request.theme}"
        
        return full_prompt
    
    def generate_poetry(self, request: PoetryGenerationRequest) -> PoetryGenerationResponse:
        """
        Generate poetry using the GPT model with comprehensive error handling.
        
        Args:
            request: PoetryGenerationRequest containing prompt and parameters
            
        Returns:
            PoetryGenerationResponse containing generated poetry and metadata
        """
        if not self.is_loaded:
            error_msg = "Model not loaded. Call load_model() first."
            logger.error(error_msg)
            return PoetryGenerationResponse(
                generated_text="",
                prompt=request.prompt,
                poet_style=request.poet_style,
                success=False,
                error_message=error_msg
            )
        
        # Initialize error handler
        error_handler = GenerationErrorHandler()
        attempt_count = 1
        max_attempts = 3
        
        while attempt_count <= max_attempts:
            try:
                # Validate request parameters
                self._validate_generation_request(request)
                
                # Build style-aware prompt
                full_prompt = self._build_style_aware_prompt(request)
                
                # Get generation config
                gen_config = request.generation_config or GenerationConfig()
                
                # Generate text with timeout protection
                logger.info(f"Generating poetry (attempt {attempt_count}): {full_prompt[:100]}...")
                
                with ErrorContext(logger, "poetry generation", reraise=True):
                    outputs = self.generation_pipeline(
                        full_prompt,
                        max_length=gen_config.max_length,
                        temperature=gen_config.temperature,
                        top_p=gen_config.top_p,
                        top_k=gen_config.top_k,
                        do_sample=gen_config.do_sample,
                        num_return_sequences=gen_config.num_return_sequences,
                        pad_token_id=gen_config.pad_token_id or self.tokenizer.pad_token_id,
                        eos_token_id=gen_config.eos_token_id or self.tokenizer.eos_token_id,
                        return_full_text=False
                    )
                
                # Extract and validate generated text
                generated_text = outputs[0]['generated_text'].strip()
                
                if not generated_text:
                    raise GenerationFailureError(
                        "Generated text is empty",
                        error_code="EMPTY_GENERATION",
                        recovery_suggestions=["Try different parameters", "Check prompt validity"]
                    )
                
                # Create metadata
                metadata = {
                    "model_name": self.model_name,
                    "device": self.device,
                    "full_prompt": full_prompt,
                    "generation_config": gen_config.__dict__,
                    "attempt_count": attempt_count
                }
                
                logger.info(f"Poetry generation completed successfully on attempt {attempt_count}")
                
                return PoetryGenerationResponse(
                    generated_text=generated_text,
                    prompt=request.prompt,
                    poet_style=request.poet_style,
                    generation_metadata=metadata,
                    success=True
                )
                
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "memory" in str(e).lower() or "cuda" in str(e).lower():
                    # Handle resource constraint errors
                    logger.warning(f"Resource constraint on attempt {attempt_count}: {e}")
                    
                    if attempt_count < max_attempts:
                        # Try with fallback configuration
                        recovery_result = error_handler.handle_generation_failure(
                            e, request.generation_config.__dict__ if request.generation_config else {}, attempt_count
                        )
                        
                        if recovery_result['fallback_config']:
                            logger.info("Retrying with fallback configuration")
                            # Update request with fallback config
                            fallback_config = GenerationConfig(**recovery_result['fallback_config'])
                            request.generation_config = fallback_config
                            attempt_count += 1
                            continue
                    
                    # No more attempts available
                    return PoetryGenerationResponse(
                        generated_text="",
                        prompt=request.prompt,
                        poet_style=request.poet_style,
                        success=False,
                        error_message=create_user_friendly_error_message(e, "poetry generation")
                    )
                else:
                    raise  # Re-raise non-resource errors
                  
            except Exception as e:
                logger.error(f"Generation failed on attempt {attempt_count}: {e}")
                
                if attempt_count < max_attempts:
                    # Get recovery suggestions
                    recovery_result = error_handler.handle_generation_failure(
                        e, request.generation_config.__dict__ if request.generation_config else {}, attempt_count
                    )
                    
                    if recovery_result.get('retry_recommended', False):
                        logger.info(f"Retrying generation (attempt {attempt_count + 1})")
                        attempt_count += 1
                        continue
                
                # Final failure
                recovery_result = error_handler.handle_generation_failure(
                    e, request.generation_config.__dict__ if request.generation_config else {}, attempt_count
                )
                
                return PoetryGenerationResponse(
                    generated_text="",
                    prompt=request.prompt,
                    poet_style=request.poet_style,
                    success=False,
                    error_message=create_user_friendly_error_message(e, "poetry generation")
                )
        
        # Should not reach here, but handle just in case
        return PoetryGenerationResponse(
            generated_text="",
            prompt=request.prompt,
            poet_style=request.poet_style,
            success=False,
            error_message="Maximum generation attempts exceeded"
        )
    
    def _validate_generation_request(self, request: PoetryGenerationRequest) -> None:
        """
        Validate generation request parameters.
        
        Args:
            request: PoetryGenerationRequest to validate
            
        Raises:
            GenerationFailureError: If validation fails
        """
        # Validate prompt
        if not request.prompt or not request.prompt.strip():
            raise GenerationFailureError(
                "Prompt cannot be empty",
                error_code="EMPTY_PROMPT",
                recovery_suggestions=["Provide a non-empty prompt"]
            )
        
        # Validate generation config if provided
        if request.generation_config:
            config = request.generation_config
            
            # Validate temperature
            if config.temperature < 0.0 or config.temperature > 2.0:
                raise GenerationFailureError(
                    f"Invalid temperature: {config.temperature}. Must be between 0.0 and 2.0",
                    error_code="INVALID_TEMPERATURE",
                    recovery_suggestions=["Use temperature between 0.0 and 2.0"]
                )
            
            # Validate top_p
            if config.top_p < 0.0 or config.top_p > 1.0:
                raise GenerationFailureError(
                    f"Invalid top_p: {config.top_p}. Must be between 0.0 and 1.0",
                    error_code="INVALID_TOP_P",
                    recovery_suggestions=["Use top_p between 0.0 and 1.0"]
                )
            
            # Validate max_length
            if config.max_length < 10 or config.max_length > 2000:
                raise GenerationFailureError(
                    f"Invalid max_length: {config.max_length}. Must be between 10 and 2000",
                    error_code="INVALID_MAX_LENGTH",
                    recovery_suggestions=["Use max_length between 10 and 2000"]
                )                 
    
    def unload_model(self) -> bool:
        """
        Unload the model to free memory.
        
        Returns:
            bool: True if model unloaded successfully, False otherwise
        """
        try:
            if self.model is not None:
                del self.model
                self.model = None
            
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            
            if self.generation_pipeline is not None:
                del self.generation_pipeline
                self.generation_pipeline = None
            
            # Clear CUDA cache if using GPU
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.is_loaded = False
            logger.info(f"Model {self.model_name} unloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload model: {str(e)}")
            return False


def create_poetry_model(model_type: str = "gpt", model_name: str = "gpt2", **kwargs) -> PoetryGenerationModel:
    """
    Factory function to create poetry generation models.
    
    Args:
        model_type: Type of model to create ("gpt")
        model_name: Name of the specific model to load
        **kwargs: Additional model-specific parameters
        
    Returns:
        PoetryGenerationModel: Initialized model instance
        
    Raises:
        ValueError: If unsupported model_type is provided
    """
    if model_type.lower() == "gpt":
        return GPTPoetryModel(model_name=model_name, **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")