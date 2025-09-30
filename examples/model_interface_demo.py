#!/usr/bin/env python3
"""
Demo script showing how to use the poetry generation model interface.

This script demonstrates the basic usage of the GPTPoetryModel class
for generating poetry in different styles.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.stylometric.model_interface import (
    GPTPoetryModel,
    PoetryGenerationRequest,
    GenerationConfig,
    create_poetry_model
)


def demo_basic_usage():
    """Demonstrate basic model usage without actually loading a model."""
    print("=== Poetry Generation Model Interface Demo ===\n")
    
    # Create a model instance
    print("1. Creating GPT poetry model...")
    model = GPTPoetryModel(model_name="gpt2", device="cpu")
    print(f"   Model created: {model.model_name}")
    print(f"   Device: {model.device}")
    print(f"   Model loaded: {model.is_model_loaded()}")
    
    # Show style prompts
    print("\n2. Available style prompts:")
    for style, prompt in model.style_prompts.items():
        print(f"   {style}: {prompt[:60]}...")
    
    # Create generation requests
    print("\n3. Creating generation requests...")
    
    # Basic request
    basic_request = PoetryGenerationRequest(
        prompt="the beauty of nature"
    )
    print(f"   Basic request: {basic_request.prompt}")
    
    # Styled request
    styled_request = PoetryGenerationRequest(
        prompt="solitude and contemplation",
        poet_style="emily_dickinson",
        theme="nature",
        form="short poem"
    )
    print(f"   Styled request: {styled_request.poet_style} style")
    
    # Request with custom config
    custom_config = GenerationConfig(
        max_length=150,
        temperature=0.7,
        top_p=0.85
    )
    config_request = PoetryGenerationRequest(
        prompt="the ocean's mystery",
        poet_style="walt_whitman",
        generation_config=custom_config
    )
    print(f"   Custom config request: temp={custom_config.temperature}")
    
    # Test prompt building
    print("\n4. Testing style-aware prompt building...")
    for request in [basic_request, styled_request, config_request]:
        prompt = model._build_style_aware_prompt(request)
        print(f"   Style: {request.poet_style or 'general'}")
        print(f"   Prompt: {prompt[:80]}...")
        print()
    
    # Test factory function
    print("5. Testing factory function...")
    factory_model = create_poetry_model("gpt", "gpt2-medium", device="cpu")
    print(f"   Factory model: {factory_model.model_name}")
    print(f"   Type: {type(factory_model).__name__}")
    
    print("\n=== Demo completed successfully! ===")
    print("\nNote: To actually generate poetry, you would need to:")
    print("1. Call model.load_model() to load the transformer model")
    print("2. Call model.generate_poetry(request) to generate text")
    print("3. Call model.unload_model() when done to free memory")


def demo_error_handling():
    """Demonstrate error handling when model is not loaded."""
    print("\n=== Error Handling Demo ===\n")
    
    model = GPTPoetryModel()
    request = PoetryGenerationRequest(prompt="test prompt")
    
    print("Attempting to generate poetry without loading model...")
    response = model.generate_poetry(request)
    
    print(f"Success: {response.success}")
    print(f"Error message: {response.error_message}")
    print(f"Generated text: '{response.generated_text}'")


if __name__ == "__main__":
    try:
        demo_basic_usage()
        demo_error_handling()
    except Exception as e:
        print(f"Demo failed with error: {e}")
        sys.exit(1)