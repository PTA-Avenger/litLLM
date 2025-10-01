"""
Emily Dickinson-specific poetry generation with style-aware prompt engineering.

This module implements Dickinson-specific generation capabilities including
prompt engineering, common meter subversion, and style consistency validation.
"""

import re
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

from .model_interface import (
    PoetryGenerationModel, 
    PoetryGenerationRequest, 
    PoetryGenerationResponse,
    GenerationConfig
)
from .dickinson_features import DickinsonFeatureDetector, DickinsonStyleProfile
from .text_processing import PoetryTextProcessor

logger = logging.getLogger(__name__)


@dataclass
class DickinsonGenerationConfig:
    """Configuration for Dickinson-specific generation parameters."""
    
    # Style enforcement parameters
    enforce_dashes: bool = True
    target_dash_frequency: float = 1.2  # Dashes per line
    enforce_slant_rhyme: bool = True
    target_slant_rhyme_ratio: float = 0.6  # 60% slant rhymes
    enforce_irregular_caps: bool = True
    target_cap_frequency: float = 0.2  # 20% irregular capitalization
    
    # Common meter parameters
    enforce_common_meter: bool = True
    allow_meter_subversion: bool = True
    subversion_probability: float = 0.3  # 30% chance of meter variation
    
    # Generation parameters
    max_attempts: int = 5  # Max attempts to generate style-compliant poetry
    style_threshold: float = 0.6  # Minimum style similarity score
    
    # Prompt engineering parameters
    use_style_prompts: bool = True
    include_examples: bool = True
    use_thematic_guidance: bool = True


class DickinsonPromptEngineer:
    """Handles Dickinson-specific prompt engineering and generation guidance."""
    
    def __init__(self):
        """Initialize the Dickinson prompt engineer."""
        self.dickinson_themes = [
            "death and immortality", "nature and seasons", "solitude and introspection",
            "love and loss", "faith and doubt", "time and eternity", "the soul and consciousness",
            "birds and bees", "flowers and gardens", "domestic life", "the sublime in ordinary"
        ]
        
        self.dickinson_vocabulary = [
            "circumference", "eternity", "immortality", "solitude", "consciousness",
            "reverence", "majesty", "infinity", "paradise", "ecstasy", "anguish",
            "rapture", "despair", "wonder", "awe", "mystery", "silence", "stillness"
        ]
        
        self.dickinson_imagery = [
            "slant of light", "loaded gun", "narrow fellow", "certain slant",
            "zero at the bone", "white heat", "purple host", "golden fleece",
            "pearl", "diamond", "amber", "crystal", "frost", "snow", "dew"
        ]
        
        self.style_instructions = {
            "dashes": "Use frequent dashes (—) for emphasis, interruption, and poetic pause",
            "capitalization": "Capitalize important nouns and concepts irregularly for emphasis",
            "slant_rhyme": "Prefer slant rhymes and near rhymes over perfect rhymes",
            "common_meter": "Follow common meter (8-6-8-6 syllables) but allow subtle variations",
            "enjambment": "Use enjambment to create flowing, continuous thought",
            "compression": "Use compressed, economical language with maximum impact"
        }
    
    def create_dickinson_prompt(self, base_prompt: str, 
                               theme: Optional[str] = None,
                               config: Optional[DickinsonGenerationConfig] = None) -> str:
        """
        Create a Dickinson-specific prompt with style guidance.
        
        Args:
            base_prompt: Base prompt or topic
            theme: Optional specific theme
            config: Generation configuration
            
        Returns:
            Enhanced prompt with Dickinson-specific guidance
        """
        config = config or DickinsonGenerationConfig()
        
        # Start with base instruction
        prompt_parts = [
            "Write a poem in the distinctive style of Emily Dickinson."
        ]
        
        # Add style-specific instructions
        if config.use_style_prompts:
            style_guidance = []
            
            if config.enforce_dashes:
                style_guidance.append(self.style_instructions["dashes"])
            
            if config.enforce_irregular_caps:
                style_guidance.append(self.style_instructions["capitalization"])
            
            if config.enforce_slant_rhyme:
                style_guidance.append(self.style_instructions["slant_rhyme"])
            
            if config.enforce_common_meter:
                if config.allow_meter_subversion:
                    style_guidance.append(
                        "Follow common meter (8-6-8-6 syllables) with occasional subtle variations"
                    )
                else:
                    style_guidance.append(self.style_instructions["common_meter"])
            
            if style_guidance:
                prompt_parts.append("Style requirements:")
                prompt_parts.extend([f"- {instruction}" for instruction in style_guidance])
        
        # Add thematic guidance
        if config.use_thematic_guidance:
            selected_theme = theme or random.choice(self.dickinson_themes)
            prompt_parts.append(f"Theme: {selected_theme}")
            
            # Add relevant vocabulary suggestions
            relevant_vocab = random.sample(self.dickinson_vocabulary, 3)
            prompt_parts.append(f"Consider using words like: {', '.join(relevant_vocab)}")
        
        # Add example if requested
        if config.include_examples:
            example = self._get_style_example()
            prompt_parts.append(f"Example of Dickinson's style:\n{example}")
        
        # Add the actual prompt
        prompt_parts.append(f"Topic/Prompt: {base_prompt}")
        prompt_parts.append("Poem:")
        
        return "\n\n".join(prompt_parts)
    
    def _get_style_example(self) -> str:
        """Get a brief example of Dickinson's style for prompt guidance."""
        examples = [
            "I'm Nobody! Who are you?\nAre you — Nobody — Too?\nThen there's a pair of us!\nDon't tell! they'd advertise — you know!",
            "Because I could not stop for Death —\nHe kindly stopped for me —\nThe Carriage held but just Ourselves —\nAnd Immortality.",
            "Hope is the thing with feathers —\nThat perches in the soul —\nAnd sings the tune without the words —\nAnd never stops — at all —"
        ]
        return random.choice(examples)
    
    def enhance_prompt_with_meter_guidance(self, prompt: str, 
                                         target_pattern: List[int] = None) -> str:
        """
        Enhance prompt with specific meter guidance.
        
        Args:
            prompt: Base prompt
            target_pattern: Target syllable pattern (default: [8,6,8,6])
            
        Returns:
            Enhanced prompt with meter guidance
        """
        target_pattern = target_pattern or [8, 6, 8, 6]
        
        meter_instruction = (
            f"Follow this syllable pattern: {'-'.join(map(str, target_pattern))} "
            f"(line 1: {target_pattern[0]} syllables, line 2: {target_pattern[1]} syllables, etc.)"
        )
        
        return f"{prompt}\n\nMeter guidance: {meter_instruction}"


class CommonMeterSubverter:
    """Handles common meter subversion modeling and application."""
    
    def __init__(self):
        """Initialize the common meter subverter."""
        self.base_pattern = [8, 6, 8, 6]  # Standard common meter
        
        # Common Dickinson variations
        self.dickinson_variations = [
            [7, 6, 8, 6],   # Shortened first line
            [8, 5, 8, 6],   # Shortened second line
            [8, 6, 7, 6],   # Shortened third line
            [8, 6, 8, 5],   # Shortened fourth line
            [9, 6, 8, 6],   # Extended first line
            [8, 7, 8, 6],   # Extended second line
            [8, 6, 9, 6],   # Extended third line
            [8, 6, 8, 7],   # Extended fourth line
        ]
        
        self.text_processor = PoetryTextProcessor()
    
    def suggest_meter_variation(self, base_pattern: List[int] = None,
                              subversion_type: str = "random") -> List[int]:
        """
        Suggest a meter variation based on Dickinson's patterns.
        
        Args:
            base_pattern: Base meter pattern
            subversion_type: Type of subversion ("random", "compression", "expansion")
            
        Returns:
            Suggested meter pattern
        """
        base_pattern = base_pattern or self.base_pattern
        
        if subversion_type == "random":
            return random.choice(self.dickinson_variations)
        elif subversion_type == "compression":
            # Prefer patterns with fewer syllables
            compressed = [p for p in self.dickinson_variations 
                         if sum(p) < sum(base_pattern)]
            return random.choice(compressed) if compressed else base_pattern
        elif subversion_type == "expansion":
            # Prefer patterns with more syllables
            expanded = [p for p in self.dickinson_variations 
                       if sum(p) > sum(base_pattern)]
            return random.choice(expanded) if expanded else base_pattern
        else:
            return base_pattern
    
    def analyze_meter_adherence(self, poem: str, 
                               target_pattern: List[int] = None) -> Dict[str, Any]:
        """
        Analyze how well a poem adheres to a target meter pattern.
        
        Args:
            poem: Poem text to analyze
            target_pattern: Target syllable pattern
            
        Returns:
            Dictionary with adherence analysis
        """
        target_pattern = target_pattern or self.base_pattern
        lines = self.text_processor.segment_lines(poem)
        
        if not lines:
            return {"adherence_score": 0.0, "analysis": "No lines found"}
        
        # Group lines into stanzas of 4 (common meter)
        stanzas = []
        for i in range(0, len(lines), 4):
            stanza = lines[i:i+4]
            if len(stanza) == 4:  # Only analyze complete 4-line stanzas
                stanzas.append(stanza)
        
        if not stanzas:
            return {"adherence_score": 0.0, "analysis": "No complete 4-line stanzas found"}
        
        total_adherence = 0.0
        stanza_analyses = []
        
        for stanza in stanzas:
            syllable_counts = []
            for line in stanza:
                count = self.text_processor.count_syllables_line(line)
                syllable_counts.append(count)
            
            # Calculate adherence for this stanza
            adherence = self._calculate_pattern_adherence(syllable_counts, target_pattern)
            total_adherence += adherence
            
            stanza_analyses.append({
                "syllable_counts": syllable_counts,
                "target_pattern": target_pattern,
                "adherence_score": adherence,
                "deviations": [abs(actual - target) for actual, target 
                              in zip(syllable_counts, target_pattern)]
            })
        
        overall_adherence = total_adherence / len(stanzas)
        
        return {
            "adherence_score": overall_adherence,
            "stanza_count": len(stanzas),
            "stanza_analyses": stanza_analyses,
            "average_deviation": sum(
                sum(analysis["deviations"]) / len(analysis["deviations"])
                for analysis in stanza_analyses
            ) / len(stanza_analyses) if stanza_analyses else 0.0
        }
    
    def _calculate_pattern_adherence(self, actual: List[int], 
                                   target: List[int], 
                                   tolerance: int = 1) -> float:
        """
        Calculate adherence score for a syllable pattern.
        
        Args:
            actual: Actual syllable counts
            target: Target syllable counts
            tolerance: Acceptable deviation
            
        Returns:
            Adherence score (0.0 to 1.0)
        """
        if len(actual) != len(target):
            return 0.0
        
        matches = sum(1 for a, t in zip(actual, target) 
                     if abs(a - t) <= tolerance)
        
        return matches / len(target)
class DickinsonStyleValidator:
    """Validates generated poetry for Dickinson style consistency."""
    
    def __init__(self):
        """Initialize the style validator."""
        self.feature_detector = DickinsonFeatureDetector()
        self.meter_subverter = CommonMeterSubverter()
    
    def validate_style_consistency(self, poem: str, 
                                 config: Optional[DickinsonGenerationConfig] = None) -> Dict[str, Any]:
        """
        Validate that a poem meets Dickinson style requirements.
        
        Args:
            poem: Generated poem to validate
            config: Generation configuration with style requirements
            
        Returns:
            Dictionary with validation results
        """
        config = config or DickinsonGenerationConfig()
        
        # Get Dickinson similarity scores
        similarity_scores = self.feature_detector.score_dickinson_similarity(poem)
        
        # Analyze specific features
        dash_analysis = self.feature_detector.detect_dash_patterns(poem)
        cap_analysis = self.feature_detector.detect_irregular_capitalization(poem)
        rhyme_analysis = self.feature_detector.detect_slant_rhymes(poem)
        meter_analysis = self.meter_subverter.analyze_meter_adherence(poem)
        
        # Check individual requirements
        validation_results = {
            "overall_similarity": similarity_scores["overall_similarity"],
            "meets_threshold": similarity_scores["overall_similarity"] >= config.style_threshold,
            "feature_validations": {}
        }
        
        # Validate dash usage
        if config.enforce_dashes:
            dash_valid = (dash_analysis["dash_frequency"] >= 
                         config.target_dash_frequency * 0.7)  # 70% of target
            validation_results["feature_validations"]["dashes"] = {
                "valid": dash_valid,
                "actual": dash_analysis["dash_frequency"],
                "target": config.target_dash_frequency,
                "score": similarity_scores["dash_usage"]
            }
        
        # Validate slant rhyme usage
        if config.enforce_slant_rhyme:
            slant_valid = (rhyme_analysis["slant_rhyme_frequency"] >= 
                          config.target_slant_rhyme_ratio * 0.6)  # 60% of target
            validation_results["feature_validations"]["slant_rhyme"] = {
                "valid": slant_valid,
                "actual": rhyme_analysis["slant_rhyme_frequency"],
                "target": config.target_slant_rhyme_ratio,
                "score": similarity_scores["slant_rhyme"]
            }
        
        # Validate irregular capitalization
        if config.enforce_irregular_caps:
            cap_valid = (cap_analysis["irregular_frequency"] >= 
                        config.target_cap_frequency * 0.5)  # 50% of target
            validation_results["feature_validations"]["capitalization"] = {
                "valid": cap_valid,
                "actual": cap_analysis["irregular_frequency"],
                "target": config.target_cap_frequency,
                "score": similarity_scores["capitalization"]
            }
        
        # Validate common meter
        if config.enforce_common_meter:
            meter_valid = meter_analysis["adherence_score"] >= 0.5  # 50% adherence
            validation_results["feature_validations"]["common_meter"] = {
                "valid": meter_valid,
                "actual": meter_analysis["adherence_score"],
                "target": 0.75,  # Target 75% adherence
                "score": similarity_scores["common_meter"]
            }
        
        # Calculate overall validation
        feature_validations = validation_results["feature_validations"]
        if feature_validations:
            valid_features = sum(1 for v in feature_validations.values() if v["valid"])
            validation_results["feature_compliance"] = valid_features / len(feature_validations)
        else:
            validation_results["feature_compliance"] = 1.0
        
        # Overall validation
        validation_results["is_valid"] = (
            validation_results["meets_threshold"] and
            validation_results["feature_compliance"] >= 0.6  # 60% of features must pass
        )
        
        return validation_results
    
    def suggest_improvements(self, poem: str, 
                           validation_results: Dict[str, Any]) -> List[str]:
        """
        Suggest improvements based on validation results.
        
        Args:
            poem: Original poem
            validation_results: Results from validate_style_consistency
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        if not validation_results["is_valid"]:
            suggestions.append("Overall style similarity is below threshold")
        
        feature_validations = validation_results.get("feature_validations", {})
        
        for feature, validation in feature_validations.items():
            if not validation["valid"]:
                if feature == "dashes":
                    suggestions.append(
                        f"Add more dashes for emphasis and pause (current: {validation['actual']:.2f}, "
                        f"target: {validation['target']:.2f} per line)"
                    )
                elif feature == "slant_rhyme":
                    suggestions.append(
                        f"Use more slant rhymes instead of perfect rhymes (current: {validation['actual']:.2f}, "
                        f"target: {validation['target']:.2f} ratio)"
                    )
                elif feature == "capitalization":
                    suggestions.append(
                        f"Capitalize more important nouns for emphasis (current: {validation['actual']:.2f}, "
                        f"target: {validation['target']:.2f} frequency)"
                    )
                elif feature == "common_meter":
                    suggestions.append(
                        f"Improve adherence to common meter pattern (current: {validation['actual']:.2f}, "
                        f"target: {validation['target']:.2f} adherence)"
                    )
        
        return suggestions


class DickinsonPoetryGenerator:
    """Main class for Dickinson-specific poetry generation."""
    
    def __init__(self, base_model: PoetryGenerationModel):
        """
        Initialize the Dickinson poetry generator.
        
        Args:
            base_model: Base poetry generation model
        """
        self.base_model = base_model
        self.prompt_engineer = DickinsonPromptEngineer()
        self.style_validator = DickinsonStyleValidator()
        self.meter_subverter = CommonMeterSubverter()
    
    def generate_dickinson_poetry(self, prompt: str,
                                theme: Optional[str] = None,
                                config: Optional[DickinsonGenerationConfig] = None,
                                generation_config: Optional[GenerationConfig] = None) -> Dict[str, Any]:
        """
        Generate poetry in Emily Dickinson's style with validation.
        
        Args:
            prompt: Base prompt or topic
            theme: Optional specific theme
            config: Dickinson-specific generation configuration
            generation_config: Base generation configuration
            
        Returns:
            Dictionary with generated poetry and validation results
        """
        config = config or DickinsonGenerationConfig()
        generation_config = generation_config or GenerationConfig()
        
        logger.info(f"Generating Dickinson-style poetry for prompt: {prompt}")
        
        best_result = None
        best_score = 0.0
        attempts = []
        
        for attempt in range(config.max_attempts):
            logger.debug(f"Generation attempt {attempt + 1}/{config.max_attempts}")
            
            # Create Dickinson-specific prompt
            enhanced_prompt = self.prompt_engineer.create_dickinson_prompt(
                prompt, theme, config
            )
            
            # Add meter guidance if requested
            if config.enforce_common_meter and config.allow_meter_subversion:
                if random.random() < config.subversion_probability:
                    meter_pattern = self.meter_subverter.suggest_meter_variation()
                    enhanced_prompt = self.prompt_engineer.enhance_prompt_with_meter_guidance(
                        enhanced_prompt, meter_pattern
                    )
            
            # Generate poetry
            request = PoetryGenerationRequest(
                prompt=enhanced_prompt,
                poet_style="emily_dickinson",
                theme=theme,
                generation_config=generation_config
            )
            
            response = self.base_model.generate_poetry(request)
            
            if not response.success:
                logger.warning(f"Generation attempt {attempt + 1} failed: {response.error_message}")
                continue
            
            # Validate style consistency
            validation_results = self.style_validator.validate_style_consistency(
                response.generated_text, config
            )
            
            attempt_result = {
                "attempt": attempt + 1,
                "generated_text": response.generated_text,
                "enhanced_prompt": enhanced_prompt,
                "validation": validation_results,
                "similarity_score": validation_results["overall_similarity"]
            }
            
            attempts.append(attempt_result)
            
            # Check if this is the best result so far
            if validation_results["overall_similarity"] > best_score:
                best_result = attempt_result
                best_score = validation_results["overall_similarity"]
            
            # If we meet the threshold, we can stop early
            if validation_results["is_valid"]:
                logger.info(f"Generated valid Dickinson-style poetry on attempt {attempt + 1}")
                break
        
        # Prepare final result
        if best_result:
            result = {
                "success": True,
                "generated_poem": best_result["generated_text"],
                "validation_results": best_result["validation"],
                "similarity_score": best_result["similarity_score"],
                "is_style_valid": best_result["validation"]["is_valid"],
                "attempts_made": len(attempts),
                "best_attempt": best_result["attempt"]
            }
            
            # Add improvement suggestions if not valid
            if not best_result["validation"]["is_valid"]:
                suggestions = self.style_validator.suggest_improvements(
                    best_result["generated_text"],
                    best_result["validation"]
                )
                result["improvement_suggestions"] = suggestions
            
            logger.info(f"Best result achieved similarity score: {best_score:.3f}")
            
        else:
            result = {
                "success": False,
                "error": "Failed to generate any poetry after maximum attempts",
                "attempts_made": len(attempts),
                "generated_poem": None,
                "validation_results": None
            }
            
            logger.error("Failed to generate Dickinson-style poetry after all attempts")
        
        # Add detailed attempt information for debugging
        result["all_attempts"] = attempts
        
        return result
    
    def batch_generate_dickinson_poetry(self, prompts: List[str],
                                      theme: Optional[str] = None,
                                      config: Optional[DickinsonGenerationConfig] = None) -> List[Dict[str, Any]]:
        """
        Generate multiple Dickinson-style poems from a list of prompts.
        
        Args:
            prompts: List of prompts
            theme: Optional theme for all poems
            config: Generation configuration
            
        Returns:
            List of generation results
        """
        results = []
        
        for i, prompt in enumerate(prompts):
            logger.info(f"Generating poem {i + 1}/{len(prompts)}")
            result = self.generate_dickinson_poetry(prompt, theme, config)
            results.append(result)
        
        return results


# Convenience functions for external use
def create_dickinson_generator(base_model: PoetryGenerationModel) -> DickinsonPoetryGenerator:
    """Create a Dickinson poetry generator with a base model."""
    return DickinsonPoetryGenerator(base_model)


def generate_dickinson_poem(base_model: PoetryGenerationModel,
                          prompt: str,
                          theme: Optional[str] = None,
                          **config_kwargs) -> Dict[str, Any]:
    """
    Convenience function to generate a single Dickinson-style poem.
    
    Args:
        base_model: Base poetry generation model
        prompt: Prompt for generation
        theme: Optional theme
        **config_kwargs: Configuration parameters
        
    Returns:
        Generation result dictionary
    """
    config = DickinsonGenerationConfig(**config_kwargs)
    generator = DickinsonPoetryGenerator(base_model)
    return generator.generate_dickinson_poetry(prompt, theme, config)