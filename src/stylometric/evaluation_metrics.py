"""
Quantitative evaluation metrics for generated poetry.

This module provides comprehensive evaluation metrics to assess the quality
and stylistic fidelity of generated poetry against target poet styles.
"""

import math
import re
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
from dataclasses import dataclass

from .lexical_analysis import LexicalAnalyzer
from .structural_analysis import StructuralAnalyzer
from .text_processing import PoetryTextProcessor


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    lexical_metrics: Dict[str, float]
    structural_metrics: Dict[str, float]
    readability_metrics: Dict[str, float]
    overall_score: float
    comparison_report: Dict[str, Any]


class QuantitativeEvaluator:
    """Main class for quantitative evaluation of generated poetry."""
    
    def __init__(self):
        """Initialize the evaluator with required analyzers."""
        self.lexical_analyzer = LexicalAnalyzer()
        self.structural_analyzer = StructuralAnalyzer()
        self.text_processor = PoetryTextProcessor()
    
    def calculate_lexical_metrics(self, text: str) -> Dict[str, float]:
        """Calculate lexical richness metrics for poetry text.
        
        Args:
            text: Poetry text to evaluate
            
        Returns:
            Dictionary containing lexical metrics
        """
        if not text or not text.strip():
            return {
                'ttr': 0.0,
                'mattr': 0.0,
                'lexical_density': 0.0,
                'unique_words': 0,
                'total_words': 0,
                'avg_word_length': 0.0
            }
        
        # Get vocabulary richness metrics
        vocab_metrics = self.lexical_analyzer.get_vocabulary_richness_metrics(text)
        
        # Calculate lexical density
        lexical_density = self.lexical_analyzer.calculate_lexical_density(text)
        
        # Get word length statistics
        word_length_stats = self.lexical_analyzer.get_word_length_distribution(text)
        
        return {
            'ttr': vocab_metrics['ttr'],
            'mattr': vocab_metrics['mattr'],
            'lexical_density': lexical_density,
            'unique_words': vocab_metrics['unique_words'],
            'total_words': vocab_metrics['total_words'],
            'avg_word_length': word_length_stats['mean']
        }
    
    def calculate_structural_adherence(self, generated_text: str, 
                                     target_structure: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Calculate structural adherence metrics.
        
        Args:
            generated_text: Generated poetry text
            target_structure: Optional target structure to compare against
            
        Returns:
            Dictionary containing structural adherence metrics
        """
        if not generated_text or not generated_text.strip():
            return {
                'line_count': 0,
                'stanza_count': 0,
                'avg_syllables_per_line': 0.0,
                'syllable_consistency': 0.0,
                'line_count_accuracy': 0.0,
                'syllable_count_accuracy': 0.0
            }
        
        # Process the text
        processed = self.text_processor.preprocess_for_analysis(generated_text)
        lines = processed['lines']
        stanzas = processed['stanzas']
        syllable_counts = processed['syllable_counts_per_line']
        
        # Basic structural metrics
        line_count = len(lines)
        stanza_count = len(stanzas)
        
        # Syllable metrics
        if syllable_counts:
            avg_syllables = sum(syllable_counts) / len(syllable_counts)
            # Calculate syllable consistency (inverse of coefficient of variation)
            if avg_syllables > 0:
                syllable_std = math.sqrt(sum((s - avg_syllables) ** 2 for s in syllable_counts) / len(syllable_counts))
                syllable_consistency = 1.0 - (syllable_std / avg_syllables)
                syllable_consistency = max(0.0, syllable_consistency)  # Ensure non-negative
            else:
                syllable_consistency = 0.0
        else:
            avg_syllables = 0.0
            syllable_consistency = 0.0
        
        # Calculate accuracy metrics if target structure is provided
        line_count_accuracy = 1.0
        syllable_count_accuracy = 1.0
        
        if target_structure:
            if 'expected_line_count' in target_structure:
                expected_lines = target_structure['expected_line_count']
                if expected_lines > 0:
                    line_count_accuracy = 1.0 - abs(line_count - expected_lines) / expected_lines
                    line_count_accuracy = max(0.0, line_count_accuracy)
            
            if 'expected_syllables_per_line' in target_structure:
                expected_syllables = target_structure['expected_syllables_per_line']
                if expected_syllables > 0 and avg_syllables > 0:
                    syllable_count_accuracy = 1.0 - abs(avg_syllables - expected_syllables) / expected_syllables
                    syllable_count_accuracy = max(0.0, syllable_count_accuracy)
        
        return {
            'line_count': line_count,
            'stanza_count': stanza_count,
            'avg_syllables_per_line': avg_syllables,
            'syllable_consistency': syllable_consistency,
            'line_count_accuracy': line_count_accuracy,
            'syllable_count_accuracy': syllable_count_accuracy
        }
    
    def calculate_readability_metrics(self, text: str) -> Dict[str, float]:
        """Calculate readability scores for poetry text.
        
        Args:
            text: Poetry text to evaluate
            
        Returns:
            Dictionary containing readability metrics
        """
        if not text or not text.strip():
            return {
                'flesch_reading_ease': 0.0,
                'flesch_kincaid_grade': 0.0,
                'automated_readability_index': 0.0,
                'avg_sentence_length': 0.0,
                'avg_syllables_per_word': 0.0
            }
        
        # Get basic text statistics
        sentences = self._count_sentences(text)
        words = len(self.lexical_analyzer.get_word_tokens(text))
        syllables = self._count_total_syllables(text)
        
        if sentences == 0 or words == 0:
            return {
                'flesch_reading_ease': 0.0,
                'flesch_kincaid_grade': 0.0,
                'automated_readability_index': 0.0,
                'avg_sentence_length': 0.0,
                'avg_syllables_per_word': 0.0
            }
        
        # Calculate averages
        avg_sentence_length = words / sentences
        avg_syllables_per_word = syllables / words
        
        # Flesch Reading Ease
        flesch_reading_ease = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        # Flesch-Kincaid Grade Level
        flesch_kincaid_grade = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
        
        # Automated Readability Index (ARI)
        characters = len(re.sub(r'\s', '', text))
        if words > 0:
            ari = (4.71 * (characters / words)) + (0.5 * avg_sentence_length) - 21.43
        else:
            ari = 0.0
        
        return {
            'flesch_reading_ease': flesch_reading_ease,
            'flesch_kincaid_grade': flesch_kincaid_grade,
            'automated_readability_index': ari,
            'avg_sentence_length': avg_sentence_length,
            'avg_syllables_per_word': avg_syllables_per_word
        }
    
    def compare_with_target(self, generated_text: str, target_text: str) -> Dict[str, Any]:
        """Compare generated poetry with target poetry.
        
        Args:
            generated_text: Generated poetry text
            target_text: Target poetry text for comparison
            
        Returns:
            Dictionary containing comparison metrics
        """
        # Calculate metrics for both texts
        gen_lexical = self.calculate_lexical_metrics(generated_text)
        target_lexical = self.calculate_lexical_metrics(target_text)
        
        gen_structural = self.calculate_structural_adherence(generated_text)
        target_structural = self.calculate_structural_adherence(target_text)
        
        gen_readability = self.calculate_readability_metrics(generated_text)
        target_readability = self.calculate_readability_metrics(target_text)
        
        # Calculate differences and similarities
        lexical_similarity = self._calculate_metric_similarity(gen_lexical, target_lexical)
        structural_similarity = self._calculate_metric_similarity(gen_structural, target_structural)
        readability_similarity = self._calculate_metric_similarity(gen_readability, target_readability)
        
        # Overall similarity score (weighted average)
        overall_similarity = (
            0.4 * lexical_similarity +
            0.4 * structural_similarity +
            0.2 * readability_similarity
        )
        
        return {
            'generated_metrics': {
                'lexical': gen_lexical,
                'structural': gen_structural,
                'readability': gen_readability
            },
            'target_metrics': {
                'lexical': target_lexical,
                'structural': target_structural,
                'readability': target_readability
            },
            'similarity_scores': {
                'lexical_similarity': lexical_similarity,
                'structural_similarity': structural_similarity,
                'readability_similarity': readability_similarity,
                'overall_similarity': overall_similarity
            }
        }
    
    def evaluate_poetry(self, generated_text: str, 
                       target_text: Optional[str] = None,
                       target_structure: Optional[Dict[str, Any]] = None) -> EvaluationResult:
        """Comprehensive evaluation of generated poetry.
        
        Args:
            generated_text: Generated poetry text to evaluate
            target_text: Optional target text for comparison
            target_structure: Optional target structure specifications
            
        Returns:
            EvaluationResult containing all metrics and scores
        """
        # Calculate individual metrics
        lexical_metrics = self.calculate_lexical_metrics(generated_text)
        structural_metrics = self.calculate_structural_adherence(generated_text, target_structure)
        readability_metrics = self.calculate_readability_metrics(generated_text)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(lexical_metrics, structural_metrics, readability_metrics)
        
        # Generate comparison report if target text is provided
        comparison_report = {}
        if target_text:
            comparison_report = self.compare_with_target(generated_text, target_text)
        
        return EvaluationResult(
            lexical_metrics=lexical_metrics,
            structural_metrics=structural_metrics,
            readability_metrics=readability_metrics,
            overall_score=overall_score,
            comparison_report=comparison_report
        )
    
    def _count_sentences(self, text: str) -> int:
        """Count sentences in text."""
        if not text:
            return 0
        
        # Count sentence-ending punctuation, considering poetry line breaks
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # If no sentence-ending punctuation, treat each line as a sentence
        if len(sentences) <= 1:
            lines = self.text_processor.segment_lines(text)
            return len(lines) if lines else 1
        
        return len(sentences)
    
    def _count_total_syllables(self, text: str) -> int:
        """Count total syllables in text."""
        if not text:
            return 0
        
        lines = self.text_processor.segment_lines(text)
        return sum(self.text_processor.count_syllables_line(line) for line in lines)
    
    def _calculate_metric_similarity(self, metrics1: Dict[str, float], 
                                   metrics2: Dict[str, float]) -> float:
        """Calculate similarity between two metric dictionaries.
        
        Args:
            metrics1: First set of metrics
            metrics2: Second set of metrics
            
        Returns:
            Similarity score between 0 and 1
        """
        if not metrics1 or not metrics2:
            return 0.0
        
        # Find common metrics
        common_keys = set(metrics1.keys()) & set(metrics2.keys())
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            val1, val2 = metrics1[key], metrics2[key]
            
            # Skip non-numeric values
            if not isinstance(val1, (int, float)) or not isinstance(val2, (int, float)):
                continue
            
            # Calculate similarity based on relative difference
            if val1 == 0 and val2 == 0:
                similarity = 1.0
            elif val1 == 0 or val2 == 0:
                similarity = 0.0
            else:
                max_val = max(abs(val1), abs(val2))
                diff = abs(val1 - val2)
                similarity = 1.0 - (diff / max_val)
                similarity = max(0.0, similarity)
            
            similarities.append(similarity)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _calculate_overall_score(self, lexical_metrics: Dict[str, float],
                               structural_metrics: Dict[str, float],
                               readability_metrics: Dict[str, float]) -> float:
        """Calculate overall quality score.
        
        Args:
            lexical_metrics: Lexical analysis results
            structural_metrics: Structural analysis results
            readability_metrics: Readability analysis results
            
        Returns:
            Overall score between 0 and 1
        """
        scores = []
        
        # Lexical quality score
        if lexical_metrics.get('ttr', 0) > 0:
            lexical_score = min(1.0, lexical_metrics['ttr'] * 2)  # TTR typically 0-0.5
            scores.append(lexical_score)
        
        # Structural consistency score
        if 'syllable_consistency' in structural_metrics:
            scores.append(structural_metrics['syllable_consistency'])
        
        # Readability score (normalized Flesch Reading Ease)
        if 'flesch_reading_ease' in readability_metrics:
            flesch = readability_metrics['flesch_reading_ease']
            # Normalize Flesch score (0-100) to 0-1
            readability_score = max(0.0, min(1.0, flesch / 100))
            scores.append(readability_score)
        
        return sum(scores) / len(scores) if scores else 0.0


# Convenience functions for direct use
def evaluate_generated_poetry(generated_text: str, target_text: Optional[str] = None) -> EvaluationResult:
    """Evaluate generated poetry using default evaluator."""
    evaluator = QuantitativeEvaluator()
    return evaluator.evaluate_poetry(generated_text, target_text)


def calculate_ttr_and_density(text: str) -> Tuple[float, float]:
    """Calculate TTR and lexical density for text."""
    evaluator = QuantitativeEvaluator()
    metrics = evaluator.calculate_lexical_metrics(text)
    return metrics['ttr'], metrics['lexical_density']


def measure_structural_adherence(generated_text: str, expected_lines: int, 
                               expected_syllables: int) -> Dict[str, float]:
    """Measure structural adherence to expected format."""
    evaluator = QuantitativeEvaluator()
    target_structure = {
        'expected_line_count': expected_lines,
        'expected_syllables_per_line': expected_syllables
    }
    return evaluator.calculate_structural_adherence(generated_text, target_structure)


def calculate_readability_scores(text: str) -> Dict[str, float]:
    """Calculate readability scores for text."""
    evaluator = QuantitativeEvaluator()
    return evaluator.calculate_readability_metrics(text)