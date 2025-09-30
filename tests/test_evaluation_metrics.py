"""
Unit tests for evaluation metrics module.

Tests the quantitative evaluation metrics for generated poetry,
ensuring consistency and accuracy of calculations.
"""

import unittest
import math
from src.stylometric.evaluation_metrics import (
    QuantitativeEvaluator,
    EvaluationResult,
    evaluate_generated_poetry,
    calculate_ttr_and_density,
    measure_structural_adherence,
    calculate_readability_scores
)


class TestQuantitativeEvaluator(unittest.TestCase):
    """Test cases for QuantitativeEvaluator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = QuantitativeEvaluator()
        
        # Sample poetry texts for testing
        self.simple_poem = """Roses are red
Violets are blue
Sugar is sweet
And so are you"""
        
        self.complex_poem = """I wandered lonely as a cloud
That floats on high o'er vales and hills,
When all at once I saw a crowd,
A host, of golden daffodils;
Beside the lake, beneath the trees,
Fluttering and dancing in the breeze."""
        
        self.empty_text = ""
        self.single_line = "This is just one line of text"
    
    def test_calculate_lexical_metrics_simple_poem(self):
        """Test lexical metrics calculation for simple poem."""
        metrics = self.evaluator.calculate_lexical_metrics(self.simple_poem)
        
        # Check that all expected keys are present
        expected_keys = ['ttr', 'mattr', 'lexical_density', 'unique_words', 'total_words', 'avg_word_length']
        for key in expected_keys:
            self.assertIn(key, metrics)
        
        # Check reasonable values
        self.assertGreater(metrics['ttr'], 0)
        self.assertLessEqual(metrics['ttr'], 1.0)
        self.assertGreater(metrics['total_words'], 0)
        self.assertGreater(metrics['unique_words'], 0)
        self.assertLessEqual(metrics['unique_words'], metrics['total_words'])
        self.assertGreater(metrics['avg_word_length'], 0)
    
    def test_calculate_lexical_metrics_empty_text(self):
        """Test lexical metrics calculation for empty text."""
        metrics = self.evaluator.calculate_lexical_metrics(self.empty_text)
        
        # All metrics should be zero for empty text
        self.assertEqual(metrics['ttr'], 0.0)
        self.assertEqual(metrics['total_words'], 0)
        self.assertEqual(metrics['unique_words'], 0)
        self.assertEqual(metrics['avg_word_length'], 0.0)
    
    def test_calculate_lexical_metrics_consistency(self):
        """Test that lexical metrics are consistent across multiple calls."""
        metrics1 = self.evaluator.calculate_lexical_metrics(self.complex_poem)
        metrics2 = self.evaluator.calculate_lexical_metrics(self.complex_poem)
        
        # Results should be identical
        for key in metrics1:
            self.assertEqual(metrics1[key], metrics2[key])
    
    def test_calculate_structural_adherence_basic(self):
        """Test structural adherence calculation."""
        metrics = self.evaluator.calculate_structural_adherence(self.simple_poem)
        
        # Check expected keys
        expected_keys = ['line_count', 'stanza_count', 'avg_syllables_per_line', 
                        'syllable_consistency', 'line_count_accuracy', 'syllable_count_accuracy']
        for key in expected_keys:
            self.assertIn(key, metrics)
        
        # Check reasonable values
        self.assertEqual(metrics['line_count'], 4)  # Simple poem has 4 lines
        self.assertGreater(metrics['avg_syllables_per_line'], 0)
        self.assertGreaterEqual(metrics['syllable_consistency'], 0)
        self.assertLessEqual(metrics['syllable_consistency'], 1.0)
    
    def test_calculate_structural_adherence_with_target(self):
        """Test structural adherence with target structure."""
        target_structure = {
            'expected_line_count': 4,
            'expected_syllables_per_line': 6
        }
        
        metrics = self.evaluator.calculate_structural_adherence(self.simple_poem, target_structure)
        
        # Should have accuracy scores
        self.assertIn('line_count_accuracy', metrics)
        self.assertIn('syllable_count_accuracy', metrics)
        self.assertGreaterEqual(metrics['line_count_accuracy'], 0)
        self.assertLessEqual(metrics['line_count_accuracy'], 1.0)
    
    def test_calculate_structural_adherence_empty_text(self):
        """Test structural adherence for empty text."""
        metrics = self.evaluator.calculate_structural_adherence(self.empty_text)
        
        # All counts should be zero
        self.assertEqual(metrics['line_count'], 0)
        self.assertEqual(metrics['stanza_count'], 0)
        self.assertEqual(metrics['avg_syllables_per_line'], 0.0)
    
    def test_calculate_readability_metrics_basic(self):
        """Test readability metrics calculation."""
        metrics = self.evaluator.calculate_readability_metrics(self.complex_poem)
        
        # Check expected keys
        expected_keys = ['flesch_reading_ease', 'flesch_kincaid_grade', 
                        'automated_readability_index', 'avg_sentence_length', 'avg_syllables_per_word']
        for key in expected_keys:
            self.assertIn(key, metrics)
        
        # Check reasonable values
        self.assertGreater(metrics['avg_sentence_length'], 0)
        self.assertGreater(metrics['avg_syllables_per_word'], 0)
        # Flesch scores can be negative, so just check they're numeric
        self.assertIsInstance(metrics['flesch_reading_ease'], (int, float))
        self.assertIsInstance(metrics['flesch_kincaid_grade'], (int, float))
    
    def test_calculate_readability_metrics_empty_text(self):
        """Test readability metrics for empty text."""
        metrics = self.evaluator.calculate_readability_metrics(self.empty_text)
        
        # All metrics should be zero for empty text
        for key in metrics:
            self.assertEqual(metrics[key], 0.0)
    
    def test_compare_with_target_identical_texts(self):
        """Test comparison with identical texts."""
        comparison = self.evaluator.compare_with_target(self.simple_poem, self.simple_poem)
        
        # Check structure
        self.assertIn('generated_metrics', comparison)
        self.assertIn('target_metrics', comparison)
        self.assertIn('similarity_scores', comparison)
        
        # Similarity should be perfect (1.0) for identical texts
        similarities = comparison['similarity_scores']
        self.assertEqual(similarities['overall_similarity'], 1.0)
        self.assertEqual(similarities['lexical_similarity'], 1.0)
        self.assertEqual(similarities['structural_similarity'], 1.0)
        self.assertEqual(similarities['readability_similarity'], 1.0)
    
    def test_compare_with_target_different_texts(self):
        """Test comparison with different texts."""
        comparison = self.evaluator.compare_with_target(self.simple_poem, self.complex_poem)
        
        # Similarity should be less than 1.0 for different texts
        similarities = comparison['similarity_scores']
        self.assertLess(similarities['overall_similarity'], 1.0)
        self.assertGreaterEqual(similarities['overall_similarity'], 0.0)
    
    def test_evaluate_poetry_comprehensive(self):
        """Test comprehensive poetry evaluation."""
        result = self.evaluator.evaluate_poetry(self.complex_poem)
        
        # Check result type
        self.assertIsInstance(result, EvaluationResult)
        
        # Check that all metric categories are present
        self.assertIsInstance(result.lexical_metrics, dict)
        self.assertIsInstance(result.structural_metrics, dict)
        self.assertIsInstance(result.readability_metrics, dict)
        self.assertIsInstance(result.overall_score, (int, float))
        
        # Overall score should be between 0 and 1
        self.assertGreaterEqual(result.overall_score, 0.0)
        self.assertLessEqual(result.overall_score, 1.0)
    
    def test_evaluate_poetry_with_target(self):
        """Test poetry evaluation with target text."""
        result = self.evaluator.evaluate_poetry(self.simple_poem, target_text=self.complex_poem)
        
        # Should include comparison report
        self.assertIsInstance(result.comparison_report, dict)
        self.assertIn('similarity_scores', result.comparison_report)
    
    def test_metric_similarity_calculation(self):
        """Test metric similarity calculation method."""
        metrics1 = {'ttr': 0.5, 'density': 0.7, 'length': 10}
        metrics2 = {'ttr': 0.6, 'density': 0.7, 'length': 12}
        
        similarity = self.evaluator._calculate_metric_similarity(metrics1, metrics2)
        
        # Should be between 0 and 1
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)
    
    def test_metric_similarity_identical_metrics(self):
        """Test metric similarity with identical metrics."""
        metrics = {'ttr': 0.5, 'density': 0.7, 'length': 10}
        
        similarity = self.evaluator._calculate_metric_similarity(metrics, metrics)
        
        # Should be perfect similarity
        self.assertEqual(similarity, 1.0)
    
    def test_metric_similarity_empty_metrics(self):
        """Test metric similarity with empty metrics."""
        similarity = self.evaluator._calculate_metric_similarity({}, {})
        
        # Should return 0 for empty metrics
        self.assertEqual(similarity, 0.0)
    
    def test_overall_score_calculation(self):
        """Test overall score calculation."""
        lexical = {'ttr': 0.5, 'mattr': 0.4}
        structural = {'syllable_consistency': 0.8}
        readability = {'flesch_reading_ease': 60.0}
        
        score = self.evaluator._calculate_overall_score(lexical, structural, readability)
        
        # Should be between 0 and 1
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestConvenienceFunctions(unittest.TestCase):
    """Test cases for convenience functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_poem = """The cat sat on the mat
        The dog ran in the park
        Birds sing in the trees
        Flowers bloom in spring"""
    
    def test_evaluate_generated_poetry(self):
        """Test evaluate_generated_poetry convenience function."""
        result = evaluate_generated_poetry(self.test_poem)
        
        self.assertIsInstance(result, EvaluationResult)
        self.assertGreaterEqual(result.overall_score, 0.0)
        self.assertLessEqual(result.overall_score, 1.0)
    
    def test_calculate_ttr_and_density(self):
        """Test calculate_ttr_and_density convenience function."""
        ttr, density = calculate_ttr_and_density(self.test_poem)
        
        self.assertIsInstance(ttr, float)
        self.assertIsInstance(density, float)
        self.assertGreater(ttr, 0)
        self.assertGreater(density, 0)
        self.assertLessEqual(ttr, 1.0)
        self.assertLessEqual(density, 1.0)
    
    def test_measure_structural_adherence(self):
        """Test measure_structural_adherence convenience function."""
        metrics = measure_structural_adherence(self.test_poem, expected_lines=4, expected_syllables=8)
        
        self.assertIn('line_count_accuracy', metrics)
        self.assertIn('syllable_count_accuracy', metrics)
        self.assertGreaterEqual(metrics['line_count_accuracy'], 0.0)
        self.assertLessEqual(metrics['line_count_accuracy'], 1.0)
    
    def test_calculate_readability_scores(self):
        """Test calculate_readability_scores convenience function."""
        scores = calculate_readability_scores(self.test_poem)
        
        expected_keys = ['flesch_reading_ease', 'flesch_kincaid_grade', 
                        'automated_readability_index', 'avg_sentence_length', 'avg_syllables_per_word']
        for key in expected_keys:
            self.assertIn(key, scores)
            self.assertIsInstance(scores[key], (int, float))


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = QuantitativeEvaluator()
    
    def test_single_word_text(self):
        """Test evaluation with single word."""
        result = self.evaluator.evaluate_poetry("Hello")
        
        self.assertIsInstance(result, EvaluationResult)
        self.assertEqual(result.lexical_metrics['total_words'], 1)
        self.assertEqual(result.lexical_metrics['unique_words'], 1)
        self.assertEqual(result.lexical_metrics['ttr'], 1.0)
    
    def test_repeated_words_text(self):
        """Test evaluation with repeated words."""
        text = "the the the the"
        result = self.evaluator.evaluate_poetry(text)
        
        self.assertEqual(result.lexical_metrics['total_words'], 4)
        self.assertEqual(result.lexical_metrics['unique_words'], 1)
        self.assertEqual(result.lexical_metrics['ttr'], 0.25)
    
    def test_punctuation_heavy_text(self):
        """Test evaluation with heavy punctuation."""
        text = "Hello! How are you? I'm fine... Really, truly fine!!!"
        result = self.evaluator.evaluate_poetry(text)
        
        # Should handle punctuation gracefully
        self.assertGreater(result.lexical_metrics['total_words'], 0)
        self.assertGreaterEqual(result.overall_score, 0.0)
    
    def test_very_long_text(self):
        """Test evaluation with very long text."""
        # Create a long text by repeating a poem
        base_poem = "Roses are red, violets are blue\n"
        long_text = base_poem * 100
        
        result = self.evaluator.evaluate_poetry(long_text)
        
        # Should handle long text without errors
        self.assertIsInstance(result, EvaluationResult)
        self.assertGreater(result.lexical_metrics['total_words'], 0)
    
    def test_unicode_text(self):
        """Test evaluation with unicode characters."""
        text = "Café résumé naïve Zürich"
        result = self.evaluator.evaluate_poetry(text)
        
        # Should handle unicode gracefully
        self.assertGreater(result.lexical_metrics['total_words'], 0)
        self.assertGreaterEqual(result.overall_score, 0.0)
    
    def test_whitespace_only_text(self):
        """Test evaluation with whitespace-only text."""
        text = "   \n\t  \n  "
        result = self.evaluator.evaluate_poetry(text)
        
        # Should treat as empty text
        self.assertEqual(result.lexical_metrics['total_words'], 0)
        self.assertEqual(result.overall_score, 0.0)


if __name__ == '__main__':
    unittest.main()