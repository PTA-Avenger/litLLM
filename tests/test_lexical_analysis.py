"""
Unit tests for lexical analysis utilities.
"""

import pytest
from src.stylometric.lexical_analysis import (
    LexicalAnalyzer,
    calculate_type_token_ratio,
    get_pos_tags,
    analyze_vocabulary_richness
)


class TestLexicalAnalyzer:
    """Test cases for LexicalAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = LexicalAnalyzer()
        
        # Sample texts for testing
        self.simple_text = "The cat sat on the mat. The dog ran in the park."
        self.repeated_text = "The the the cat cat sat sat sat on on the mat"
        self.complex_text = """I wandered lonely as a cloud
That floats on high o'er vales and hills,
When all at once I saw a crowd,
A host, of golden daffodils.""" 
       
    def test_get_word_tokens_basic(self):
        """Test basic word tokenization."""
        tokens = self.analyzer.get_word_tokens(self.simple_text)
        expected_count = 12  # "The cat sat on the mat The dog ran in the park"
        assert len(tokens) == expected_count
        assert "cat" in tokens
        assert "park" in tokens
        
    def test_calculate_ttr_basic(self):
        """Test basic TTR calculation."""
        # Simple case: all unique words
        unique_text = "cat dog bird fish"
        ttr_unique = self.analyzer.calculate_ttr(unique_text)
        assert ttr_unique == 1.0
        
        # Case with repetition
        ttr_repeated = self.analyzer.calculate_ttr(self.repeated_text)
        assert 0 < ttr_repeated < 1.0
        
    def test_calculate_ttr_case_sensitivity(self):
        """Test TTR with case sensitivity."""
        text = "The the THE cat Cat CAT"
        
        ttr_case_sensitive = self.analyzer.calculate_ttr(text, case_sensitive=True)
        ttr_case_insensitive = self.analyzer.calculate_ttr(text, case_sensitive=False)
        
        assert ttr_case_sensitive > ttr_case_insensitive
        
    def test_calculate_ttr_empty_input(self):
        """Test TTR with empty input."""
        assert self.analyzer.calculate_ttr("") == 0.0
        assert self.analyzer.calculate_ttr("   ") == 0.0
        
    def test_calculate_lexical_density(self):
        """Test lexical density calculation."""
        # Text with mostly content words
        content_heavy = "Beautiful flowers bloom magnificently"
        density_high = self.analyzer.calculate_lexical_density(content_heavy)
        
        # Text with many function words
        function_heavy = "The and the or the but the"
        density_low = self.analyzer.calculate_lexical_density(function_heavy)
        
        assert density_high > density_low
        assert 0 <= density_high <= 1.0
        assert 0 <= density_low <= 1.0    
    
    def test_get_pos_distribution(self):
        """Test POS distribution calculation."""
        pos_dist = self.analyzer.get_pos_distribution(self.simple_text)
        
        # Check that it returns a dictionary
        assert isinstance(pos_dist, dict)
        
        # Check that probabilities sum to approximately 1
        total_prob = sum(pos_dist.values())
        assert abs(total_prob - 1.0) < 0.001
        
        # Check that common POS tags are present
        pos_tags = set(pos_dist.keys())
        assert len(pos_tags) > 0
        
    def test_get_vocabulary_richness_metrics(self):
        """Test comprehensive vocabulary richness metrics."""
        richness = self.analyzer.get_vocabulary_richness_metrics(self.complex_text)
        
        # Check all expected keys are present
        expected_keys = ['ttr', 'mattr', 'mtld', 'hdd', 'unique_words', 'total_words']
        for key in expected_keys:
            assert key in richness
            
        # Check that values are reasonable
        assert 0 <= richness['ttr'] <= 1.0
        assert 0 <= richness['mattr'] <= 1.0
        assert richness['unique_words'] > 0
        assert richness['total_words'] > 0
        assert richness['unique_words'] <= richness['total_words']
        
    def test_analyze_lexical_features_comprehensive(self):
        """Test comprehensive lexical analysis."""
        features = self.analyzer.analyze_lexical_features(self.complex_text)
        
        # Check all main feature categories
        expected_categories = [
            'vocabulary_richness', 'lexical_density', 'pos_distribution',
            'word_length_stats', 'sentence_length_stats', 'total_tokens', 'total_sentences'
        ]
        
        for category in expected_categories:
            assert category in features
            
        # Check that numerical values are reasonable
        assert features['total_tokens'] > 0
        assert features['total_sentences'] > 0
        assert 0 <= features['lexical_density'] <= 1.0


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    def test_calculate_type_token_ratio(self):
        """Test convenience function for TTR calculation."""
        text = "cat dog bird cat"
        ttr = calculate_type_token_ratio(text)
        assert ttr == 0.75  # 3 unique words out of 4 total
        
    def test_get_pos_tags(self):
        """Test convenience function for POS tagging."""
        text = "The cat sat"
        pos_tags = get_pos_tags(text)
        
        assert len(pos_tags) == 3
        assert all(isinstance(tag, tuple) and len(tag) == 2 for tag in pos_tags)
        
        # Check that we get expected word-tag pairs
        words = [word for word, tag in pos_tags]
        assert "The" in words
        assert "cat" in words
        assert "sat" in words