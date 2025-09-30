"""Unit tests for structural analysis utilities."""

import pytest
from src.stylometric.structural_analysis import StructuralAnalyzer


class TestStructuralAnalyzer:
    """Test cases for StructuralAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = StructuralAnalyzer()
        
    def test_analyze_meter_basic_empty_line(self):
        """Test meter analysis with empty line."""
        result = self.analyzer.analyze_meter_basic("")
        
        assert result['syllable_count'] == 0
        assert result['meter_type'] == 'unknown'
        
    def test_analyze_meter_basic_simple_line(self):
        """Test meter analysis with simple line."""
        line = "The cat sat on the mat"
        result = self.analyzer.analyze_meter_basic(line)
        
        assert result['syllable_count'] == 6
        assert result['meter_type'] in ['short_meter', 'free_verse']
        
    def test_analyze_rhyme_scheme_empty_stanza(self):
        """Test rhyme scheme analysis with empty stanza."""
        result = self.analyzer.analyze_rhyme_scheme([])
        
        assert result['rhyme_scheme'] == ''
        assert result['rhyme_type'] == 'none'
        
    def test_words_rhyme_simple(self):
        """Test simple rhyme detection."""
        assert self.analyzer._words_rhyme('cat', 'hat') == True
        assert self.analyzer._words_rhyme('cat', 'dog') == False
        assert self.analyzer._words_rhyme('', '') == True
        
    def test_analyze_structural_features_comprehensive(self):
        """Test comprehensive structural analysis."""
        poem = "Roses are red\nViolets are blue\nSugar is sweet\nAnd so are you"
        result = self.analyzer.analyze_structural_features(poem)
        
        expected_keys = ['meter_analysis', 'rhyme_analysis', 'total_lines', 'total_stanzas']
        for key in expected_keys:
            assert key in result
            
        assert result['total_lines'] == 4
        assert result['total_stanzas'] == 1