"""
Unit tests for text processing utilities.
"""

import pytest
from src.stylometric.text_processing import (
    PoetryTextProcessor,
    clean_poetry_text,
    segment_poem_lines,
    segment_poem_stanzas,
    count_syllables
)


class TestPoetryTextProcessor:
    """Test cases for PoetryTextProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = PoetryTextProcessor()
        
        # Sample Emily Dickinson poem for testing
        self.sample_poem = """I'm Nobody! Who are you?
Are you Nobody too?
Then there's a pair of us — don't tell!
They'd banish us, you know.

How dreary to be Somebody!
How public, like a Frog
To tell your name the livelong day
To an admiring Bog!"""

        # Simple test cases
        self.simple_line = "The cat sat on the mat"
        self.complex_line = "I wandered lonely as a cloud"   
     
    def test_clean_text_basic(self):
        """Test basic text cleaning functionality."""
        dirty_text = "  \n\n  Hello   world  \n\n  "
        expected = "Hello world"
        assert self.processor.clean_text(dirty_text) == expected
        
    def test_clean_text_preserves_line_structure(self):
        """Test that cleaning preserves important line breaks."""
        text_with_lines = "Line one\n\nLine two\nLine three"
        cleaned = self.processor.clean_text(text_with_lines)
        lines = cleaned.split('\n')
        assert len(lines) == 4  # Including empty line
        assert lines[0] == "Line one"
        assert lines[1] == ""
        assert lines[2] == "Line two"
        assert lines[3] == "Line three"
        
    def test_clean_text_empty_input(self):
        """Test cleaning with empty input."""
        assert self.processor.clean_text("") == ""
        assert self.processor.clean_text("   \n\n   ") == ""
        
    def test_segment_lines_basic(self):
        """Test basic line segmentation."""
        lines = self.processor.segment_lines(self.sample_poem)
        assert len(lines) == 8
        assert lines[0] == "I'm Nobody! Who are you?"
        assert lines[1] == "Are you Nobody too?"
        
    def test_segment_lines_filters_empty(self):
        """Test that line segmentation filters out empty lines."""
        text_with_empty = "Line 1\n\n\nLine 2\n\nLine 3"
        lines = self.processor.segment_lines(text_with_empty)
        assert len(lines) == 3
        assert lines == ["Line 1", "Line 2", "Line 3"]
        
    def test_segment_stanzas_basic(self):
        """Test basic stanza segmentation."""
        stanzas = self.processor.segment_stanzas(self.sample_poem)
        assert len(stanzas) == 2
        assert len(stanzas[0]) == 4  # First stanza has 4 lines
        assert len(stanzas[1]) == 4  # Second stanza has 4 lines
        
    def test_segment_stanzas_single_stanza(self):
        """Test stanza segmentation with single stanza."""
        single_stanza = "Line 1\nLine 2\nLine 3"
        stanzas = self.processor.segment_stanzas(single_stanza)
        assert len(stanzas) == 1
        assert len(stanzas[0]) == 3   
     
    def test_count_syllables_word_basic(self):
        """Test basic syllable counting for individual words."""
        test_cases = [
            ("cat", 1),
            ("hello", 2),
            ("beautiful", 3),
            ("university", 4),
            ("", 0),
        ]
        
        for word, expected in test_cases:
            result = self.processor.count_syllables_word(word)
            assert result == expected, f"Word '{word}' should have {expected} syllables, got {result}"
            
    def test_count_syllables_word_with_punctuation(self):
        """Test syllable counting with punctuation."""
        assert self.processor.count_syllables_word("don't") == 1
        assert self.processor.count_syllables_word("world!") == 1
        assert self.processor.count_syllables_word("beautiful,") == 3
        
    def test_count_syllables_line_basic(self):
        """Test syllable counting for lines."""
        test_cases = [
            ("The cat sat", 3),  # 1 + 1 + 1
            ("I wandered lonely", 5),  # 1 + 2 + 2
            ("", 0),
            ("Hello beautiful world", 6),  # 2 + 3 + 1
        ]
        
        for line, expected in test_cases:
            result = self.processor.count_syllables_line(line)
            assert result == expected, f"Line '{line}' should have {expected} syllables, got {result}"
            
    def test_detect_line_boundaries(self):
        """Test line boundary detection."""
        text = "Line 1\nLine 2\n\nLine 3"
        boundaries = self.processor.detect_line_boundaries(text)
        assert len(boundaries) == 3  # Empty lines are excluded
        assert boundaries[0] == (0, 6)  # "Line 1"
        assert boundaries[1] == (7, 13)  # "Line 2"
        
    def test_get_word_tokens_basic(self):
        """Test word tokenization."""
        text = "Hello, beautiful world! Don't you think?"
        tokens = self.processor.get_word_tokens(text)
        expected = ["Hello", "beautiful", "world", "Don't", "you", "think"]
        assert tokens == expected        

    def test_get_word_tokens_preserves_case(self):
        """Test that tokenization preserves case."""
        text = "I'm Nobody! Who are you?"
        tokens = self.processor.get_word_tokens(text)
        assert "I'm" in tokens
        assert "Nobody" in tokens
        assert "Who" in tokens
        
    def test_preprocess_for_analysis_complete(self):
        """Test complete preprocessing pipeline."""
        result = self.processor.preprocess_for_analysis(self.sample_poem)
        
        # Check all expected keys are present
        expected_keys = [
            'cleaned_text', 'lines', 'stanzas', 'line_boundaries', 
            'word_tokens', 'syllable_counts_per_line'
        ]
        for key in expected_keys:
            assert key in result
            
        # Check data integrity
        assert len(result['lines']) == 8
        assert len(result['stanzas']) == 2
        assert len(result['syllable_counts_per_line']) == 8
        assert len(result['word_tokens']) > 0


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    def test_clean_poetry_text(self):
        """Test convenience function for text cleaning."""
        dirty_text = "  Hello   world  \n\n  "
        cleaned = clean_poetry_text(dirty_text)
        assert cleaned == "Hello world"
        
    def test_segment_poem_lines(self):
        """Test convenience function for line segmentation."""
        poem = "Line 1\nLine 2\n\nLine 3"
        lines = segment_poem_lines(poem)
        assert lines == ["Line 1", "Line 2", "Line 3"]
        
    def test_segment_poem_stanzas(self):
        """Test convenience function for stanza segmentation."""
        poem = "Line 1\nLine 2\n\nLine 3\nLine 4"
        stanzas = segment_poem_stanzas(poem)
        assert len(stanzas) == 2
        assert stanzas[0] == ["Line 1", "Line 2"]
        assert stanzas[1] == ["Line 3", "Line 4"]
        
    def test_count_syllables_single_line(self):
        """Test convenience function for syllable counting - single line."""
        line = "The cat sat on the mat"
        syllables = count_syllables(line)
        assert syllables == 6  # 1+1+1+1+1+1
        
    def test_count_syllables_multi_line(self):
        """Test convenience function for syllable counting - multi-line."""
        poem = "The cat sat\nOn the mat"
        syllables = count_syllables(poem)
        assert syllables == 6  # 3 + 3