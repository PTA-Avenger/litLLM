"""
Text processing utilities for poetry analysis.

This module provides functions for cleaning, preprocessing, and segmenting poetry text
for stylometric analysis.
"""

import re
import string
from typing import List, Tuple, Dict, Optional
import nltk
import pyphen

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/cmudict')
except LookupError:
    nltk.download('cmudict')


class PoetryTextProcessor:
    """Main class for processing poetry text."""
    
    def __init__(self, language: str = 'en_US'):
        """Initialize the text processor.
        
        Args:
            language: Language code for syllable counting (default: 'en_US')
        """
        self.language = language
        self.syllable_dict = pyphen.Pyphen(lang=language)
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize poetry text.
        
        Args:
            text: Raw poetry text
            
        Returns:
            Cleaned text with normalized whitespace and preserved line breaks
        """
        if not text:
            return ""
            
        # Remove excessive whitespace while preserving line structure
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Strip leading/trailing whitespace but preserve internal spacing
            cleaned_line = re.sub(r'\s+', ' ', line.strip())
            cleaned_lines.append(cleaned_line)
        
        # Remove empty lines at start and end, but preserve internal empty lines
        while cleaned_lines and not cleaned_lines[0]:
            cleaned_lines.pop(0)
        while cleaned_lines and not cleaned_lines[-1]:
            cleaned_lines.pop()
            
        return '\n'.join(cleaned_lines)
    
    def segment_lines(self, text: str) -> List[str]:
        """Segment poem into individual lines.
        
        Args:
            text: Poetry text
            
        Returns:
            List of individual lines (non-empty)
        """
        if not text:
            return []
            
        lines = text.split('\n')
        # Filter out empty lines
        return [line.strip() for line in lines if line.strip()]
    
    def segment_stanzas(self, text: str) -> List[List[str]]:
        """Segment poem into stanzas (groups of lines separated by blank lines).
        
        Args:
            text: Poetry text
            
        Returns:
            List of stanzas, where each stanza is a list of lines
        """
        if not text:
            return []
            
        lines = text.split('\n')
        stanzas = []
        current_stanza = []
        
        for line in lines:
            stripped_line = line.strip()
            if stripped_line:
                current_stanza.append(stripped_line)
            else:
                # Empty line indicates stanza break
                if current_stanza:
                    stanzas.append(current_stanza)
                    current_stanza = []
        
        # Add the last stanza if it exists
        if current_stanza:
            stanzas.append(current_stanza)
            
        return stanzas
    
    def count_syllables_word(self, word: str) -> int:
        """Count syllables in a single word using pyphen.
        
        Args:
            word: Single word to analyze
            
        Returns:
            Number of syllables in the word
        """
        if not word:
            return 0
            
        # Remove punctuation and convert to lowercase
        clean_word = re.sub(r'[^\w]', '', word.lower())
        if not clean_word:
            return 0
            
        # Use pyphen to get syllable breaks
        syllables = self.syllable_dict.inserted(clean_word)
        if syllables:
            # Count the number of hyphens + 1 to get syllable count
            return syllables.count('-') + 1
        else:
            # Fallback: simple vowel counting heuristic
            return self._count_syllables_fallback(clean_word)
    
    def _count_syllables_fallback(self, word: str) -> int:
        """Fallback syllable counting using vowel patterns.
        
        Args:
            word: Clean word (lowercase, no punctuation)
            
        Returns:
            Estimated syllable count
        """
        if not word:
            return 0
            
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
        
        # Handle silent 'e' at the end
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
            
        # Ensure at least one syllable
        return max(1, syllable_count)
    
    def count_syllables_line(self, line: str) -> int:
        """Count total syllables in a line of poetry.
        
        Args:
            line: Line of poetry text
            
        Returns:
            Total syllable count for the line
        """
        if not line:
            return 0
            
        # Split into words and count syllables for each
        words = re.findall(r'\b\w+\b', line)
        return sum(self.count_syllables_word(word) for word in words)
    
    def detect_line_boundaries(self, text: str) -> List[Tuple[int, int]]:
        """Detect line boundaries in text and return their positions.
        
        Args:
            text: Poetry text
            
        Returns:
            List of (start, end) positions for each line
        """
        if not text:
            return []
            
        boundaries = []
        lines = text.split('\n')
        current_pos = 0
        
        for line in lines:
            if line.strip():  # Only include non-empty lines
                start = current_pos
                end = current_pos + len(line)
                boundaries.append((start, end))
            current_pos += len(line) + 1  # +1 for the newline character
            
        return boundaries
    
    def get_word_tokens(self, text: str) -> List[str]:
        """Extract word tokens from text, preserving case.
        
        Args:
            text: Input text
            
        Returns:
            List of word tokens
        """
        if not text:
            return []
            
        # Use regex to find word tokens (letters, numbers, apostrophes)
        return re.findall(r"\b\w+(?:'\w+)?\b", text)
    
    def preprocess_for_analysis(self, text: str) -> Dict[str, any]:
        """Complete preprocessing pipeline for stylometric analysis.
        
        Args:
            text: Raw poetry text
            
        Returns:
            Dictionary containing all processed elements
        """
        cleaned_text = self.clean_text(text)
        
        return {
            'cleaned_text': cleaned_text,
            'lines': self.segment_lines(cleaned_text),
            'stanzas': self.segment_stanzas(cleaned_text),
            'line_boundaries': self.detect_line_boundaries(cleaned_text),
            'word_tokens': self.get_word_tokens(cleaned_text),
            'syllable_counts_per_line': [
                self.count_syllables_line(line) 
                for line in self.segment_lines(cleaned_text)
            ]
        }


# Convenience functions for direct use
def clean_poetry_text(text: str) -> str:
    """Clean poetry text using default processor."""
    processor = PoetryTextProcessor()
    return processor.clean_text(text)


def segment_poem_lines(text: str) -> List[str]:
    """Segment poem into lines using default processor."""
    processor = PoetryTextProcessor()
    return processor.segment_lines(text)


def segment_poem_stanzas(text: str) -> List[List[str]]:
    """Segment poem into stanzas using default processor."""
    processor = PoetryTextProcessor()
    return processor.segment_stanzas(text)


def count_syllables(text: str) -> int:
    """Count syllables in text using default processor."""
    processor = PoetryTextProcessor()
    if '\n' in text:
        # Multi-line text
        return sum(processor.count_syllables_line(line) 
                  for line in processor.segment_lines(text))
    else:
        # Single line
        return processor.count_syllables_line(text)