"""Structural feature analysis for poetry."""

from typing import List, Dict
from .text_processing import PoetryTextProcessor


class StructuralAnalyzer:
    """Main class for structural feature analysis of poetry."""
    
    def __init__(self):
        """Initialize the structural analyzer."""
        self.text_processor = PoetryTextProcessor()
        
    def analyze_meter_basic(self, line: str) -> Dict[str, any]:
        """Analyze basic meter patterns in a line of poetry."""
        if not line:
            return {
                'syllable_count': 0,
                'meter_type': 'unknown'
            }
            
        syllable_count = self.text_processor.count_syllables_line(line)
        
        if syllable_count == 10:
            meter_type = 'pentameter'
        elif syllable_count == 8:
            meter_type = 'common_meter'
        elif syllable_count in [6, 7]:
            meter_type = 'short_meter'
        else:
            meter_type = 'free_verse'
        
        return {
            'syllable_count': syllable_count,
            'meter_type': meter_type
        }
    
    def analyze_rhyme_scheme(self, stanza: List[str]) -> Dict[str, any]:
        """Analyze rhyme scheme of a stanza."""
        if not stanza:
            return {
                'rhyme_scheme': '',
                'rhyme_type': 'none'
            }
            
        end_words = []
        for line in stanza:
            words = self.text_processor.get_word_tokens(line)
            if words:
                end_words.append(words[-1].lower())
            else:
                end_words.append('')
        
        # Simple rhyme scheme detection
        scheme = self._simple_rhyme_scheme(end_words)
        
        return {
            'rhyme_scheme': scheme,
            'rhyme_type': self._classify_rhyme_type(scheme)
        }
    
    def _simple_rhyme_scheme(self, end_words: List[str]) -> str:
        """Simple rhyme scheme detection."""
        if not end_words:
            return ''
            
        scheme = []
        groups = {}
        letter = 'A'
        
        for word in end_words:
            found = False
            for group_word, group_letter in groups.items():
                if self._words_rhyme(word, group_word):
                    scheme.append(group_letter)
                    found = True
                    break
            
            if not found:
                groups[word] = letter
                scheme.append(letter)
                letter = chr(ord(letter) + 1)
        
        return ''.join(scheme)
    
    def _words_rhyme(self, word1: str, word2: str) -> bool:
        """Check if two words rhyme."""
        if not word1 or not word2:
            return word1 == word2
        
        if word1 == word2:
            return True
            
        # Simple ending check
        min_len = min(len(word1), len(word2))
        for i in range(2, min(min_len + 1, 4)):
            if word1[-i:] == word2[-i:]:
                return True
        return False
    
    def _classify_rhyme_type(self, scheme: str) -> str:
        """Classify rhyme scheme type."""
        if not scheme:
            return 'none'
        elif scheme == 'AABB':
            return 'couplet'
        elif scheme == 'ABAB':
            return 'alternate'
        elif len(set(scheme)) == len(scheme):
            return 'no_rhyme'
        else:
            return 'irregular'
    
    def analyze_structural_features(self, text: str) -> Dict[str, any]:
        """Comprehensive structural analysis."""
        processed = self.text_processor.preprocess_for_analysis(text)
        lines = processed['lines']
        stanzas = processed['stanzas']
        
        meter_analysis = [self.analyze_meter_basic(line) for line in lines]
        rhyme_analysis = [self.analyze_rhyme_scheme(stanza) for stanza in stanzas]
        
        return {
            'meter_analysis': meter_analysis,
            'rhyme_analysis': rhyme_analysis,
            'total_lines': len(lines),
            'total_stanzas': len(stanzas)
        }


def analyze_poem_meter(text: str) -> List[Dict[str, any]]:
    """Analyze meter for each line."""
    analyzer = StructuralAnalyzer()
    processor = PoetryTextProcessor()
    lines = processor.segment_lines(text)
    return [analyzer.analyze_meter_basic(line) for line in lines]


def analyze_poem_rhyme_scheme(text: str) -> List[Dict[str, any]]:
    """Analyze rhyme scheme for each stanza."""
    analyzer = StructuralAnalyzer()
    processor = PoetryTextProcessor()
    stanzas = processor.segment_stanzas(text)
    return [analyzer.analyze_rhyme_scheme(stanza) for stanza in stanzas]