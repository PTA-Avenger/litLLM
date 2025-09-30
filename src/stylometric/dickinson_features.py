"""Emily Dickinson-specific stylistic feature detection and analysis."""

import re
import string
from typing import List, Dict, Tuple, Set, Optional
from collections import Counter, defaultdict
from dataclasses import dataclass
from .text_processing import PoetryTextProcessor


@dataclass
class DickinsonStyleProfile:
    """Profile for Emily Dickinson's distinctive stylistic features."""
    
    dash_frequency: float  # Dashes per line
    dash_positions: Dict[str, float]  # Position patterns (beginning, middle, end)
    capitalization_patterns: Dict[str, float]  # Irregular capitalization frequency
    slant_rhyme_frequency: float  # Percentage of slant rhymes vs perfect rhymes
    common_meter_adherence: float  # Adherence to common meter (8-6-8-6)
    enjambment_frequency: float  # Line breaks within phrases
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'dash_frequency': self.dash_frequency,
            'dash_positions': self.dash_positions,
            'capitalization_patterns': self.capitalization_patterns,
            'slant_rhyme_frequency': self.slant_rhyme_frequency,
            'common_meter_adherence': self.common_meter_adherence,
            'enjambment_frequency': self.enjambment_frequency
        }


class DickinsonFeatureDetector:
    """Detector for Emily Dickinson's distinctive stylistic features."""
    
    def __init__(self):
        """Initialize the Dickinson feature detector."""
        self.text_processor = PoetryTextProcessor()
        
        # Common words that Dickinson capitalized irregularly
        self.dickinson_capitalized_words = {
            'death', 'life', 'soul', 'heart', 'mind', 'nature', 'god', 'heaven',
            'earth', 'sun', 'moon', 'day', 'night', 'time', 'eternity', 'love',
            'hope', 'faith', 'truth', 'beauty', 'pain', 'joy', 'sorrow', 'fear',
            'bird', 'bee', 'flower', 'garden', 'house', 'door', 'window', 'room'
        }
        
        # Phonetic endings for slant rhyme detection
        self.vowel_sounds = {
            'a': ['ay', 'ah', 'ae'],
            'e': ['ee', 'eh', 'ay'],
            'i': ['eye', 'ih', 'ee'],
            'o': ['oh', 'aw', 'oo'],
            'u': ['oo', 'uh', 'you']
        }
    
    def detect_dash_patterns(self, text: str) -> Dict[str, any]:
        """Detect Emily Dickinson's characteristic dash usage patterns."""
        lines = self.text_processor.segment_lines(text)
        
        dash_count = 0
        dash_positions = {'beginning': 0, 'middle': 0, 'end': 0}
        total_lines = len(lines)
        
        # Patterns for different types of dashes (hyphen, em dash, en dash)
        dash_pattern = re.compile(r'[-\u2014\u2013\u2012]')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            dashes = dash_pattern.findall(line)
            dash_count += len(dashes)
            
            # Analyze dash positions
            for match in dash_pattern.finditer(line):
                pos = match.start()
                line_length = len(line)
                
                if pos < line_length * 0.2:
                    dash_positions['beginning'] += 1
                elif pos > line_length * 0.8:
                    dash_positions['end'] += 1
                else:
                    dash_positions['middle'] += 1
        
        # Calculate frequencies
        dash_frequency = dash_count / max(total_lines, 1)
        total_dashes = sum(dash_positions.values())
        
        if total_dashes > 0:
            position_frequencies = {
                pos: count / total_dashes 
                for pos, count in dash_positions.items()
            }
        else:
            position_frequencies = {'beginning': 0, 'middle': 0, 'end': 0}
        
        return {
            'dash_frequency': dash_frequency,
            'dash_positions': position_frequencies,
            'total_dashes': dash_count,
            'dash_types': self._classify_dash_usage(lines)
        }
    
    def _classify_dash_usage(self, lines: List[str]) -> Dict[str, int]:
        """Classify different types of dash usage."""
        usage_types = {
            'interruption': 0,  # Mid-sentence breaks
            'emphasis': 0,      # End of line emphasis
            'substitution': 0,  # Instead of punctuation
            'caesura': 0        # Poetic pause
        }
        
        for line in lines:
            # Simple heuristics for dash classification
            if re.search(r'\w+\s*[-—–]\s*\w+', line):
                usage_types['interruption'] += 1
            if re.search(r'[-—–]\s*$', line):
                usage_types['emphasis'] += 1
            if re.search(r'[-—–](?=[.!?]|$)', line):
                usage_types['substitution'] += 1
            if re.search(r'^\s*[-—–]', line):
                usage_types['caesura'] += 1
        
        return usage_types
    
    def detect_irregular_capitalization(self, text: str) -> Dict[str, any]:
        """Detect Dickinson's irregular capitalization patterns."""
        lines = self.text_processor.segment_lines(text)
        
        irregular_caps = 0
        total_words = 0
        capitalized_words = Counter()
        position_patterns = {'line_start': 0, 'mid_line': 0, 'line_end': 0}
        
        for line in lines:
            words = self.text_processor.get_word_tokens(line)
            
            for i, word in enumerate(words):
                total_words += 1
                clean_word = word.lower().strip(string.punctuation)
                
                # Check if word is capitalized but not at sentence start
                if (word[0].isupper() and 
                    i > 0 and 
                    not self._is_sentence_start(words, i) and
                    clean_word not in ['i']):  # Exclude proper "I"
                    
                    irregular_caps += 1
                    capitalized_words[clean_word] += 1
                    
                    # Track position patterns
                    if i == 0:
                        position_patterns['line_start'] += 1
                    elif i == len(words) - 1:
                        position_patterns['line_end'] += 1
                    else:
                        position_patterns['mid_line'] += 1
        
        # Calculate frequencies
        irregular_frequency = irregular_caps / max(total_words, 1)
        
        # Identify most commonly capitalized words
        common_caps = dict(capitalized_words.most_common(10))
        
        # Check alignment with known Dickinson patterns
        dickinson_alignment = sum(
            count for word, count in capitalized_words.items()
            if word in self.dickinson_capitalized_words
        ) / max(irregular_caps, 1)
        
        return {
            'irregular_frequency': irregular_frequency,
            'total_irregular': irregular_caps,
            'common_capitalized_words': common_caps,
            'position_patterns': position_patterns,
            'dickinson_alignment': dickinson_alignment
        }
    
    def _is_sentence_start(self, words: List[str], index: int) -> bool:
        """Check if word is at the start of a sentence."""
        if index == 0:
            return True
        
        # Check if previous word ends with sentence-ending punctuation
        prev_word = words[index - 1]
        return bool(re.search(r'[.!?]$', prev_word))
    
    def detect_slant_rhymes(self, text: str) -> Dict[str, any]:
        """Detect Emily Dickinson's characteristic slant rhymes."""
        stanzas = self.text_processor.segment_stanzas(text)
        
        total_rhyme_pairs = 0
        slant_rhymes = 0
        perfect_rhymes = 0
        rhyme_examples = []
        
        for stanza in stanzas:
            if len(stanza) < 2:
                continue
                
            # Get end words for rhyme analysis
            end_words = []
            for line in stanza:
                words = self.text_processor.get_word_tokens(line)
                if words:
                    # Clean the end word of punctuation
                    end_word = words[-1].lower().strip(string.punctuation)
                    end_words.append(end_word)
            
            # Analyze rhyme pairs (typically 2nd and 4th lines in common meter)
            if len(end_words) >= 4:
                pairs = [(end_words[1], end_words[3]), (end_words[0], end_words[2])]
            elif len(end_words) >= 2:
                pairs = [(end_words[0], end_words[-1])]
            else:
                continue
            
            for word1, word2 in pairs:
                if word1 and word2 and word1 != word2:
                    total_rhyme_pairs += 1
                    
                    if self._is_perfect_rhyme(word1, word2):
                        perfect_rhymes += 1
                        rhyme_examples.append((word1, word2, 'perfect'))
                    elif self._is_slant_rhyme(word1, word2):
                        slant_rhymes += 1
                        rhyme_examples.append((word1, word2, 'slant'))
        
        # Calculate frequencies
        slant_frequency = slant_rhymes / max(total_rhyme_pairs, 1)
        perfect_frequency = perfect_rhymes / max(total_rhyme_pairs, 1)
        
        return {
            'slant_rhyme_frequency': slant_frequency,
            'perfect_rhyme_frequency': perfect_frequency,
            'total_rhyme_pairs': total_rhyme_pairs,
            'slant_rhymes': slant_rhymes,
            'perfect_rhymes': perfect_rhymes,
            'rhyme_examples': rhyme_examples[:10]  # Sample examples
        }
    
    def _is_perfect_rhyme(self, word1: str, word2: str) -> bool:
        """Check if two words form a perfect rhyme."""
        if len(word1) < 2 or len(word2) < 2:
            return False
        
        # Simple perfect rhyme: same ending sounds (2-3 characters)
        for length in [3, 2]:
            if (len(word1) >= length and len(word2) >= length and
                word1[-length:] == word2[-length:]):
                return True
        
        return False
    
    def _is_slant_rhyme(self, word1: str, word2: str) -> bool:
        """Check if two words form a slant rhyme (near rhyme)."""
        if len(word1) < 2 or len(word2) < 2:
            return False
        
        # Slant rhyme patterns Dickinson used
        patterns = [
            # Consonance: same final consonant, different vowel
            self._has_consonance(word1, word2),
            # Assonance: same vowel sound, different consonant
            self._has_assonance(word1, word2),
            # Eye rhyme: similar spelling, different sound
            self._has_eye_rhyme(word1, word2),
            # Partial rhyme: shared sounds but not perfect
            self._has_partial_rhyme(word1, word2)
        ]
        
        return any(patterns)
    
    def _has_consonance(self, word1: str, word2: str) -> bool:
        """Check for consonance (same final consonant)."""
        consonants = 'bcdfghjklmnpqrstvwxyz'
        
        if len(word1) < 2 or len(word2) < 2:
            return False
        
        # Check if final consonants match but preceding vowels differ
        if (word1[-1] in consonants and word2[-1] in consonants and
            word1[-1] == word2[-1]):
            # Check if the vowel sounds before the consonant are different
            if len(word1) >= 2 and len(word2) >= 2:
                return word1[-2] != word2[-2]
        
        return False
    
    def _has_assonance(self, word1: str, word2: str) -> bool:
        """Check for assonance (same vowel sound)."""
        vowels = 'aeiou'
        
        # Find last vowel in each word
        last_vowel1 = None
        last_vowel2 = None
        
        for char in reversed(word1):
            if char in vowels:
                last_vowel1 = char
                break
        
        for char in reversed(word2):
            if char in vowels:
                last_vowel2 = char
                break
        
        return (last_vowel1 and last_vowel2 and 
                last_vowel1 == last_vowel2 and
                word1[-1] != word2[-1])
    
    def _has_eye_rhyme(self, word1: str, word2: str) -> bool:
        """Check for eye rhyme (similar spelling, different pronunciation)."""
        # Common eye rhyme patterns
        eye_rhyme_pairs = [
            ('ough', 'ough'),  # though/rough
            ('ead', 'ead'),    # read/dead
            ('ove', 'ove'),    # love/move
        ]
        
        for pattern1, pattern2 in eye_rhyme_pairs:
            if (word1.endswith(pattern1) and word2.endswith(pattern2) and
                pattern1 == pattern2):
                return True
        
        return False
    
    def _has_partial_rhyme(self, word1: str, word2: str) -> bool:
        """Check for partial rhyme (some shared sounds)."""
        # Check if words share some ending sounds but not perfect rhyme
        min_len = min(len(word1), len(word2))
        
        for i in range(1, min(min_len, 3)):
            if (word1[-i:] == word2[-i:] and 
                not self._is_perfect_rhyme(word1, word2)):
                return True
        
        return False
    
    def analyze_common_meter_adherence(self, text: str) -> Dict[str, any]:
        """Analyze adherence to common meter (8-6-8-6 syllable pattern)."""
        stanzas = self.text_processor.segment_stanzas(text)
        
        common_meter_stanzas = 0
        total_stanzas = len(stanzas)
        syllable_patterns = []
        
        for stanza in stanzas:
            if len(stanza) != 4:  # Common meter is typically 4 lines
                continue
            
            syllable_counts = []
            for line in stanza:
                count = self.text_processor.count_syllables_line(line)
                syllable_counts.append(count)
            
            syllable_patterns.append(syllable_counts)
            
            # Check for common meter pattern (8-6-8-6) with some tolerance
            if (self._is_common_meter_pattern(syllable_counts)):
                common_meter_stanzas += 1
        
        adherence = common_meter_stanzas / max(total_stanzas, 1)
        
        return {
            'common_meter_adherence': adherence,
            'common_meter_stanzas': common_meter_stanzas,
            'total_stanzas': total_stanzas,
            'syllable_patterns': syllable_patterns
        }
    
    def _is_common_meter_pattern(self, syllable_counts: List[int]) -> bool:
        """Check if syllable pattern matches common meter with tolerance."""
        if len(syllable_counts) != 4:
            return False
        
        # Common meter: 8-6-8-6 with ±1 syllable tolerance
        target = [8, 6, 8, 6]
        tolerance = 1
        
        for actual, expected in zip(syllable_counts, target):
            if abs(actual - expected) > tolerance:
                return False
        
        return True
    
    def create_dickinson_profile(self, text: str) -> DickinsonStyleProfile:
        """Create a comprehensive Dickinson style profile from text."""
        dash_analysis = self.detect_dash_patterns(text)
        cap_analysis = self.detect_irregular_capitalization(text)
        rhyme_analysis = self.detect_slant_rhymes(text)
        meter_analysis = self.analyze_common_meter_adherence(text)
        
        # Calculate enjambment frequency (simplified)
        lines = self.text_processor.segment_lines(text)
        enjambment_count = sum(
            1 for line in lines 
            if not re.search(r'[.!?;,]$', line.strip())
        )
        enjambment_frequency = enjambment_count / max(len(lines), 1)
        
        return DickinsonStyleProfile(
            dash_frequency=dash_analysis['dash_frequency'],
            dash_positions=dash_analysis['dash_positions'],
            capitalization_patterns=cap_analysis,
            slant_rhyme_frequency=rhyme_analysis['slant_rhyme_frequency'],
            common_meter_adherence=meter_analysis['common_meter_adherence'],
            enjambment_frequency=enjambment_frequency
        )
    
    def score_dickinson_similarity(self, text: str) -> Dict[str, float]:
        """Score how similar a text is to Dickinson's style."""
        profile = self.create_dickinson_profile(text)
        
        # Expected Dickinson ranges (based on literary analysis)
        dickinson_ranges = {
            'dash_frequency': (0.5, 2.0),  # Dashes per line
            'slant_rhyme_frequency': (0.3, 0.7),  # 30-70% slant rhymes
            'common_meter_adherence': (0.4, 0.8),  # 40-80% common meter
            'irregular_caps': (0.1, 0.3),  # 10-30% irregular capitalization
        }
        
        scores = {}
        
        # Score dash usage
        dash_score = self._score_in_range(
            profile.dash_frequency, 
            dickinson_ranges['dash_frequency']
        )
        scores['dash_usage'] = dash_score
        
        # Score slant rhyme usage
        slant_score = self._score_in_range(
            profile.slant_rhyme_frequency,
            dickinson_ranges['slant_rhyme_frequency']
        )
        scores['slant_rhyme'] = slant_score
        
        # Score common meter adherence
        meter_score = self._score_in_range(
            profile.common_meter_adherence,
            dickinson_ranges['common_meter_adherence']
        )
        scores['common_meter'] = meter_score
        
        # Score irregular capitalization
        cap_score = self._score_in_range(
            profile.capitalization_patterns.get('irregular_frequency', 0),
            dickinson_ranges['irregular_caps']
        )
        scores['capitalization'] = cap_score
        
        # Overall similarity score (weighted average)
        weights = {
            'dash_usage': 0.3,
            'slant_rhyme': 0.25,
            'common_meter': 0.25,
            'capitalization': 0.2
        }
        
        overall_score = sum(
            scores[feature] * weights[feature]
            for feature in weights.keys()
        )
        scores['overall_similarity'] = overall_score
        
        return scores
    
    def _score_in_range(self, value: float, target_range: Tuple[float, float]) -> float:
        """Score how well a value fits within a target range."""
        min_val, max_val = target_range
        
        if min_val <= value <= max_val:
            return 1.0
        elif value < min_val:
            # Score based on how close to minimum
            return max(0.0, 1.0 - (min_val - value) / min_val)
        else:
            # Score based on how close to maximum
            return max(0.0, 1.0 - (value - max_val) / max_val)


# Convenience functions for external use
def analyze_dickinson_features(text: str) -> Dict[str, any]:
    """Analyze Dickinson-specific features in a text."""
    detector = DickinsonFeatureDetector()
    return {
        'dash_patterns': detector.detect_dash_patterns(text),
        'capitalization': detector.detect_irregular_capitalization(text),
        'slant_rhymes': detector.detect_slant_rhymes(text),
        'common_meter': detector.analyze_common_meter_adherence(text)
    }


def score_dickinson_similarity(text: str) -> Dict[str, float]:
    """Score similarity to Emily Dickinson's style."""
    detector = DickinsonFeatureDetector()
    return detector.score_dickinson_similarity(text)


def create_dickinson_profile(text: str) -> DickinsonStyleProfile:
    """Create a Dickinson style profile from text."""
    detector = DickinsonFeatureDetector()
    return detector.create_dickinson_profile(text)