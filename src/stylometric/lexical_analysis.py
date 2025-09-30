"""
Lexical feature extraction for poetry analysis.

This module provides functions for extracting lexical features such as
Type-Token Ratio (TTR), vocabulary richness, POS tagging, and word/sentence
length distributions.
"""

import re
import math
from typing import List, Dict, Tuple, Optional, Counter
from collections import Counter, defaultdict
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    try:
        nltk.download('averaged_perceptron_tagger')
    except:
        # Fallback for newer NLTK versions
        nltk.download('averaged_perceptron_tagger_eng')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class LexicalAnalyzer:
    """Main class for lexical feature extraction from poetry."""
    
    def __init__(self, language: str = 'english'):
        """Initialize the lexical analyzer.
        
        Args:
            language: Language for stopwords and analysis (default: 'english')
        """
        self.language = language
        try:
            self.stop_words = set(stopwords.words(language))
        except OSError:
            # Fallback if stopwords not available
            self.stop_words = set()
    
    def get_word_tokens(self, text: str, include_punctuation: bool = False) -> List[str]:
        """Extract word tokens from text.
        
        Args:
            text: Input text
            include_punctuation: Whether to include punctuation tokens
            
        Returns:
            List of word tokens
        """
        if not text:
            return []
            
        if include_punctuation:
            tokens = word_tokenize(text)
        else:
            # Extract only word tokens (letters, numbers, apostrophes)
            tokens = re.findall(r"\b\w+(?:'\w+)?\b", text)
            
        return tokens
    
    def calculate_ttr(self, text: str, case_sensitive: bool = False) -> float:
        """Calculate Type-Token Ratio (TTR).
        
        Args:
            text: Input text
            case_sensitive: Whether to consider case in token counting
            
        Returns:
            TTR value (types/tokens), 0 if no tokens
        """
        tokens = self.get_word_tokens(text)
        if not tokens:
            return 0.0
            
        if not case_sensitive:
            tokens = [token.lower() for token in tokens]
            
        types = len(set(tokens))
        total_tokens = len(tokens)
        
        return types / total_tokens if total_tokens > 0 else 0.0
    
    def calculate_mattr(self, text: str, window_size: int = 100, case_sensitive: bool = False) -> float:
        """Calculate Moving Average Type-Token Ratio (MATTR).
        
        MATTR is more stable than TTR for texts of varying lengths.
        
        Args:
            text: Input text
            window_size: Size of moving window for calculation
            case_sensitive: Whether to consider case in token counting
            
        Returns:
            MATTR value, 0 if insufficient tokens
        """
        tokens = self.get_word_tokens(text)
        if len(tokens) < window_size:
            return self.calculate_ttr(text, case_sensitive)
            
        if not case_sensitive:
            tokens = [token.lower() for token in tokens]
            
        ttr_values = []
        for i in range(len(tokens) - window_size + 1):
            window_tokens = tokens[i:i + window_size]
            types = len(set(window_tokens))
            ttr_values.append(types / window_size)
            
        return sum(ttr_values) / len(ttr_values) if ttr_values else 0.0
    
    def calculate_lexical_density(self, text: str) -> float:
        """Calculate lexical density (content words / total words).
        
        Args:
            text: Input text
            
        Returns:
            Lexical density ratio
        """
        tokens = self.get_word_tokens(text)
        if not tokens:
            return 0.0
            
        # Get POS tags with error handling
        try:
            pos_tags = pos_tag(tokens)
        except LookupError:
            # Fallback: assume all words are content words
            return 1.0
        
        # Content word POS tags (nouns, verbs, adjectives, adverbs)
        content_pos = {'NN', 'NNS', 'NNP', 'NNPS',  # Nouns
                      'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',  # Verbs
                      'JJ', 'JJR', 'JJS',  # Adjectives
                      'RB', 'RBR', 'RBS'}  # Adverbs
        
        content_words = sum(1 for _, pos in pos_tags if pos in content_pos)
        
        return content_words / len(tokens) if tokens else 0.0
    
    def get_pos_distribution(self, text: str) -> Dict[str, float]:
        """Get Part-of-Speech distribution.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping POS tags to their relative frequencies
        """
        tokens = self.get_word_tokens(text)
        if not tokens:
            return {}
            
        try:
            pos_tags = pos_tag(tokens)
        except LookupError:
            # Fallback: return empty distribution if POS tagger not available
            return {}
        pos_counts = Counter(pos for _, pos in pos_tags)
        total_tags = len(pos_tags)
        
        return {pos: count / total_tags for pos, count in pos_counts.items()}
    
    def get_word_length_distribution(self, text: str) -> Dict[str, float]:
        """Get word length distribution statistics.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with length statistics (mean, std, distribution)
        """
        tokens = self.get_word_tokens(text)
        if not tokens:
            return {'mean': 0.0, 'std': 0.0, 'distribution': {}}
            
        lengths = [len(token) for token in tokens]
        
        # Calculate statistics
        mean_length = sum(lengths) / len(lengths)
        variance = sum((length - mean_length) ** 2 for length in lengths) / len(lengths)
        std_length = math.sqrt(variance)
        
        # Length distribution
        length_counts = Counter(lengths)
        total_words = len(tokens)
        length_dist = {str(length): count / total_words 
                      for length, count in length_counts.items()}
        
        return {
            'mean': mean_length,
            'std': std_length,
            'distribution': length_dist
        }
    
    def get_sentence_length_distribution(self, text: str) -> Dict[str, float]:
        """Get sentence length distribution statistics.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentence length statistics
        """
        try:
            sentences = sent_tokenize(text)
        except LookupError:
            # Fallback: split on periods, exclamation marks, and question marks
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return {'mean': 0.0, 'std': 0.0, 'distribution': {}}
            
        # Count words in each sentence
        sentence_lengths = []
        for sentence in sentences:
            words = self.get_word_tokens(sentence)
            sentence_lengths.append(len(words))
        
        if not sentence_lengths:
            return {'mean': 0.0, 'std': 0.0, 'distribution': {}}
            
        # Calculate statistics
        mean_length = sum(sentence_lengths) / len(sentence_lengths)
        variance = sum((length - mean_length) ** 2 for length in sentence_lengths) / len(sentence_lengths)
        std_length = math.sqrt(variance)
        
        # Length distribution
        length_counts = Counter(sentence_lengths)
        total_sentences = len(sentences)
        length_dist = {str(length): count / total_sentences 
                      for length, count in length_counts.items()}
        
        return {
            'mean': mean_length,
            'std': std_length,
            'distribution': length_dist
        }
    
    def get_vocabulary_richness_metrics(self, text: str) -> Dict[str, float]:
        """Calculate various vocabulary richness metrics.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with multiple richness metrics
        """
        tokens = self.get_word_tokens(text)
        if not tokens:
            return {
                'ttr': 0.0,
                'mattr': 0.0,
                'mtld': 0.0,
                'hdd': 0.0,
                'unique_words': 0,
                'total_words': 0
            }
            
        # Convert to lowercase for consistency
        tokens_lower = [token.lower() for token in tokens]
        
        # Basic metrics
        unique_words = len(set(tokens_lower))
        total_words = len(tokens)
        ttr = unique_words / total_words
        mattr = self.calculate_mattr(' '.join(tokens))
        
        # Measure of Textual Lexical Diversity (MTLD)
        mtld = self._calculate_mtld(tokens_lower)
        
        # HD-D (Hypergeometric Distribution D)
        hdd = self._calculate_hdd(tokens_lower)
        
        return {
            'ttr': ttr,
            'mattr': mattr,
            'mtld': mtld,
            'hdd': hdd,
            'unique_words': unique_words,
            'total_words': total_words
        }
    
    def _calculate_mtld(self, tokens: List[str], threshold: float = 0.72) -> float:
        """Calculate Measure of Textual Lexical Diversity (MTLD).
        
        Args:
            tokens: List of lowercase tokens
            threshold: TTR threshold for factor calculation
            
        Returns:
            MTLD value
        """
        if len(tokens) < 50:  # MTLD requires sufficient text
            return 0.0
            
        def calculate_factors(token_list, reverse=False):
            if reverse:
                token_list = token_list[::-1]
                
            factors = 0
            start = 0
            
            while start < len(token_list):
                types = set()
                for i in range(start, len(token_list)):
                    types.add(token_list[i])
                    current_ttr = len(types) / (i - start + 1)
                    
                    if current_ttr <= threshold:
                        factors += 1
                        start = i + 1
                        break
                else:
                    # Partial factor for remaining tokens
                    if len(token_list) - start > 0:
                        remaining_types = len(set(token_list[start:]))
                        remaining_tokens = len(token_list) - start
                        remaining_ttr = remaining_types / remaining_tokens
                        factors += (1 - remaining_ttr) / (1 - threshold)
                    break
                    
            return factors
        
        # Calculate factors in both directions
        forward_factors = calculate_factors(tokens)
        reverse_factors = calculate_factors(tokens, reverse=True)
        
        average_factors = (forward_factors + reverse_factors) / 2
        
        return len(tokens) / average_factors if average_factors > 0 else 0.0
    
    def _calculate_hdd(self, tokens: List[str], sample_size: int = 42) -> float:
        """Calculate HD-D (Hypergeometric Distribution D).
        
        Args:
            tokens: List of lowercase tokens
            sample_size: Sample size for hypergeometric calculation
            
        Returns:
            HD-D value
        """
        if len(tokens) < sample_size:
            return 0.0
            
        word_counts = Counter(tokens)
        total_tokens = len(tokens)
        
        # Calculate probability that each type appears in random sample
        probabilities = []
        for word, count in word_counts.items():
            # Probability that word does NOT appear in sample
            prob_not_in_sample = 1.0
            for i in range(sample_size):
                prob_not_in_sample *= (total_tokens - count - i) / (total_tokens - i)
            
            # Probability that word DOES appear in sample
            prob_in_sample = 1 - prob_not_in_sample
            probabilities.append(prob_in_sample)
        
        return sum(probabilities)
    
    def _count_sentences(self, text: str) -> int:
        """Count sentences in text with fallback handling.
        
        Args:
            text: Input text
            
        Returns:
            Number of sentences
        """
        if not text:
            return 0
            
        try:
            return len(sent_tokenize(text))
        except LookupError:
            # Fallback: count sentence-ending punctuation
            sentences = re.split(r'[.!?]+', text)
            return len([s for s in sentences if s.strip()])
    
    def analyze_lexical_features(self, text: str) -> Dict[str, any]:
        """Comprehensive lexical analysis of text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary containing all lexical features
        """
        return {
            'vocabulary_richness': self.get_vocabulary_richness_metrics(text),
            'lexical_density': self.calculate_lexical_density(text),
            'pos_distribution': self.get_pos_distribution(text),
            'word_length_stats': self.get_word_length_distribution(text),
            'sentence_length_stats': self.get_sentence_length_distribution(text),
            'total_tokens': len(self.get_word_tokens(text)),
            'total_sentences': self._count_sentences(text)
        }


# Convenience functions for direct use
def calculate_type_token_ratio(text: str, case_sensitive: bool = False) -> float:
    """Calculate TTR using default analyzer."""
    analyzer = LexicalAnalyzer()
    return analyzer.calculate_ttr(text, case_sensitive)


def get_pos_tags(text: str) -> List[Tuple[str, str]]:
    """Get POS tags using default analyzer."""
    analyzer = LexicalAnalyzer()
    tokens = analyzer.get_word_tokens(text)
    if not tokens:
        return []
    try:
        return pos_tag(tokens)
    except LookupError:
        # Return tokens with unknown POS tags if tagger not available
        return [(token, 'UNK') for token in tokens]


def analyze_vocabulary_richness(text: str) -> Dict[str, float]:
    """Analyze vocabulary richness using default analyzer."""
    analyzer = LexicalAnalyzer()
    return analyzer.get_vocabulary_richness_metrics(text)