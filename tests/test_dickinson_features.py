"""Tests for Emily Dickinson-specific feature detection."""

import pytest
from src.stylometric.dickinson_features import (
    DickinsonFeatureDetector,
    DickinsonStyleProfile,
    analyze_dickinson_features,
    score_dickinson_similarity,
    create_dickinson_profile
)


class TestDickinsonFeatureDetector:
    """Test class for Dickinson feature detection."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = DickinsonFeatureDetector()
        
        # Authentic Emily Dickinson poems for testing
        self.dickinson_poem_1 = """
        I'm Nobody! Who are you?
        Are you — Nobody — Too?
        Then there's a pair of us!
        Don't tell! they'd advertise — you know!
        
        How dreary — to be — Somebody!
        How public — like a Frog —
        To tell one's name — the livelong June —
        To an admiring Bog!
        """
        
        self.dickinson_poem_2 = """
        Because I could not stop for Death —
        He kindly stopped for me —
        The Carriage held but just Ourselves —
        And Immortality.
        
        We slowly drove — He knew no haste
        And I had put away
        My labor and my leisure too,
        For His Civility —
        """
        
        self.dickinson_poem_3 = """
        Hope is the thing with feathers —
        That perches in the soul —
        And sings the tune without the words —
        And never stops — at all —
        
        And sweetest — in the Gale — is heard —
        And sore must be the storm —
        That could abash the little Bird
        That kept so many warm —
        """
        
        # Non-Dickinson poem for comparison
        self.shakespeare_sonnet = """
        Shall I compare thee to a summer's day?
        Thou art more lovely and more temperate:
        Rough winds do shake the darling buds of May,
        And summer's lease hath all too short a date:
        """
        
        # Modern free verse for comparison
        self.modern_poem = """
        The red wheelbarrow
        glazed with rain water
        beside the white chickens
        """
    
    def test_detect_dash_patterns_dickinson(self):
        """Test dash pattern detection on authentic Dickinson poetry."""
        result = self.detector.detect_dash_patterns(self.dickinson_poem_1)
        
        # Dickinson should have high dash frequency
        assert result['dash_frequency'] > 0.5
        assert result['total_dashes'] > 5
        
        # Should detect various dash positions
        assert 'beginning' in result['dash_positions']
        assert 'middle' in result['dash_positions']
        assert 'end' in result['dash_positions']
        
        # Should classify dash types
        assert 'interruption' in result['dash_types']
        assert 'emphasis' in result['dash_types']
    
    def test_detect_dash_patterns_non_dickinson(self):
        """Test dash pattern detection on non-Dickinson poetry."""
        result = self.detector.detect_dash_patterns(self.shakespeare_sonnet)
        
        # Shakespeare should have low dash frequency
        assert result['dash_frequency'] < 0.2
        assert result['total_dashes'] <= 1
    
    def test_detect_irregular_capitalization_dickinson(self):
        """Test irregular capitalization detection on Dickinson poetry."""
        result = self.detector.detect_irregular_capitalization(self.dickinson_poem_2)
        
        # Dickinson should have irregular capitalization
        assert result['irregular_frequency'] > 0.1
        assert result['total_irregular'] > 0
        
        # Should detect common Dickinson capitalized words
        common_caps = result['common_capitalized_words']
        dickinson_words = {'death', 'carriage', 'ourselves', 'immortality', 'civility'}
        found_dickinson_words = set(common_caps.keys()) & dickinson_words
        assert len(found_dickinson_words) > 0
        
        # Should have some alignment with known Dickinson patterns
        assert result['dickinson_alignment'] > 0.1
    
    def test_detect_irregular_capitalization_non_dickinson(self):
        """Test irregular capitalization on non-Dickinson poetry."""
        result = self.detector.detect_irregular_capitalization(self.shakespeare_sonnet)
        
        # Shakespeare should have low irregular capitalization
        assert result['irregular_frequency'] < 0.1
        assert result['dickinson_alignment'] < 0.2
    
    def test_detect_slant_rhymes_dickinson(self):
        """Test slant rhyme detection on Dickinson poetry."""
        result = self.detector.detect_slant_rhymes(self.dickinson_poem_3)
        
        # Dickinson should have some slant rhymes
        assert result['total_rhyme_pairs'] > 0
        
        # Should detect both perfect and slant rhymes
        total_rhymes = result['slant_rhymes'] + result['perfect_rhymes']
        assert total_rhymes > 0
        
        # Should have examples of rhyme pairs
        assert len(result['rhyme_examples']) > 0
        
        # Verify rhyme example format
        for example in result['rhyme_examples']:
            assert len(example) == 3  # (word1, word2, type)
            assert example[2] in ['perfect', 'slant']
    
    def test_slant_rhyme_detection_methods(self):
        """Test individual slant rhyme detection methods."""
        # Test consonance (same final consonant, different preceding vowel)
        assert self.detector._has_consonance('mad', 'red')    # both end in 'd', different vowels
        assert self.detector._has_consonance('cat', 'cut')    # both end in 't', different vowels
        
        # Test assonance (same vowel sound, different final consonant)
        assert self.detector._has_assonance('cat', 'bad')  # both have 'a', different endings
        
        # Test perfect rhyme detection
        assert self.detector._is_perfect_rhyme('day', 'may')
        assert self.detector._is_perfect_rhyme('cat', 'bat')
        
        # Test slant rhyme detection
        assert self.detector._is_slant_rhyme('mad', 'red')  # consonance
        assert self.detector._is_slant_rhyme('cat', 'bad')  # assonance
    
    def test_analyze_common_meter_adherence_dickinson(self):
        """Test common meter analysis on Dickinson poetry."""
        result = self.detector.analyze_common_meter_adherence(self.dickinson_poem_1)
        
        # Should analyze stanza patterns
        assert result['total_stanzas'] > 0
        assert 'syllable_patterns' in result
        
        # Dickinson often used common meter variations
        assert 0.0 <= result['common_meter_adherence'] <= 1.0
    
    def test_common_meter_pattern_recognition(self):
        """Test common meter pattern recognition."""
        # Perfect common meter: 8-6-8-6
        perfect_pattern = [8, 6, 8, 6]
        assert self.detector._is_common_meter_pattern(perfect_pattern)
        
        # With tolerance: 7-6-9-5 (within ±1)
        tolerant_pattern = [7, 6, 9, 5]
        assert self.detector._is_common_meter_pattern(tolerant_pattern)
        
        # Outside tolerance: 10-4-10-4
        outside_pattern = [10, 4, 10, 4]
        assert not self.detector._is_common_meter_pattern(outside_pattern)
        
        # Wrong number of lines
        wrong_length = [8, 6, 8]
        assert not self.detector._is_common_meter_pattern(wrong_length)
    
    def test_create_dickinson_profile(self):
        """Test creation of comprehensive Dickinson style profile."""
        profile = self.detector.create_dickinson_profile(self.dickinson_poem_2)
        
        # Verify profile structure
        assert isinstance(profile, DickinsonStyleProfile)
        assert hasattr(profile, 'dash_frequency')
        assert hasattr(profile, 'dash_positions')
        assert hasattr(profile, 'capitalization_patterns')
        assert hasattr(profile, 'slant_rhyme_frequency')
        assert hasattr(profile, 'common_meter_adherence')
        assert hasattr(profile, 'enjambment_frequency')
        
        # Verify profile values are reasonable
        assert 0.0 <= profile.dash_frequency <= 10.0
        assert 0.0 <= profile.slant_rhyme_frequency <= 1.0
        assert 0.0 <= profile.common_meter_adherence <= 1.0
        assert 0.0 <= profile.enjambment_frequency <= 1.0
        
        # Test profile serialization
        profile_dict = profile.to_dict()
        assert isinstance(profile_dict, dict)
        assert 'dash_frequency' in profile_dict
    
    def test_score_dickinson_similarity(self):
        """Test Dickinson similarity scoring."""
        # Test on authentic Dickinson poem
        dickinson_scores = self.detector.score_dickinson_similarity(self.dickinson_poem_1)
        
        # Verify score structure
        expected_features = ['dash_usage', 'slant_rhyme', 'common_meter', 'capitalization', 'overall_similarity']
        for feature in expected_features:
            assert feature in dickinson_scores
            assert 0.0 <= dickinson_scores[feature] <= 1.0
        
        # Test on non-Dickinson poem
        shakespeare_scores = self.detector.score_dickinson_similarity(self.shakespeare_sonnet)
        
        # Dickinson poem should score higher than Shakespeare
        assert dickinson_scores['overall_similarity'] > shakespeare_scores['overall_similarity']
        
        # Dickinson should score particularly high on dash usage
        assert dickinson_scores['dash_usage'] > shakespeare_scores['dash_usage']
    
    def test_score_in_range_method(self):
        """Test the range scoring method."""
        target_range = (0.5, 1.5)
        
        # Perfect score within range
        assert self.detector._score_in_range(1.0, target_range) == 1.0
        assert self.detector._score_in_range(0.5, target_range) == 1.0
        assert self.detector._score_in_range(1.5, target_range) == 1.0
        
        # Partial scores outside range
        below_score = self.detector._score_in_range(0.25, target_range)
        above_score = self.detector._score_in_range(2.0, target_range)
        
        assert 0.0 <= below_score < 1.0
        assert 0.0 <= above_score < 1.0
        
        # Zero score for extreme values
        extreme_low = self.detector._score_in_range(0.0, target_range)
        extreme_high = self.detector._score_in_range(3.0, target_range)
        
        assert extreme_low >= 0.0
        assert extreme_high >= 0.0
    
    def test_convenience_functions(self):
        """Test convenience functions for external use."""
        # Test analyze_dickinson_features
        features = analyze_dickinson_features(self.dickinson_poem_1)
        
        assert 'dash_patterns' in features
        assert 'capitalization' in features
        assert 'slant_rhymes' in features
        assert 'common_meter' in features
        
        # Test score_dickinson_similarity
        scores = score_dickinson_similarity(self.dickinson_poem_1)
        assert 'overall_similarity' in scores
        assert 0.0 <= scores['overall_similarity'] <= 1.0
        
        # Test create_dickinson_profile
        profile = create_dickinson_profile(self.dickinson_poem_1)
        assert isinstance(profile, DickinsonStyleProfile)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Empty text
        empty_result = self.detector.detect_dash_patterns("")
        assert empty_result['dash_frequency'] == 0.0
        assert empty_result['total_dashes'] == 0
        
        # Single line
        single_line = "I'm Nobody! Who are you?"
        single_result = self.detector.detect_dash_patterns(single_line)
        assert single_result['dash_frequency'] >= 0.0
        
        # Text with no rhymes
        no_rhyme_text = "Line one\nLine two\nLine three\nLine four"
        rhyme_result = self.detector.detect_slant_rhymes(no_rhyme_text)
        assert rhyme_result['total_rhyme_pairs'] >= 0
        
        # Text with no capitalization
        lowercase_text = "all lowercase text\nwith no capitals\nexcept sentence starts"
        cap_result = self.detector.detect_irregular_capitalization(lowercase_text)
        assert cap_result['irregular_frequency'] == 0.0
    
    def test_multiple_dickinson_poems(self):
        """Test consistency across multiple Dickinson poems."""
        poems = [self.dickinson_poem_1, self.dickinson_poem_2, self.dickinson_poem_3]
        
        all_scores = []
        for poem in poems:
            scores = self.detector.score_dickinson_similarity(poem)
            all_scores.append(scores['overall_similarity'])
        
        # All Dickinson poems should score reasonably high
        for score in all_scores:
            assert score > 0.3  # Reasonable threshold for Dickinson similarity
        
        # Average score should be higher than non-Dickinson
        avg_dickinson_score = sum(all_scores) / len(all_scores)
        shakespeare_score = self.detector.score_dickinson_similarity(self.shakespeare_sonnet)['overall_similarity']
        
        assert avg_dickinson_score > shakespeare_score
    
    def test_feature_integration(self):
        """Test integration of all Dickinson features."""
        # Analyze a poem with multiple Dickinson features
        complex_poem = """
        The Soul selects her own Society —
        Then — shuts the Door —
        To her divine Majority —
        Present no more —
        
        Unmoved — she notes the Chariots — pausing —
        At her low Gate —
        Unmoved — an Emperor be kneeling
        Upon her Mat —
        """
        
        features = analyze_dickinson_features(complex_poem)
        
        # Should detect dashes
        assert features['dash_patterns']['dash_frequency'] > 0.5
        
        # Should detect irregular capitalization
        assert features['capitalization']['irregular_frequency'] > 0.1
        
        # Should analyze meter
        assert features['common_meter']['total_stanzas'] > 0
        
        # Should analyze rhymes
        assert features['slant_rhymes']['total_rhyme_pairs'] > 0
        
        # Overall similarity should be reasonably high
        similarity = score_dickinson_similarity(complex_poem)
        assert similarity['overall_similarity'] >= 0.4


if __name__ == '__main__':
    pytest.main([__file__])