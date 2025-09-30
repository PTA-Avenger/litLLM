"""Integration tests for poet profile creation and management."""

import pytest
import json
import tempfile
from pathlib import Path
from src.stylometric.poet_profile import PoetProfile, PoetProfileBuilder


class TestPoetProfile:
    """Test cases for PoetProfile data model."""
    
    def test_poet_profile_creation(self):
        """Test basic poet profile creation."""
        profile = PoetProfile(
            poet_name="Test Poet",
            corpus_size=10,
            total_lines=50,
            total_words=200,
            vocabulary_richness={'ttr': 0.75, 'mean_word_length': 4.2},
            lexical_density=0.65,
            pos_distribution={'NOUN': 0.3, 'VERB': 0.2, 'ADJ': 0.15},
            word_length_stats={'mean': 4.2, 'std': 1.8},
            meter_patterns={'iambic_pentameter': 0.6, 'free_verse': 0.4},
            rhyme_patterns={'ABAB': 0.5, 'AABB': 0.3, 'free': 0.2},
            average_syllables_per_line=8.5
        )
        
        assert profile.poet_name == "Test Poet"
        assert profile.corpus_size == 10
        assert profile.total_lines == 50
        assert profile.total_words == 200
        assert profile.creation_timestamp is not None
        assert profile.analysis_version == "1.0"
    
    def test_profile_serialization_json(self):
        """Test JSON serialization and deserialization."""
        profile = PoetProfile(
            poet_name="Emily Dickinson",
            corpus_size=5,
            total_lines=25,
            total_words=100,
            vocabulary_richness={'ttr': 0.8},
            lexical_density=0.7,
            pos_distribution={'NOUN': 0.35},
            word_length_stats={'mean': 4.0, 'std': 2.0},
            meter_patterns={'common_meter': 0.8},
            rhyme_patterns={'slant': 0.6},
            average_syllables_per_line=7.0
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save and load
            profile.save_json(temp_path)
            loaded_profile = PoetProfile.load_json(temp_path)
            
            # Verify all fields match
            assert loaded_profile.poet_name == profile.poet_name
            assert loaded_profile.corpus_size == profile.corpus_size
            assert loaded_profile.vocabulary_richness == profile.vocabulary_richness
            assert loaded_profile.lexical_density == profile.lexical_density
            
        finally:
            Path(temp_path).unlink()
    
    def test_profile_serialization_pickle(self):
        """Test pickle serialization and deserialization."""
        profile = PoetProfile(
            poet_name="Walt Whitman",
            corpus_size=3,
            total_lines=30,
            total_words=150,
            vocabulary_richness={'ttr': 0.65},
            lexical_density=0.6,
            pos_distribution={'NOUN': 0.25, 'VERB': 0.25},
            word_length_stats={'mean': 5.0, 'std': 2.5},
            meter_patterns={'free_verse': 1.0},
            rhyme_patterns={'free': 1.0},
            average_syllables_per_line=10.0
        )
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save and load
            profile.save_pickle(temp_path)
            loaded_profile = PoetProfile.load_pickle(temp_path)
            
            # Verify all fields match
            assert loaded_profile.poet_name == profile.poet_name
            assert loaded_profile.total_words == profile.total_words
            assert loaded_profile.meter_patterns == profile.meter_patterns
            
        finally:
            Path(temp_path).unlink()
    
    def test_profile_similarity_calculation(self):
        """Test similarity calculation between profiles."""
        profile1 = PoetProfile(
            poet_name="Poet A",
            corpus_size=5,
            total_lines=25,
            total_words=100,
            vocabulary_richness={'ttr': 0.7, 'mean_word_length': 4.0},
            lexical_density=0.6,
            pos_distribution={'NOUN': 0.3, 'VERB': 0.2, 'ADJ': 0.1},
            word_length_stats={'mean': 4.0, 'std': 1.5},
            meter_patterns={'iambic': 0.8, 'free': 0.2},
            rhyme_patterns={'ABAB': 0.6, 'free': 0.4},
            average_syllables_per_line=8.0
        )
        
        profile2 = PoetProfile(
            poet_name="Poet B",
            corpus_size=5,
            total_lines=25,
            total_words=100,
            vocabulary_richness={'ttr': 0.75, 'mean_word_length': 4.2},
            lexical_density=0.65,
            pos_distribution={'NOUN': 0.32, 'VERB': 0.18, 'ADJ': 0.12},
            word_length_stats={'mean': 4.2, 'std': 1.6},
            meter_patterns={'iambic': 0.7, 'free': 0.3},
            rhyme_patterns={'ABAB': 0.5, 'free': 0.5},
            average_syllables_per_line=8.2
        )
        
        similarities = profile1.calculate_similarity(profile2)
        
        # Check that all similarity metrics are present
        expected_keys = [
            'lexical_density', 'vocabulary_richness', 'pos_distribution',
            'meter_patterns', 'rhyme_patterns', 'syllable_patterns', 'overall'
        ]
        
        for key in expected_keys:
            assert key in similarities
            assert 0.0 <= similarities[key] <= 1.0
        
        # Overall similarity should be reasonable for similar profiles
        assert similarities['overall'] > 0.5
    
    def test_summary_stats(self):
        """Test summary statistics generation."""
        profile = PoetProfile(
            poet_name="Test Poet",
            corpus_size=10,
            total_lines=40,
            total_words=160,
            vocabulary_richness={'ttr': 0.8},
            lexical_density=0.7,
            pos_distribution={'NOUN': 0.3},
            word_length_stats={'mean': 4.5, 'std': 2.0},
            meter_patterns={'iambic_pentameter': 0.6, 'free_verse': 0.4},
            rhyme_patterns={'ABAB': 0.5, 'free': 0.5},
            average_syllables_per_line=9.0
        )
        
        stats = profile.get_summary_stats()
        
        assert stats['poet_name'] == "Test Poet"
        assert stats['corpus_size'] == 10
        assert stats['avg_words_per_line'] == 4.0  # 160/40
        assert stats['most_common_meter'] == 'iambic_pentameter'
        assert stats['most_common_rhyme'] == 'ABAB'
        assert stats['ttr'] == 0.8


class TestPoetProfileBuilder:
    """Test cases for PoetProfileBuilder."""
    
    def test_profile_builder_initialization(self):
        """Test profile builder initialization."""
        builder = PoetProfileBuilder()
        
        assert builder.text_processor is not None
        assert builder.lexical_analyzer is not None
        assert builder.structural_analyzer is not None
    
    def test_build_profile_from_texts_empty_list(self):
        """Test building profile from empty poem list raises error."""
        builder = PoetProfileBuilder()
        
        with pytest.raises(ValueError, match="Cannot build profile from empty poem list"):
            builder.build_profile_from_texts("Test Poet", [])
    
    def test_build_profile_from_texts_basic(self):
        """Test building profile from basic poem texts."""
        builder = PoetProfileBuilder()
        
        # Sample poems for testing
        poems = [
            """I never saw a moor,
            Yet know I how the heather looks,
            And what a wave must be.""",
            
            """Hope is the thing with feathers
            That perches in the soul,
            And sings the tune without the words,
            And never stops at all."""
        ]
        
        profile = builder.build_profile_from_texts("Emily Dickinson", poems)
        
        # Verify basic profile structure
        assert profile.poet_name == "Emily Dickinson"
        assert profile.corpus_size == 2
        assert profile.total_lines > 0
        assert profile.total_words > 0
        assert isinstance(profile.vocabulary_richness, dict)
        assert isinstance(profile.pos_distribution, dict)
        assert isinstance(profile.meter_patterns, dict)
        assert isinstance(profile.rhyme_patterns, dict)
        assert profile.average_syllables_per_line > 0
    
    def test_normalize_counts(self):
        """Test count normalization utility."""
        builder = PoetProfileBuilder()
        
        counts = {'A': 10, 'B': 20, 'C': 70}
        normalized = builder._normalize_counts(counts)
        
        assert abs(normalized['A'] - 0.1) < 0.001
        assert abs(normalized['B'] - 0.2) < 0.001
        assert abs(normalized['C'] - 0.7) < 0.001
        assert abs(sum(normalized.values()) - 1.0) < 0.001
    
    def test_normalize_counts_empty(self):
        """Test count normalization with empty dictionary."""
        builder = PoetProfileBuilder()
        
        counts = {}
        normalized = builder._normalize_counts(counts)
        
        assert normalized == {}


class TestPoetProfileIntegration:
    """Integration tests for complete stylometric analysis pipeline."""
    
    def test_complete_analysis_pipeline(self):
        """Test complete pipeline from text to profile analysis."""
        builder = PoetProfileBuilder()
        
        # Create two different poet profiles
        dickinson_poems = [
            """I'm nobody! Who are you?
            Are you nobody, too?
            Then there's a pair of us - don't tell!
            They'd banish us, you know.""",
            
            """Because I could not stop for Death,
            He kindly stopped for me;
            The carriage held but just ourselves
            And Immortality."""
        ]
        
        whitman_poems = [
            """I celebrate myself, and sing myself,
            And what I assume you shall assume,
            For every atom belonging to me as good belongs to you.""",
            
            """O Captain! my Captain! our fearful trip is done,
            The ship has weather'd every rack, the prize we sought is won."""
        ]
        
        # Build profiles
        dickinson_profile = builder.build_profile_from_texts("Emily Dickinson", dickinson_poems)
        whitman_profile = builder.build_profile_from_texts("Walt Whitman", whitman_poems)
        
        # Verify profiles are different
        assert dickinson_profile.poet_name != whitman_profile.poet_name
        assert dickinson_profile.corpus_size == 2
        assert whitman_profile.corpus_size == 2
        
        # Calculate similarity
        similarities = dickinson_profile.calculate_similarity(whitman_profile)
        
        # Verify similarity calculation works
        assert 'overall' in similarities
        assert 0.0 <= similarities['overall'] <= 1.0
        
        # Test serialization round-trip
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            dickinson_profile.save_json(temp_path)
            loaded_profile = PoetProfile.load_json(temp_path)
            
            # Verify loaded profile matches original
            assert loaded_profile.poet_name == dickinson_profile.poet_name
            assert loaded_profile.corpus_size == dickinson_profile.corpus_size
            
            # Verify similarity calculation still works with loaded profile
            loaded_similarities = loaded_profile.calculate_similarity(whitman_profile)
            assert abs(loaded_similarities['overall'] - similarities['overall']) < 0.001
            
        finally:
            Path(temp_path).unlink()
    
    def test_profile_comparison_consistency(self):
        """Test that profile comparisons are consistent and symmetric."""
        builder = PoetProfileBuilder()
        
        poems = [
            """Shall I compare thee to a summer's day?
            Thou art more lovely and more temperate.""",
            
            """When in eternal lines to time thou grow'st,
            So long as men can breathe or eyes can see."""
        ]
        
        profile1 = builder.build_profile_from_texts("Shakespeare", poems)
        profile2 = builder.build_profile_from_texts("Shakespeare Copy", poems)
        
        # Profiles built from same text should be very similar
        similarities = profile1.calculate_similarity(profile2)
        assert similarities['overall'] > 0.9
        
        # Similarity should be symmetric (approximately)
        reverse_similarities = profile2.calculate_similarity(profile1)
        assert abs(similarities['overall'] - reverse_similarities['overall']) < 0.01