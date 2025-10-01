"""Poet profile data model for storing and managing stylistic features."""

import json
import pickle
import math
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from .text_processing import PoetryTextProcessor
from .lexical_analysis import LexicalAnalyzer
from .structural_analysis import StructuralAnalyzer


@dataclass
class PoetProfile:
    """Data model for storing a poet's stylistic profile."""
    
    poet_name: str
    corpus_size: int
    total_lines: int
    total_words: int
    
    # Lexical features
    vocabulary_richness: Dict[str, float]
    lexical_density: float
    pos_distribution: Dict[str, float]
    word_length_stats: Dict[str, float]
    
    # Structural features
    meter_patterns: Dict[str, float]
    rhyme_patterns: Dict[str, float]
    average_syllables_per_line: float
    
    # Metadata
    analysis_version: str = "1.0"
    creation_timestamp: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.creation_timestamp is None:
            from datetime import datetime
            self.creation_timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PoetProfile':
        """Create profile from dictionary."""
        return cls(**data)
    
    def save_json(self, filepath: Union[str, Path]) -> None:
        """Save profile to JSON file."""
        filepath = Path(filepath)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_json(cls, filepath: Union[str, Path]) -> 'PoetProfile':
        """Load profile from JSON file."""
        filepath = Path(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def save_pickle(self, filepath: Union[str, Path]) -> None:
        """Save profile to pickle file."""
        filepath = Path(filepath)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_pickle(cls, filepath: Union[str, Path]) -> 'PoetProfile':
        """Load profile from pickle file."""
        filepath = Path(filepath)
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def calculate_similarity(self, other: 'PoetProfile') -> Dict[str, float]:
        """Calculate similarity metrics with another poet profile."""
        similarities = {}
        
        # Lexical similarity
        similarities['lexical_density'] = 1.0 - abs(self.lexical_density - other.lexical_density)
        similarities['vocabulary_richness'] = self._compare_vocabulary_richness(other)
        similarities['pos_distribution'] = self._cosine_similarity(self.pos_distribution, other.pos_distribution)
        
        # Structural similarity
        similarities['meter_patterns'] = self._cosine_similarity(self.meter_patterns, other.meter_patterns)
        similarities['rhyme_patterns'] = self._cosine_similarity(self.rhyme_patterns, other.rhyme_patterns)
        similarities['syllable_patterns'] = 1.0 - abs(
            self.average_syllables_per_line - other.average_syllables_per_line
        ) / max(self.average_syllables_per_line, other.average_syllables_per_line, 1.0)
        
        # Overall similarity (weighted average)
        weights = {
            'lexical_density': 0.2,
            'vocabulary_richness': 0.2,
            'pos_distribution': 0.2,
            'meter_patterns': 0.2,
            'rhyme_patterns': 0.1,
            'syllable_patterns': 0.1
        }
        
        overall_similarity = sum(
            similarities[key] * weights[key] 
            for key in weights.keys()
        )
        similarities['overall'] = overall_similarity
        
        return similarities
    
    def _compare_vocabulary_richness(self, other: 'PoetProfile') -> float:
        """Compare vocabulary richness metrics."""
        ttr_sim = 1.0 - abs(
            self.vocabulary_richness.get('ttr', 0) - 
            other.vocabulary_richness.get('ttr', 0)
        )
        
        # Compare other vocabulary metrics if available
        mean_sim = 1.0 - abs(
            self.vocabulary_richness.get('mean_word_length', 0) - 
            other.vocabulary_richness.get('mean_word_length', 0)
        ) / max(
            self.vocabulary_richness.get('mean_word_length', 1),
            other.vocabulary_richness.get('mean_word_length', 1), 1
        )
        
        return (ttr_sim + mean_sim) / 2
    
    def _cosine_similarity(self, dict1: Dict[str, float], dict2: Dict[str, float]) -> float:
        """Calculate cosine similarity between two dictionaries."""
        # Get all keys from both dictionaries
        all_keys = set(dict1.keys()) | set(dict2.keys())
        
        if not all_keys:
            return 1.0
        
        # Create vectors
        vec1 = [dict1.get(key, 0.0) for key in all_keys]
        vec2 = [dict2.get(key, 0.0) for key in all_keys]
        
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # Calculate magnitudes
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of the profile."""
        return {
            'poet_name': self.poet_name,
            'corpus_size': self.corpus_size,
            'total_lines': self.total_lines,
            'total_words': self.total_words,
            'avg_words_per_line': self.total_words / max(self.total_lines, 1),
            'lexical_density': self.lexical_density,
            'ttr': self.vocabulary_richness.get('ttr', 0),
            'most_common_meter': max(self.meter_patterns.items(), key=lambda x: x[1])[0] if self.meter_patterns else 'unknown',
            'most_common_rhyme': max(self.rhyme_patterns.items(), key=lambda x: x[1])[0] if self.rhyme_patterns else 'unknown',
            'average_syllables_per_line': self.average_syllables_per_line
        }


class PoetProfileBuilder:
    """Builder class for creating poet profiles from text corpora."""
    
    def __init__(self):
        """Initialize the profile builder."""
        self.text_processor = PoetryTextProcessor()
        self.lexical_analyzer = LexicalAnalyzer()
        self.structural_analyzer = StructuralAnalyzer()
    
    def build_profile_from_texts(self, poet_name: str, poems: List[str]) -> PoetProfile:
        """Build a poet profile from a list of poem texts."""
        if not poems:
            raise ValueError("Cannot build profile from empty poem list")
        
        # Initialize counters
        total_lines = 0
        total_words = 0
        all_lexical_features = []
        all_structural_features = []
        syllable_counts = []
        meter_counts = {}
        rhyme_counts = {}
        stanza_counts = {}
        
        # Process each poem
        for poem in poems:
            poem = poem.strip()
            if not poem:
                continue
            
            # Get basic text processing
            processed = self.text_processor.preprocess_for_analysis(poem)
            
            # Update counters
            total_lines += len(processed['lines'])
            total_words += len(processed['words'])
            
            # Collect syllable counts
            syllable_counts.extend(processed['syllable_counts'])
            
            # Analyze lexical features
            lexical_features = self.lexical_analyzer.analyze_lexical_features(poem)
            all_lexical_features.append(lexical_features)
            
            # Analyze structural features
            structural_features = self.structural_analyzer.analyze_structural_features(poem)
            all_structural_features.append(structural_features)
            
            # Count meter patterns
            for meter_analysis in structural_features['meter_analysis']:
                meter_type = meter_analysis['meter_type']
                meter_counts[meter_type] = meter_counts.get(meter_type, 0) + 1
            
            # Count rhyme patterns
            for rhyme_analysis in structural_features['rhyme_analysis']:
                rhyme_type = rhyme_analysis['rhyme_type']
                rhyme_counts[rhyme_type] = rhyme_counts.get(rhyme_type, 0) + 1
            
            # Count stanza patterns (simplified)
            stanza_count = len(processed['stanzas'])
            stanza_counts[stanza_count] = stanza_counts.get(stanza_count, 0) + 1
        
        # Aggregate lexical features
        aggregated_lexical = self._aggregate_lexical_features(all_lexical_features)
        
        # Calculate distributions
        pos_distribution = self._calculate_pos_distribution(all_lexical_features)
        meter_patterns = self._normalize_counts(meter_counts)
        rhyme_patterns = self._normalize_counts(rhyme_counts)
        
        # Calculate average syllables per line
        average_syllables_per_line = sum(syllable_counts) / len(syllable_counts) if syllable_counts else 0
        
        return PoetProfile(
            poet_name=poet_name,
            corpus_size=len(poems),
            total_lines=total_lines,
            total_words=total_words,
            vocabulary_richness=aggregated_lexical['vocabulary_richness'],
            lexical_density=aggregated_lexical['lexical_density'],
            pos_distribution=pos_distribution,
            word_length_stats=aggregated_lexical['word_length_stats'],
            meter_patterns=meter_patterns,
            rhyme_patterns=rhyme_patterns,
            average_syllables_per_line=average_syllables_per_line
        )
    
    def _aggregate_lexical_features(self, all_features: List[Dict]) -> Dict[str, Any]:
        """Aggregate lexical features across all poems."""
        if not all_features:
            return {
                'vocabulary_richness': {'ttr': 0.0},
                'lexical_density': 0.0,
                'word_length_stats': {'mean': 0.0, 'std': 0.0}
            }
        
        # Calculate averages for key metrics
        ttr_values = [f['vocabulary_richness']['ttr'] for f in all_features if 'vocabulary_richness' in f]
        density_values = [f['lexical_density'] for f in all_features if 'lexical_density' in f]
        
        avg_ttr = sum(ttr_values) / len(ttr_values) if ttr_values else 0.0
        avg_density = sum(density_values) / len(density_values) if density_values else 0.0
        
        # Aggregate word length statistics
        word_lengths = []
        for features in all_features:
            if 'word_length_distribution' in features:
                word_lengths.extend(features['word_length_distribution'])
        
        if word_lengths:
            mean_length = sum(word_lengths) / len(word_lengths)
            variance = sum((x - mean_length) ** 2 for x in word_lengths) / len(word_lengths)
            std_length = math.sqrt(variance)
        else:
            mean_length = 0.0
            std_length = 0.0
        
        return {
            'vocabulary_richness': {'ttr': avg_ttr, 'mean_word_length': mean_length},
            'lexical_density': avg_density,
            'word_length_stats': {'mean': mean_length, 'std': std_length}
        }
    
    def _calculate_pos_distribution(self, all_features: List[Dict]) -> Dict[str, float]:
        """Calculate POS tag distribution across all poems."""
        pos_counts = {}
        total_tags = 0
        
        for features in all_features:
            if 'pos_distribution' in features:
                for pos, count in features['pos_distribution'].items():
                    pos_counts[pos] = pos_counts.get(pos, 0) + count
                    total_tags += count
        
        # Normalize to percentages
        if total_tags > 0:
            return {pos: count / total_tags for pos, count in pos_counts.items()}
        else:
            return {}
    
    def _normalize_counts(self, counts: Dict[str, int]) -> Dict[str, float]:
        """Normalize count dictionary to percentages."""
        total = sum(counts.values())
        if total > 0:
            return {k: v / total for k, v in counts.items()}
        else:
            return {}


class PoetProfileManager:
    """Manager class for poet profiles with loading, saving, and comparison functionality."""
    
    def __init__(self, profiles_dir: Optional[Path] = None):
        """
        Initialize the poet profile manager.
        
        Args:
            profiles_dir: Directory to store poet profiles (default: ./data/profiles)
        """
        self.profiles_dir = profiles_dir or Path("./data/profiles")
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self._loaded_profiles: Dict[str, PoetProfile] = {}
    
    def load_profile(self, poet_name: str) -> Optional[PoetProfile]:
        """
        Load a poet profile from disk.
        
        Args:
            poet_name: Name of the poet
            
        Returns:
            PoetProfile if found, None otherwise
        """
        if poet_name in self._loaded_profiles:
            return self._loaded_profiles[poet_name]
        
        profile_path = self.profiles_dir / f"{poet_name}.json"
        
        if profile_path.exists():
            try:
                with open(profile_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                profile = PoetProfile.from_dict(data)
                self._loaded_profiles[poet_name] = profile
                return profile
                
            except Exception as e:
                print(f"Error loading profile for {poet_name}: {e}")
                return None
        
        return None
    
    def save_profile(self, profile: PoetProfile) -> bool:
        """
        Save a poet profile to disk.
        
        Args:
            profile: PoetProfile to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            profile_path = self.profiles_dir / f"{profile.poet_name}.json"
            
            with open(profile_path, 'w', encoding='utf-8') as f:
                json.dump(profile.to_dict(), f, indent=2, ensure_ascii=False)
            
            # Cache the profile
            self._loaded_profiles[profile.poet_name] = profile
            return True
            
        except Exception as e:
            print(f"Error saving profile for {profile.poet_name}: {e}")
            return False
    
    def list_available_poets(self) -> List[str]:
        """
        List all available poet profiles.
        
        Returns:
            List of poet names
        """
        poets = []
        
        # Add loaded profiles
        poets.extend(self._loaded_profiles.keys())
        
        # Add profiles from disk
        for profile_file in self.profiles_dir.glob("*.json"):
            poet_name = profile_file.stem
            if poet_name not in poets:
                poets.append(poet_name)
        
        return sorted(poets)
    
    def create_profile_from_corpus(self, poet_name: str, corpus_path: Path) -> PoetProfile:
        """
        Create a poet profile from a corpus directory or file.
        
        Args:
            poet_name: Name of the poet
            corpus_path: Path to corpus file or directory
            
        Returns:
            Created PoetProfile
        """
        builder = PoetProfileBuilder()
        
        if corpus_path.is_file():
            # Single file
            with open(corpus_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into poems (simple approach)
            poems = [poem.strip() for poem in content.split('\n\n') if poem.strip()]
            
        elif corpus_path.is_dir():
            # Directory of files
            poems = []
            for file_path in corpus_path.glob("*.txt"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        poems.append(content)
        else:
            raise ValueError(f"Corpus path does not exist: {corpus_path}")
        
        profile = builder.build_profile(poems, poet_name)
        
        # Save the profile
        self.save_profile(profile)
        
        return profile
    
    def compare_profiles(self, profile1: PoetProfile, profile2: PoetProfile) -> Dict[str, float]:
        """
        Compare two poet profiles and return similarity scores.
        
        Args:
            profile1: First poet profile
            profile2: Second poet profile
            
        Returns:
            Dictionary of similarity scores for different aspects
        """
        similarities = {}
        
        # Compare lexical features
        similarities['lexical_density'] = 1.0 - abs(profile1.lexical_density - profile2.lexical_density)
        
        # Compare vocabulary richness
        ttr1 = profile1.vocabulary_richness.get('ttr', 0)
        ttr2 = profile2.vocabulary_richness.get('ttr', 0)
        similarities['vocabulary_richness'] = 1.0 - abs(ttr1 - ttr2)
        
        # Overall similarity (simplified)
        similarities['overall'] = (similarities['lexical_density'] + similarities['vocabulary_richness']) / 2
        
        return similarities
    
    def validate_profile(self, profile: PoetProfile) -> bool:
        """
        Validate a poet profile for completeness and consistency.
        
        Args:
            profile: PoetProfile to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required fields
            if not profile.poet_name:
                return False
            
            if profile.corpus_size <= 0:
                return False
            
            if profile.total_lines <= 0:
                return False
            
            # Check lexical features
            if not isinstance(profile.vocabulary_richness, dict):
                return False
            
            if not (0 <= profile.lexical_density <= 1):
                return False
            
            return True
            
        except Exception:
            return False