"""Training data preparation system for poetry fine-tuning."""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
import logging

from .text_processing import PoetryTextProcessor
from .lexical_analysis import LexicalAnalyzer
from .structural_analysis import StructuralAnalyzer
from .poet_profile import PoetProfile, PoetProfileBuilder

# Import error handling utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.exceptions import (
    PoetryLLMError, DataQualityError, ValidationError,
    ErrorContext, create_user_friendly_error_message
)
from utils.error_handlers import DataQualityValidator, ErrorHandlingCoordinator

logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """Data model for a single training example."""
    
    instruction: str
    input_text: str
    output_text: str
    stylometric_features: Dict[str, Any]
    poet_name: str
    poem_title: Optional[str] = None
    line_number: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingExample':
        """Create from dictionary."""
        return cls(**data)


class PoetryCorpusLoader:
    """Loads and parses poetry corpora from various text file formats."""
    
    def __init__(self):
        """Initialize the corpus loader."""
        self.text_processor = PoetryTextProcessor()
        self.data_validator = DataQualityValidator()
        self.error_coordinator = ErrorHandlingCoordinator()
    
    def load_corpus_from_file(self, filepath: Union[str, Path], 
                            poet_name: str,
                            format_type: str = "auto") -> List[Dict[str, Any]]:
        """
        Load poetry corpus from a text file with comprehensive error handling.
        
        Args:
            filepath: Path to the corpus file
            poet_name: Name of the poet
            format_type: Format type ('auto', 'simple', 'gutenberg', 'json')
            
        Returns:
            List of poem dictionaries with metadata
            
        Raises:
            DataQualityError: If corpus loading or validation fails
        """
        filepath = Path(filepath)
        
        # Validate file existence
        if not filepath.exists():
            error_context = {
                'operation': 'corpus_loading',
                'filepath': str(filepath),
                'poet_name': poet_name,
                'component': 'data_processing'
            }
            
            error = FileNotFoundError(f"Corpus file not found: {filepath}")
            result = self.error_coordinator.handle_component_error('data_processing', error, error_context)
            
            raise DataQualityError(
                f"Failed to load corpus for {poet_name}: {result['error_message']}",
                error_code="FILE_NOT_FOUND",
                details=result
            )
        
        logger.info(f"Loading corpus for {poet_name} from {filepath}")
        
        # Load file content with encoding fallback
        content = None
        encoding_errors = []
        
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    content = f.read()
                logger.debug(f"Successfully loaded file with {encoding} encoding")
                break
            except UnicodeDecodeError as e:
                encoding_errors.append(f"{encoding}: {str(e)}")
                continue
            except Exception as e:
                error_context = {
                    'operation': 'file_reading',
                    'filepath': str(filepath),
                    'poet_name': poet_name,
                    'encoding': encoding,
                    'component': 'data_processing'
                }
                
                result = self.error_coordinator.handle_component_error('data_processing', e, error_context)
                
                raise DataQualityError(
                    f"Failed to read corpus file: {result['error_message']}",
                    error_code="FILE_READ_ERROR",
                    details=result
                )
        
        if content is None:
            error_context = {
                'operation': 'encoding_detection',
                'filepath': str(filepath),
                'poet_name': poet_name,
                'encoding_errors': encoding_errors,
                'component': 'data_processing'
            }
            
            error = UnicodeDecodeError('multiple', b'', 0, 1, 'All encoding attempts failed')
            result = self.error_coordinator.handle_component_error('data_processing', error, error_context)
            
            raise DataQualityError(
                f"Could not decode corpus file with any supported encoding",
                error_code="ENCODING_ERROR",
                details=result
            )
        
        # Detect and parse format
        try:
            if format_type == "auto":
                format_type = self._detect_format(content)
            
            if format_type == "json":
                poems = self._parse_json_format(content, poet_name)
            elif format_type == "gutenberg":
                poems = self._parse_gutenberg_format(content, poet_name)
            else:
                poems = self._parse_simple_format(content, poet_name)
            
            # Validate parsed content
            if not poems:
                raise DataQualityError(
                    f"No poems found in corpus file {filepath}",
                    error_code="EMPTY_CORPUS",
                    details={
                        'filepath': str(filepath),
                        'poet_name': poet_name,
                        'format_type': format_type,
                        'content_length': len(content)
                    }
                )
            
            logger.info(f"Successfully loaded {len(poems)} poems for {poet_name}")
            return poems
            
        except DataQualityError:
            raise  # Re-raise data quality errors
        except Exception as e:
            error_context = {
                'operation': 'corpus_parsing',
                'filepath': str(filepath),
                'poet_name': poet_name,
                'format_type': format_type,
                'content_preview': content[:200] if content else '',
                'component': 'data_processing'
            }
            
            result = self.error_coordinator.handle_component_error('data_processing', e, error_context)
            
            raise DataQualityError(
                f"Failed to parse corpus file: {result['error_message']}",
                error_code="PARSING_ERROR",
                details=result
            )
    
    def load_corpus_from_directory(self, directory: Union[str, Path],
                                   poet_name: str,
                                   file_pattern: str = "*.txt") -> List[Dict[str, Any]]:
        """
        Load poetry corpus from multiple files in a directory.
        
        Args:
            directory: Path to directory containing corpus files
            poet_name: Name of the poet
            file_pattern: Glob pattern for files to include
            
        Returns:
            List of poem dictionaries with metadata
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Corpus directory not found: {directory}")
        
        poems = []
        files = list(directory.glob(file_pattern))
        
        if not files:
            raise ValueError(f"No files found matching pattern {file_pattern} in {directory}")
        
        logger.info(f"Loading corpus for {poet_name} from {len(files)} files")
        
        for filepath in files:
            try:
                file_poems = self.load_corpus_from_file(filepath, poet_name)
                poems.extend(file_poems)
            except Exception as e:
                logger.warning(f"Failed to load {filepath}: {e}")
                continue
        
        return poems
    
    def _detect_format(self, content: str) -> str:
        """Detect the format of the corpus content."""
        content_stripped = content.strip()
        
        if content_stripped.startswith('{') or content_stripped.startswith('['):
            return "json"
        elif "*** START OF" in content and "*** END OF" in content:
            return "gutenberg"
        else:
            return "simple"
    
    def _parse_json_format(self, content: str, poet_name: str) -> List[Dict[str, Any]]:
        """Parse JSON format corpus with error handling."""
        try:
            data = json.loads(content)
            
            if isinstance(data, list):
                poems = data
            elif isinstance(data, dict) and 'poems' in data:
                poems = data['poems']
            else:
                # Try to handle malformed JSON gracefully
                logger.warning(f"Unrecognized JSON structure for {poet_name}, attempting fallback parsing")
                if isinstance(data, dict):
                    # Look for any list values that might contain poems
                    for key, value in data.items():
                        if isinstance(value, list) and len(value) > 0:
                            poems = value
                            break
                    else:
                        raise ValueError("No poem list found in JSON structure")
                else:
                    raise ValueError("JSON format not recognized")
            
            result = []
            parsing_errors = []
            
            for i, poem_data in enumerate(poems):
                try:
                    if isinstance(poem_data, str):
                        poem_text = poem_data
                        title = f"Poem {i+1}"
                    elif isinstance(poem_data, dict):
                        poem_text = poem_data.get('text', poem_data.get('content', poem_data.get('poem', '')))
                        title = poem_data.get('title', poem_data.get('name', f"Poem {i+1}"))
                    else:
                        parsing_errors.append(f"Poem {i+1}: Invalid data type {type(poem_data)}")
                        continue
                    
                    if poem_text and poem_text.strip():
                        result.append({
                            'text': poem_text.strip(),
                            'title': title,
                            'poet': poet_name,
                            'source_format': 'json'
                        })
                    else:
                        parsing_errors.append(f"Poem {i+1}: Empty or missing text")
                        
                except Exception as e:
                    parsing_errors.append(f"Poem {i+1}: {str(e)}")
                    continue
            
            if parsing_errors:
                logger.warning(f"JSON parsing issues for {poet_name}: {'; '.join(parsing_errors[:5])}")
            
            if not result:
                raise ValueError(f"No valid poems extracted from JSON. Errors: {'; '.join(parsing_errors[:3])}")
            
            return result
            
        except json.JSONDecodeError as e:
            # Try to recover from malformed JSON
            logger.warning(f"JSON decode error for {poet_name}: {e}. Attempting fallback parsing.")
            
            # Try to extract text that looks like poems from malformed JSON
            try:
                # Look for text patterns that might be poems
                import re
                text_patterns = re.findall(r'"text"\s*:\s*"([^"]+)"', content)
                if not text_patterns:
                    text_patterns = re.findall(r'"content"\s*:\s*"([^"]+)"', content)
                if not text_patterns:
                    text_patterns = re.findall(r'"poem"\s*:\s*"([^"]+)"', content)
                
                if text_patterns:
                    result = []
                    for i, text in enumerate(text_patterns):
                        if text.strip():
                            result.append({
                                'text': text.strip().replace('\\n', '\n'),
                                'title': f"Poem {i+1}",
                                'poet': poet_name,
                                'source_format': 'json_recovered'
                            })
                    
                    if result:
                        logger.info(f"Recovered {len(result)} poems from malformed JSON for {poet_name}")
                        return result
                
            except Exception as recovery_error:
                logger.error(f"JSON recovery failed for {poet_name}: {recovery_error}")
            
            raise ValueError(f"Invalid JSON format and recovery failed: {e}")
    
    def _parse_gutenberg_format(self, content: str, poet_name: str) -> List[Dict[str, Any]]:
        """Parse Project Gutenberg format corpus."""
        # Find the actual content between START and END markers
        start_pattern = r'\*\*\* START OF .*? \*\*\*'
        end_pattern = r'\*\*\* END OF .*? \*\*\*'
        
        start_match = re.search(start_pattern, content, re.IGNORECASE)
        end_match = re.search(end_pattern, content, re.IGNORECASE)
        
        if start_match and end_match:
            content = content[start_match.end():end_match.start()]
        
        return self._parse_simple_format(content, poet_name)
    
    def _parse_simple_format(self, content: str, poet_name: str) -> List[Dict[str, Any]]:
        """Parse simple text format corpus."""
        # Split by double newlines or more to separate poems
        poem_texts = re.split(r'\n\s*\n\s*\n+', content.strip())
        
        poems = []
        for i, poem_text in enumerate(poem_texts):
            poem_text = poem_text.strip()
            if not poem_text:
                continue
            
            # Try to extract title from first line if it looks like a title
            lines = poem_text.split('\n')
            if len(lines) > 1:
                first_line = lines[0].strip()
                # Simple heuristic: if first line is short and doesn't end with punctuation
                if (len(first_line) < 50 and 
                    not first_line.endswith(('.', '!', '?', ';', ':')) and
                    len(lines) > 2):
                    title = first_line
                    poem_text = '\n'.join(lines[1:]).strip()
                else:
                    title = f"Poem {i+1}"
            else:
                title = f"Poem {i+1}"
            
            if poem_text:
                poems.append({
                    'text': poem_text,
                    'title': title,
                    'poet': poet_name,
                    'source_format': 'simple'
                })
        
        return poems
    
    def validate_corpus_quality(self, poems: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate the quality of loaded corpus data.
        
        Args:
            poems: List of poem dictionaries
            
        Returns:
            Dictionary with quality metrics and warnings
        """
        if not poems:
            return {
                'valid': False,
                'warnings': ['No poems found in corpus'],
                'metrics': {}
            }
        
        warnings = []
        metrics = {
            'total_poems': len(poems),
            'total_lines': 0,
            'total_words': 0,
            'avg_lines_per_poem': 0,
            'avg_words_per_poem': 0,
            'short_poems': 0,
            'empty_poems': 0
        }
        
        for poem in poems:
            text = poem.get('text', '')
            if not text.strip():
                metrics['empty_poems'] += 1
                continue
            
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            words = text.split()
            
            metrics['total_lines'] += len(lines)
            metrics['total_words'] += len(words)
            
            if len(lines) < 4:  # Very short poems
                metrics['short_poems'] += 1
        
        if metrics['total_poems'] > 0:
            metrics['avg_lines_per_poem'] = metrics['total_lines'] / metrics['total_poems']
            metrics['avg_words_per_poem'] = metrics['total_words'] / metrics['total_poems']
        
        # Generate warnings
        if metrics['empty_poems'] > 0:
            warnings.append(f"{metrics['empty_poems']} empty poems found")
        
        if metrics['short_poems'] > metrics['total_poems'] * 0.3:
            warnings.append(f"High proportion of very short poems ({metrics['short_poems']}/{metrics['total_poems']})")
        
        if metrics['total_poems'] < 10:
            warnings.append("Very small corpus size - may not be sufficient for training")
        
        if metrics['avg_lines_per_poem'] < 4:
            warnings.append("Average poem length is very short")
        
        return {
            'valid': metrics['empty_poems'] < metrics['total_poems'],
            'warnings': warnings,
            'metrics': metrics
        }


class StylemetricFeatureEncoder:
    """Encodes stylometric features as metadata tags for training data."""
    
    def __init__(self):
        """Initialize the feature encoder."""
        self.text_processor = PoetryTextProcessor()
        self.lexical_analyzer = LexicalAnalyzer()
        self.structural_analyzer = StructuralAnalyzer()
    
    def encode_line_features(self, line: str, poem_context: str, 
                           line_index: int, poet_profile: Optional[PoetProfile] = None) -> Dict[str, Any]:
        """
        Encode stylometric features for a single line of poetry.
        
        Args:
            line: The line of poetry to analyze
            poem_context: The full poem text for context
            line_index: Index of the line within the poem
            poet_profile: Optional poet profile for comparison
            
        Returns:
            Dictionary of encoded features
        """
        features = {}
        
        # Basic line features
        features['line_length'] = len(line.split())
        features['character_count'] = len(line)
        features['syllable_count'] = self.text_processor.count_syllables_line(line)
        
        # Lexical features
        words = line.split()
        if words:
            features['avg_word_length'] = sum(len(word) for word in words) / len(words)
            features['lexical_density'] = self._calculate_line_lexical_density(line)
        else:
            features['avg_word_length'] = 0
            features['lexical_density'] = 0
        
        # Structural features
        features['rhyme_position'] = self._get_rhyme_position(line, poem_context, line_index)
        features['meter_pattern'] = self._get_meter_pattern(line)
        features['punctuation_pattern'] = self._get_punctuation_pattern(line)
        
        # Stylistic markers
        features['capitalization_pattern'] = self._get_capitalization_pattern(line)
        features['special_characters'] = self._count_special_characters(line)
        
        # Poet-specific features (if profile available)
        if poet_profile:
            features['deviation_from_avg_syllables'] = (
                features['syllable_count'] - poet_profile.average_syllables_per_line
            )
        
        return features
    
    def encode_poem_features(self, poem_text: str, 
                           poet_profile: Optional[PoetProfile] = None) -> Dict[str, Any]:
        """
        Encode stylometric features for an entire poem.
        
        Args:
            poem_text: The full poem text
            poet_profile: Optional poet profile for comparison
            
        Returns:
            Dictionary of encoded features
        """
        features = {}
        
        # Basic poem structure
        lines = [line.strip() for line in poem_text.split('\n') if line.strip()]
        stanzas = self.text_processor.segment_stanzas(poem_text)
        
        features['total_lines'] = len(lines)
        features['total_stanzas'] = len(stanzas)
        features['avg_lines_per_stanza'] = len(lines) / len(stanzas) if stanzas else 0
        
        # Lexical features for the whole poem
        lexical_analysis = self.lexical_analyzer.analyze_lexical_features(poem_text)
        features['poem_ttr'] = lexical_analysis['vocabulary_richness']['ttr']
        features['poem_lexical_density'] = lexical_analysis['lexical_density']
        
        # Structural features
        structural_analysis = self.structural_analyzer.analyze_structural_features(poem_text)
        features['dominant_meter'] = self._get_dominant_meter(structural_analysis)
        features['rhyme_scheme'] = self._get_rhyme_scheme(structural_analysis)
        
        return features
    
    def _calculate_line_lexical_density(self, line: str) -> float:
        """Calculate lexical density for a single line."""
        words = line.split()
        if not words:
            return 0.0
        
        # Simple approximation: ratio of content words to total words
        # Content words are typically longer than function words
        content_words = [w for w in words if len(w) > 3]
        return len(content_words) / len(words) if words else 0.0
    
    def _get_rhyme_position(self, line: str, poem_context: str, line_index: int) -> str:
        """Determine the rhyme position of the line within the poem."""
        lines = [l.strip() for l in poem_context.split('\n') if l.strip()]
        if line_index >= len(lines):
            return "unknown"
        
        # Simple heuristic based on position
        total_lines = len(lines)
        if line_index == total_lines - 1:
            return "end"
        elif line_index % 2 == 0:
            return "even"
        else:
            return "odd"
    
    def _get_meter_pattern(self, line: str) -> str:
        """Get a simple meter pattern for the line."""
        syllable_count = self.text_processor.count_syllables_line(line)
        
        # Simple classification based on syllable count
        if syllable_count <= 6:
            return "short"
        elif syllable_count <= 10:
            return "medium"
        else:
            return "long"
    
    def _get_punctuation_pattern(self, line: str) -> Dict[str, int]:
        """Get punctuation pattern for the line."""
        punctuation_counts = {
            'comma': line.count(','),
            'period': line.count('.'),
            'dash': line.count('-') + line.count('—'),
            'question': line.count('?'),
            'exclamation': line.count('!'),
            'semicolon': line.count(';'),
            'colon': line.count(':')
        }
        return punctuation_counts
    
    def _get_capitalization_pattern(self, line: str) -> Dict[str, Any]:
        """Get capitalization pattern for the line."""
        words = line.split()
        if not words:
            return {'capitalized_words': 0, 'all_caps_words': 0, 'irregular_caps': False}
        
        capitalized = sum(1 for word in words if word and word[0].isupper())
        all_caps = sum(1 for word in words if word.isupper() and len(word) > 1)
        
        # Check for irregular capitalization (not just first word)
        irregular = capitalized > 1 and not all(word[0].isupper() for word in words)
        
        return {
            'capitalized_words': capitalized,
            'all_caps_words': all_caps,
            'irregular_caps': irregular
        }
    
    def _count_special_characters(self, line: str) -> Dict[str, int]:
        """Count special characters in the line."""
        return {
            'dashes': line.count('-') + line.count('—'),
            'apostrophes': line.count("'"),
            'quotation_marks': line.count('"') + line.count('"') + line.count('"'),
            'parentheses': line.count('(') + line.count(')'),
            'brackets': line.count('[') + line.count(']')
        }
    
    def _get_dominant_meter(self, structural_analysis: Dict[str, Any]) -> str:
        """Extract dominant meter from structural analysis."""
        # This would use the structural analyzer results
        # For now, return a placeholder
        return "iambic"
    
    def _get_rhyme_scheme(self, structural_analysis: Dict[str, Any]) -> str:
        """Extract rhyme scheme from structural analysis."""
        # This would use the structural analyzer results
        # For now, return a placeholder
        return "ABAB"


class TrainingDatasetFormatter:
    """Formats poetry data into instruction-output pairs for supervised fine-tuning."""
    
    def __init__(self):
        """Initialize the dataset formatter."""
        self.feature_encoder = StylemetricFeatureEncoder()
        self.augmentation_strategies = [
            'style_variation',
            'length_variation', 
            'structural_variation',
            'thematic_variation'
        ]
    
    def create_instruction_output_pairs(self, poems: List[Dict[str, Any]], 
                                      poet_profile: Optional[PoetProfile] = None) -> List[TrainingExample]:
        """
        Create instruction-output pairs from poetry corpus.
        
        Args:
            poems: List of poem dictionaries
            poet_profile: Optional poet profile for feature encoding
            
        Returns:
            List of TrainingExample objects
        """
        training_examples = []
        
        for poem in poems:
            poem_text = poem['text']
            poet_name = poem['poet']
            title = poem.get('title', 'Untitled')
            
            # Create different types of training examples
            examples = []
            
            # 1. Complete poem generation
            examples.extend(self._create_complete_poem_examples(poem_text, poet_name, title, poet_profile))
            
            # 2. Line completion examples
            examples.extend(self._create_line_completion_examples(poem_text, poet_name, title, poet_profile))
            
            # 3. Style-specific generation examples
            examples.extend(self._create_style_specific_examples(poem_text, poet_name, title, poet_profile))
            
            training_examples.extend(examples)
        
        return training_examples
    
    def _create_complete_poem_examples(self, poem_text: str, poet_name: str, 
                                     title: str, poet_profile: Optional[PoetProfile]) -> List[TrainingExample]:
        """Create examples for complete poem generation."""
        examples = []
        
        # Encode poem-level features
        poem_features = self.feature_encoder.encode_poem_features(poem_text, poet_profile)
        
        # Basic generation instruction
        instruction = f"Write a poem in the style of {poet_name}."
        examples.append(TrainingExample(
            instruction=instruction,
            input_text="",
            output_text=poem_text,
            stylometric_features=poem_features,
            poet_name=poet_name,
            poem_title=title
        ))
        
        # Style-specific instruction
        if poet_profile:
            style_instruction = f"Write a poem in the style of {poet_name} with {poem_features['total_lines']} lines."
            examples.append(TrainingExample(
                instruction=style_instruction,
                input_text="",
                output_text=poem_text,
                stylometric_features=poem_features,
                poet_name=poet_name,
                poem_title=title
            ))
        
        return examples
    
    def _create_line_completion_examples(self, poem_text: str, poet_name: str,
                                       title: str, poet_profile: Optional[PoetProfile]) -> List[TrainingExample]:
        """Create examples for line completion."""
        examples = []
        lines = [line.strip() for line in poem_text.split('\n') if line.strip()]
        
        if len(lines) < 2:
            return examples
        
        # Create examples with partial context
        for i in range(1, min(len(lines), 5)):  # Limit to first few lines
            context = '\n'.join(lines[:i])
            target_line = lines[i]
            
            # Encode features for the target line
            line_features = self.feature_encoder.encode_line_features(
                target_line, poem_text, i, poet_profile
            )
            
            instruction = f"Continue this poem in the style of {poet_name}:"
            examples.append(TrainingExample(
                instruction=instruction,
                input_text=context,
                output_text=target_line,
                stylometric_features=line_features,
                poet_name=poet_name,
                poem_title=title,
                line_number=i
            ))
        
        return examples
    
    def _create_style_specific_examples(self, poem_text: str, poet_name: str,
                                      title: str, poet_profile: Optional[PoetProfile]) -> List[TrainingExample]:
        """Create examples with explicit style markers."""
        examples = []
        
        # Encode poem features
        poem_features = self.feature_encoder.encode_poem_features(poem_text, poet_profile)
        
        # Create instruction with style markers
        style_markers = []
        if poem_features['total_lines'] <= 4:
            style_markers.append("short form")
        elif poem_features['total_lines'] >= 12:
            style_markers.append("long form")
        
        if poem_features.get('rhyme_scheme') and poem_features['rhyme_scheme'] != "free":
            style_markers.append(f"with {poem_features['rhyme_scheme']} rhyme scheme")
        
        if style_markers:
            instruction = f"Write a {', '.join(style_markers)} poem in the style of {poet_name}."
            examples.append(TrainingExample(
                instruction=instruction,
                input_text="",
                output_text=poem_text,
                stylometric_features=poem_features,
                poet_name=poet_name,
                poem_title=title
            ))
        
        return examples
    
    def format_for_huggingface(self, training_examples: List[TrainingExample],
                             format_style: str = 'instruction_following') -> List[Dict[str, Any]]:
        """
        Format training examples for HuggingFace transformers.
        
        Args:
            training_examples: List of TrainingExample objects
            format_style: Style of formatting ('instruction_following', 'chat', 'completion')
            
        Returns:
            List of dictionaries compatible with HuggingFace datasets
        """
        formatted_examples = []
        
        for example in training_examples:
            if format_style == 'instruction_following':
                formatted_example = self._format_instruction_following(example)
            elif format_style == 'chat':
                formatted_example = self._format_chat_style(example)
            elif format_style == 'completion':
                formatted_example = self._format_completion_style(example)
            else:
                raise ValueError(f"Unknown format style: {format_style}")
            
            formatted_examples.append(formatted_example)
        
        return formatted_examples
    
    def _format_instruction_following(self, example: TrainingExample) -> Dict[str, Any]:
        """Format for instruction-following fine-tuning."""
        # Create the full prompt
        if example.input_text:
            prompt = f"{example.instruction}\n\n{example.input_text}"
        else:
            prompt = example.instruction
        
        return {
            'instruction': example.instruction,
            'input': example.input_text or "",
            'output': example.output_text,
            'poet_name': example.poet_name,
            'stylometric_features': json.dumps(example.stylometric_features),
            'poem_title': example.poem_title,
            'line_number': example.line_number
        }
    
    def _format_chat_style(self, example: TrainingExample) -> Dict[str, Any]:
        """Format for chat-based fine-tuning."""
        messages = [
            {"role": "user", "content": example.instruction}
        ]
        
        if example.input_text:
            messages.append({"role": "user", "content": example.input_text})
        
        messages.append({"role": "assistant", "content": example.output_text})
        
        return {
            'messages': messages,
            'poet_name': example.poet_name,
            'stylometric_features': json.dumps(example.stylometric_features),
            'poem_title': example.poem_title,
            'line_number': example.line_number
        }
    
    def _format_completion_style(self, example: TrainingExample) -> Dict[str, Any]:
        """Format for completion-style fine-tuning."""
        if example.input_text:
            prompt = f"{example.instruction}\n\n{example.input_text}"
        else:
            prompt = example.instruction
        
        return {
            'prompt': prompt,
            'completion': example.output_text,
            'poet_name': example.poet_name,
            'stylometric_features': json.dumps(example.stylometric_features),
            'poem_title': example.poem_title,
            'line_number': example.line_number
        }
    
    def create_dataset_splits(self, training_examples: List[TrainingExample],
                            train_ratio: float = 0.8, val_ratio: float = 0.1,
                            test_ratio: float = 0.1) -> Dict[str, List[TrainingExample]]:
        """
        Split dataset into train/validation/test sets.
        
        Args:
            training_examples: List of training examples
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set  
            test_ratio: Proportion for test set
            
        Returns:
            Dictionary with 'train', 'val', 'test' splits
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        import random
        shuffled = training_examples.copy()
        random.shuffle(shuffled)
        
        total = len(shuffled)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        splits = {
            'train': shuffled[:train_end],
            'val': shuffled[train_end:val_end],
            'test': shuffled[val_end:]
        }
        
        logger.info(f"Dataset splits - Train: {len(splits['train'])}, "
                   f"Val: {len(splits['val'])}, Test: {len(splits['test'])}")
        
        return splits
    
    def save_training_data(self, training_examples: List[TrainingExample], 
                          output_path: Union[str, Path], format_type: str = "jsonl") -> None:
        """
        Save training data to file.
        
        Args:
            training_examples: List of TrainingExample objects
            output_path: Path to save the data
            format_type: Format type ('jsonl', 'json', 'huggingface')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format_type == "huggingface":
            data = self.format_for_huggingface(training_examples)
        else:
            data = [example.to_dict() for example in training_examples]
        
        if format_type == "jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(training_examples)} training examples to {output_path}")
    
    def apply_data_augmentation(self, training_examples: List[TrainingExample], 
                              augmentation_factor: float = 2.0) -> List[TrainingExample]:
        """
        Apply data augmentation strategies to increase training diversity.
        
        Args:
            training_examples: Original training examples
            augmentation_factor: Multiplier for dataset size (e.g., 2.0 = double the data)
            
        Returns:
            Augmented list of training examples
        """
        augmented_examples = training_examples.copy()
        target_size = int(len(training_examples) * augmentation_factor)
        
        while len(augmented_examples) < target_size:
            for strategy in self.augmentation_strategies:
                if len(augmented_examples) >= target_size:
                    break
                
                # Select random example to augment
                import random
                base_example = random.choice(training_examples)
                
                if strategy == 'style_variation':
                    augmented = self._augment_style_variation(base_example)
                elif strategy == 'length_variation':
                    augmented = self._augment_length_variation(base_example)
                elif strategy == 'structural_variation':
                    augmented = self._augment_structural_variation(base_example)
                elif strategy == 'thematic_variation':
                    augmented = self._augment_thematic_variation(base_example)
                else:
                    continue
                
                if augmented:
                    augmented_examples.append(augmented)
        
        logger.info(f"Augmented dataset from {len(training_examples)} to {len(augmented_examples)} examples")
        return augmented_examples[:target_size]
    
    def _augment_style_variation(self, example: TrainingExample) -> Optional[TrainingExample]:
        """Create style variation by modifying instruction specificity."""
        variations = [
            f"Compose a poem in the distinctive style of {example.poet_name}.",
            f"Create a {example.poet_name}-inspired poem.",
            f"Write poetry that captures {example.poet_name}'s unique voice.",
            f"Generate a poem using {example.poet_name}'s characteristic style."
        ]
        
        import random
        new_instruction = random.choice(variations)
        
        return TrainingExample(
            instruction=new_instruction,
            input_text=example.input_text,
            output_text=example.output_text,
            stylometric_features=example.stylometric_features.copy(),
            poet_name=example.poet_name,
            poem_title=example.poem_title,
            line_number=example.line_number
        )
    
    def _augment_length_variation(self, example: TrainingExample) -> Optional[TrainingExample]:
        """Create length-based variations in instructions."""
        features = example.stylometric_features
        total_lines = features.get('total_lines', 0)
        
        if total_lines == 0:
            return None
        
        length_descriptors = {
            (1, 4): "short",
            (5, 8): "medium-length", 
            (9, 16): "long",
            (17, float('inf')): "extended"
        }
        
        descriptor = "standard"
        for (min_lines, max_lines), desc in length_descriptors.items():
            if min_lines <= total_lines <= max_lines:
                descriptor = desc
                break
        
        new_instruction = f"Write a {descriptor} poem in the style of {example.poet_name}."
        
        return TrainingExample(
            instruction=new_instruction,
            input_text=example.input_text,
            output_text=example.output_text,
            stylometric_features=features.copy(),
            poet_name=example.poet_name,
            poem_title=example.poem_title,
            line_number=example.line_number
        )
    
    def _augment_structural_variation(self, example: TrainingExample) -> Optional[TrainingExample]:
        """Create structural variation by emphasizing different poetic elements."""
        features = example.stylometric_features
        
        structural_elements = []
        if features.get('rhyme_scheme') and features['rhyme_scheme'] != 'free':
            structural_elements.append(f"with {features['rhyme_scheme']} rhyme scheme")
        
        if features.get('dominant_meter') and features['dominant_meter'] != 'free':
            structural_elements.append(f"in {features['dominant_meter']} meter")
        
        if features.get('total_stanzas', 0) > 1:
            structural_elements.append(f"with {features['total_stanzas']} stanzas")
        
        if not structural_elements:
            return None
        
        import random
        selected_element = random.choice(structural_elements)
        new_instruction = f"Write a poem in the style of {example.poet_name} {selected_element}."
        
        return TrainingExample(
            instruction=new_instruction,
            input_text=example.input_text,
            output_text=example.output_text,
            stylometric_features=features.copy(),
            poet_name=example.poet_name,
            poem_title=example.poem_title,
            line_number=example.line_number
        )
    
    def _augment_thematic_variation(self, example: TrainingExample) -> Optional[TrainingExample]:
        """Create thematic variations by adding context prompts."""
        themes = [
            "about nature",
            "exploring human emotions", 
            "reflecting on life",
            "with vivid imagery",
            "expressing deep feelings"
        ]
        
        import random
        theme = random.choice(themes)
        new_instruction = f"Write a poem in the style of {example.poet_name} {theme}."
        
        return TrainingExample(
            instruction=new_instruction,
            input_text=example.input_text,
            output_text=example.output_text,
            stylometric_features=example.stylometric_features.copy(),
            poet_name=example.poet_name,
            poem_title=example.poem_title,
            line_number=example.line_number
        )
    
    def create_balanced_dataset(self, training_examples: List[TrainingExample],
                              balance_by: str = 'poet_name') -> List[TrainingExample]:
        """
        Create a balanced dataset by ensuring equal representation.
        
        Args:
            training_examples: Original training examples
            balance_by: Field to balance by ('poet_name', 'poem_length', etc.)
            
        Returns:
            Balanced list of training examples
        """
        if balance_by == 'poet_name':
            return self._balance_by_poet(training_examples)
        elif balance_by == 'poem_length':
            return self._balance_by_length(training_examples)
        else:
            return training_examples
    
    def _balance_by_poet(self, examples: List[TrainingExample]) -> List[TrainingExample]:
        """Balance dataset by poet representation."""
        poet_groups = {}
        for example in examples:
            poet = example.poet_name
            if poet not in poet_groups:
                poet_groups[poet] = []
            poet_groups[poet].append(example)
        
        if not poet_groups:
            return examples
        
        # Find the minimum group size
        min_size = min(len(group) for group in poet_groups.values())
        
        # Sample equally from each group
        balanced_examples = []
        import random
        for poet, group in poet_groups.items():
            sampled = random.sample(group, min(min_size, len(group)))
            balanced_examples.extend(sampled)
        
        logger.info(f"Balanced dataset: {len(examples)} -> {len(balanced_examples)} examples")
        return balanced_examples
    
    def _balance_by_length(self, examples: List[TrainingExample]) -> List[TrainingExample]:
        """Balance dataset by poem length categories."""
        length_groups = {'short': [], 'medium': [], 'long': []}
        
        for example in examples:
            total_lines = example.stylometric_features.get('total_lines', 0)
            if total_lines <= 4:
                length_groups['short'].append(example)
            elif total_lines <= 12:
                length_groups['medium'].append(example)
            else:
                length_groups['long'].append(example)
        
        # Find minimum non-empty group size
        non_empty_groups = [group for group in length_groups.values() if group]
        if not non_empty_groups:
            return examples
        
        min_size = min(len(group) for group in non_empty_groups)
        
        # Sample equally from each non-empty group
        balanced_examples = []
        import random
        for category, group in length_groups.items():
            if group:
                sampled = random.sample(group, min(min_size, len(group)))
                balanced_examples.extend(sampled)
        
        return balanced_examples