"""
Comprehensive system integration module for the Stylistic Poetry LLM Framework.

This module provides the main system integration class that coordinates all
components and provides end-to-end functionality for poetry generation.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import sys

# Import all system components
from .text_processing import PoetryTextProcessor
from .lexical_analysis import LexicalAnalyzer
from .structural_analysis import StructuralAnalyzer
from .poet_profile import PoetProfile, PoetProfileManager
from .training_data import TrainingDataProcessor, PoetryCorpusLoader, TrainingDatasetFormatter
from .fine_tuning import FineTuningManager
from .model_interface import create_poetry_model, PoetryGenerationRequest, GenerationConfig
from .evaluation_metrics import QuantitativeEvaluator
from .evaluation_comparison import EvaluationComparator
from .dickinson_features import DickinsonFeatureExtractor, DickinsonStyleGenerator
from .output_formatter import PoetryOutputFormatter
from .performance_monitor import create_performance_monitor

# Import configuration and utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import config_manager, SystemConfig
from utils.logging import initialize_logging, get_logger
from utils.error_integration import initialize_error_handling
from utils.exceptions import PoetryLLMError, ConfigurationError, ValidationError


@dataclass
class SystemStatus:
    """System status tracking."""
    initialized: bool = False
    configuration_valid: bool = False
    components_loaded: Dict[str, bool] = field(default_factory=dict)
    dependencies_available: Dict[str, bool] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    last_validation: Optional[datetime] = None


class PoetryLLMSystem:
    """Main system integration class for the Stylistic Poetry LLM Framework."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the poetry LLM system.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config_path = config_path
        self.config: Optional[SystemConfig] = None
        self.logger: Optional[logging.Logger] = None
        self.status = SystemStatus()
        
        # Component instances
        self.text_processor: Optional[PoetryTextProcessor] = None
        self.lexical_analyzer: Optional[LexicalAnalyzer] = None
        self.structural_analyzer: Optional[StructuralAnalyzer] = None
        self.profile_manager: Optional[PoetProfileManager] = None
        self.training_processor: Optional[TrainingDataProcessor] = None
        self.corpus_loader: Optional[PoetryCorpusLoader] = None
        self.dataset_formatter: Optional[TrainingDatasetFormatter] = None
        self.fine_tuning_manager: Optional[FineTuningManager] = None
        self.quantitative_evaluator: Optional[QuantitativeEvaluator] = None
        self.evaluation_comparator: Optional[EvaluationComparator] = None
        self.dickinson_extractor: Optional[DickinsonFeatureExtractor] = None
        self.dickinson_generator: Optional[DickinsonStyleGenerator] = None
        self.output_formatter: Optional[PoetryOutputFormatter] = None
        self.performance_monitor = None
        
        # Active models tracking
        self.active_models: Dict[str, Any] = {}
    
    def initialize(self, log_level: str = "INFO", enable_error_recovery: bool = True,
                   validate_dependencies: bool = True) -> bool:
        """
        Initialize the complete system.
        
        Args:
            log_level: Logging level to use
            enable_error_recovery: Whether to enable error recovery
            validate_dependencies: Whether to validate dependencies
            
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Initialize logging and error handling
            initialize_logging(log_level)
            initialize_error_handling(log_errors=True, enable_recovery=enable_error_recovery)
            self.logger = get_logger('system_integration')
            
            self.logger.info("Starting Poetry LLM System initialization")
            
            # Load configuration
            if not self._load_configuration():
                return False
            
            # Validate dependencies if requested
            if validate_dependencies and not self._validate_dependencies():
                self.logger.warning("Some dependencies are missing, but continuing initialization")
            
            # Initialize all components
            if not self._initialize_components():
                return False
            
            # Validate system setup
            if not self._validate_system_setup():
                return False
            
            self.status.initialized = True
            self.status.last_validation = datetime.now()
            
            self.logger.info("Poetry LLM System initialization completed successfully")
            return True
            
        except Exception as e:
            error_msg = f"System initialization failed: {str(e)}"
            self.status.errors.append(error_msg)
            if self.logger:
                self.logger.error(error_msg)
            return False
    
    def _load_configuration(self) -> bool:
        """Load and validate configuration."""
        try:
            if self.config_path:
                self.config = config_manager.load_config(self.config_path)
            else:
                self.config = config_manager.get_config()
            
            self._validate_configuration()
            self.status.configuration_valid = True
            
            self.logger.info("Configuration loaded and validated successfully")
            return True
            
        except Exception as e:
            error_msg = f"Configuration loading failed: {str(e)}"
            self.status.errors.append(error_msg)
            self.logger.error(error_msg)
            return False
    
    def _validate_configuration(self) -> None:
        """Validate configuration parameters."""
        if not self.config:
            raise ConfigurationError("No configuration loaded")
        
        # Validate model configuration
        if self.config.model.temperature < 0.0 or self.config.model.temperature > 2.0:
            raise ConfigurationError("Temperature must be between 0.0 and 2.0")
        
        if self.config.model.top_p < 0.0 or self.config.model.top_p > 1.0:
            raise ConfigurationError("top_p must be between 0.0 and 1.0")
        
        if self.config.model.top_k < 1:
            raise ConfigurationError("top_k must be at least 1")
        
        # Validate training configuration
        if self.config.training.learning_rate <= 0:
            raise ConfigurationError("Learning rate must be positive")
        
        if self.config.training.batch_size < 1:
            raise ConfigurationError("Batch size must be at least 1")
        
        if self.config.training.num_epochs < 1:
            raise ConfigurationError("Number of epochs must be at least 1")
        
        # Create required directories
        directories = [
            self.config.data.data_dir,
            self.config.data.output_dir,
            self.config.evaluation.metrics_output_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _validate_dependencies(self) -> bool:
        """Validate that required dependencies are available."""
        required_packages = [
            'torch', 'transformers', 'nltk', 'pyphen', 'pronouncing',
            'numpy', 'pandas', 'scikit-learn', 'pydantic', 'pyyaml', 'click'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                self.status.dependencies_available[package] = True
            except ImportError:
                self.status.dependencies_available[package] = False
                missing_packages.append(package)
        
        if missing_packages:
            warning_msg = f"Missing packages: {', '.join(missing_packages)}"
            self.status.warnings.append(warning_msg)
            self.logger.warning(warning_msg)
            return False
        
        self.logger.info("All dependencies validated successfully")
        return True
    
    def _initialize_components(self) -> bool:
        """Initialize all system components."""
        try:
            # Text processing components
            self.text_processor = PoetryTextProcessor()
            self.status.components_loaded['text_processor'] = True
            
            self.lexical_analyzer = LexicalAnalyzer()
            self.status.components_loaded['lexical_analyzer'] = True
            
            self.structural_analyzer = StructuralAnalyzer()
            self.status.components_loaded['structural_analyzer'] = True
            
            # Profile management
            self.profile_manager = PoetProfileManager()
            self.status.components_loaded['profile_manager'] = True
            
            # Training data components
            self.training_processor = TrainingDataProcessor()
            self.status.components_loaded['training_processor'] = True
            
            self.corpus_loader = PoetryCorpusLoader()
            self.status.components_loaded['corpus_loader'] = True
            
            self.dataset_formatter = TrainingDatasetFormatter()
            self.status.components_loaded['dataset_formatter'] = True
            
            # Fine-tuning
            self.fine_tuning_manager = FineTuningManager()
            self.status.components_loaded['fine_tuning_manager'] = True
            
            # Evaluation components
            self.quantitative_evaluator = QuantitativeEvaluator()
            self.status.components_loaded['quantitative_evaluator'] = True
            
            self.evaluation_comparator = EvaluationComparator()
            self.status.components_loaded['evaluation_comparator'] = True
            
            # Dickinson-specific components
            self.dickinson_extractor = DickinsonFeatureExtractor()
            self.status.components_loaded['dickinson_extractor'] = True
            
            self.dickinson_generator = DickinsonStyleGenerator()
            self.status.components_loaded['dickinson_generator'] = True
            
            # Output formatting
            self.output_formatter = PoetryOutputFormatter()
            self.status.components_loaded['output_formatter'] = True
            
            # Performance monitoring
            self.performance_monitor = PerformanceMonitor()
            self.status.components_loaded['performance_monitor'] = True
            return True
            
        except Exception as e:
            error_msg = f"Failed to initialize components: {str(e)}"
            self.status.errors.append(error_msg)
            self.logger.error(error_msg)
            return False
    
    def _validate_system_setup(self) -> bool:
        """Validate component initialization and basic functionality."""
        try:
            # Verify all critical components are loaded
            critical_components = [
                'text_processor', 'lexical_analyzer', 'structural_analyzer',
                'quantitative_evaluator', 'output_formatter'
            ]
        
            for component in critical_components:
                if not self.status.components_loaded.get(component, False):
                    raise ValidationError(f"Critical component not loaded: {component}")
            
            # Test basic functionality
            test_text = "This is a test poem line\nWith multiple lines for testing"
            
            # Test text processing
            processed = self.text_processor.clean_text(test_text)
            if not processed or len(processed.strip()) == 0:
                raise ValidationError("Text processor validation failed")
            
            # Test lexical analysis
            lexical_metrics = self.lexical_analyzer.analyze_lexical_features(test_text)
            if not lexical_metrics:
                raise ValidationError("Lexical analyzer validation failed")
            
            
            # Test evaluation
            eval_metrics = self.quantitative_evaluator.evaluate_poetry(test_text)
            if not eval_metrics:
                raise ValidationError("Quantitative evaluator validation failed")
            
            return True
            
        except Exception as e:
            error_msg = f"System validation failed: {str(e)}"
            self.status.errors.append(error_msg)
            self.logger.error(error_msg)
            return False
    
    def generate_poetry_end_to_end(self, prompt: str, poet_style: Optional[str] = None, 
                                   model_name: str = "gpt2", **kwargs) -> Dict[str, Any]:
        """
        Generate poetry with complete analysis and processing.
        
        Args:
            prompt: Text prompt for generation
            poet_style: Optional poet style to emulate
            model_name: Model to use for generation
            **kwargs: Additional generation parameters
         
        Returns:
            Dictionary containing complete results
        """
        if not self.status.initialized:
            raise PoetryLLMError("System not initialized. Call initialize() first.")
        
        try:
            self.logger.info(f"Starting end-to-end poetry generation")
            
            # Create generation request
            gen_config = GenerationConfig(**kwargs)
            request = PoetryGenerationRequest(
                prompt=prompt,
                poet_style=poet_style,
                generation_config=gen_config
            )
            
            # Load model if not active
            model_key = f"{model_name}_{poet_style or 'general'}"
            
            if model_key not in self.active_models:
                model = create_poetry_model("gpt", model_name)
                if not model.load_model():
                    raise PoetryLLMError(f"Failed to load model: {model_name}")
                self.active_models[model_key] = model
            # Generate poetry
            response = model.generate_poetry(request)
            
            if not response.success:
                raise PoetryLLMError(f"Poetry generation failed: {response.error_message}")
            
            # Perform analysis
            generated_text = response.generated_text
            analysis_results = self.analyze_existing_poetry(generated_text)
            
            # Perform analysis
            generated_text = response.generated_text
            analysis_results = self.analyze_existing_poetry(generated_text)
            
            # Apply poet-specific analysis
            poet_specific_analysis = {}
            if poet_style == "emily_dickinson":
                poet_specific_analysis = self.dickinson_extractor.extract_features(generated_text)
            
            # Compile complete results
            results = {
                'generated_text': generated_text,
                'success': True,
                'poet_style': poet_style,
                'model_name': model_name,
                'analysis_results': analysis_results,
                'poet_specific_analysis': poet_specific_analysis,
                'formatted_output': self.output_formatter.create_comprehensive_output(
                    generated_text, analysis_results
                )
            }
            
            return results
            
        except Exception as e:
            error_msg = f"End-to-end generation failed: {str(e)}"
            self.logger.error(error_msg)
            return {
                'generated_text': '',
                'success': False,
                'error_message': error_msg,
                'poet_style': poet_style,
                'model_name': model_name
            }
        
        except Exception as e:
            error_msg = f"End-to-end generation failed: {str(e)}"
            self.logger.error(error_msg)
            return {
                'generated_text': '',
                'success': False,
                'error_message': error_msg,
                'poet_style': poet_style,
                'model_name': model_name
            }


    def analyze_existing_poetry(self, text: str, compare_with: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze existing poetry for stylistic features.
        
        Args:
            text: Poetry text to analyze
            compare_with: Optional poet style to compare against
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            self.logger.info("Starting poetry analysis")
            
            # Perform quantitative analysis
            analysis_results = self.quantitative_evaluator.evaluate_poetry(text)
            
            # Add comparison if requested
            if compare_with:
                comparison_results = self.evaluation_comparator.compare_poetry_side_by_side(
                    text, "", compare_with
                )
                analysis_results['comparison'] = comparison_results
            
            self.logger.info("Poetry analysis completed successfully")
            return analysis_results
            
        except Exception as e:
            error_msg = f"Poetry analysis failed: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error_message': error_msg,
                'text': text
            }
    
    def get_system_status(self) -> SystemStatus:
        """Get current system status."""
        return self.status
    
    def cleanup(self) -> None:
        """Clean up system resources."""
        try:
            # Unload active models
            for model in self.active_models.values():
                model.unload_model()
            self.active_models.clear()
            
            self.logger.info("System cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")


# Global system instance

# Global system instance for singleton pattern
_global_system = None

def get_global_system() -> PoetryLLMSystem:
    """
    Get the global Poetry LLM system instance.
    
    Returns:
        PoetryLLMSystem: Global system instance
        
    Raises:
        PoetryLLMError: If system not initialized
    """
    global _global_system
    if _global_system is None:
        raise PoetryLLMError("Global system not initialized")
    return _global_system


def initialize_global_system(**kwargs) -> PoetryLLMSystem:
    """
    Initialize the global Poetry LLM system.
    
    Args:
        **kwargs: Initialization parameters
        
    Returns:
        PoetryLLMSystem: Initialized global system instance
    """
    global _global_system
    _global_system = PoetryLLMSystem(**kwargs)
    _global_system.initialize()
    return _global_system