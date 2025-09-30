"""
Stylometric analysis package for poetry.

This package provides tools for analyzing the stylistic features of poetry,
including text processing, lexical analysis, structural analysis, and poet profiling.
"""

from .text_processing import (
    PoetryTextProcessor,
    clean_poetry_text,
    segment_poem_lines,
    segment_poem_stanzas,
    count_syllables
)

from .lexical_analysis import (
    LexicalAnalyzer,
    calculate_type_token_ratio,
    get_pos_tags,
    analyze_vocabulary_richness
)

from .structural_analysis import (
    StructuralAnalyzer,
    analyze_poem_meter,
    analyze_poem_rhyme_scheme
)

from .model_interface import (
    PoetryGenerationModel,
    GPTPoetryModel,
    PoetryGenerationRequest,
    PoetryGenerationResponse,
    GenerationConfig,
    create_poetry_model
)

from .evaluation_metrics import (
    QuantitativeEvaluator,
    EvaluationResult,
    evaluate_generated_poetry,
    calculate_ttr_and_density,
    measure_structural_adherence,
    calculate_readability_scores
)

from .dickinson_features import (
    DickinsonFeatureDetector,
    DickinsonStyleProfile,
    analyze_dickinson_features,
    score_dickinson_similarity,
    create_dickinson_profile
)

from .performance_monitor import (
    PerformanceMonitor,
    PerformanceMetrics,
    BenchmarkResult,
    PerformanceBenchmark,
    MemoryMonitor,
    create_performance_monitor,
    create_benchmark_suite
)

from .performance_regression import (
    RegressionThresholds,
    RegressionAlert,
    PerformanceBaseline,
    RegressionDetector,
    create_regression_tester
)

__version__ = "0.1.0"

__all__ = [
    "PoetryTextProcessor",
    "clean_poetry_text", 
    "segment_poem_lines",
    "segment_poem_stanzas",
    "count_syllables",
    "LexicalAnalyzer",
    "calculate_type_token_ratio",
    "get_pos_tags",
    "analyze_vocabulary_richness",
    "StructuralAnalyzer",
    "analyze_poem_meter",
    "analyze_poem_rhyme_scheme",
    "PoetryGenerationModel",
    "GPTPoetryModel",
    "PoetryGenerationRequest",
    "PoetryGenerationResponse",
    "GenerationConfig",
    "create_poetry_model",
    "QuantitativeEvaluator",
    "EvaluationResult",
    "evaluate_generated_poetry",
    "calculate_ttr_and_density",
    "measure_structural_adherence",
    "calculate_readability_scores",
    "DickinsonFeatureDetector",
    "DickinsonStyleProfile",
    "analyze_dickinson_features",
    "score_dickinson_similarity",
    "create_dickinson_profile",
    "PerformanceMonitor",
    "PerformanceMetrics",
    "BenchmarkResult",
    "PerformanceBenchmark",
    "MemoryMonitor",
    "create_performance_monitor",
    "create_benchmark_suite",
    "RegressionThresholds",
    "RegressionAlert",
    "PerformanceBaseline",
    "RegressionDetector",
    "create_regression_tester"
]