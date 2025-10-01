"""
Tests for evaluation comparison system.
"""

import pytest
import json
from unittest.mock import Mock, patch

from src.stylometric.evaluation_comparison import (
    EvaluationComparator,
    ComparisonResult,
    MetricStatistics
)


class TestEvaluationComparator:
    """Test cases for EvaluationComparator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.comparator = EvaluationComparator()
        self.sample_poem1 = "Hope is the thing with feathers\nThat perches in the soul"
        self.sample_poem2 = "I celebrate myself and sing myself\nAnd what I assume you shall assume"
    
    def test_compare_poetry_side_by_side(self):
        """Test side-by-side poetry comparison."""
        result = self.comparator.compare_poetry_side_by_side(
            self.sample_poem1, 
            self.sample_poem2
        )
        
        assert isinstance(result, ComparisonResult)
        assert hasattr(result, 'overall_comparison_score')
        assert hasattr(result, 'metric_differences')
        assert hasattr(result, 'similarity_scores')
    
    def test_analyze_metric_differences(self):
        """Test metric differences analysis."""
        generated_metrics = {'score': 0.8, 'complexity': 0.6}
        target_metrics = {'score': 0.9, 'complexity': 0.7}
        
        results = self.comparator.analyze_metric_differences(generated_metrics, target_metrics)
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, MetricStatistics) for r in results)
    
    def test_generate_comparison_report(self):
        """Test comparison report generation."""
        comparison = ComparisonResult(
            generated_text="Test poem 1",
            target_text="Test poem 2", 
            generated_metrics={'score': 0.8},
            target_metrics={'score': 0.9},
            metric_differences={'score': 0.1},
            statistical_analysis={'mean_diff': 0.1},
            similarity_scores={'overall': 0.75},
            overall_comparison_score=0.75
        )
        
        report = self.comparator.create_comprehensive_report(comparison)
        assert isinstance(report, dict)
        assert len(report) > 0


class TestComparisonResult:
    """Test cases for ComparisonResult."""
    
    def test_comparison_result_creation(self):
        """Test ComparisonResult creation."""
        result = ComparisonResult(
            generated_text="Test poem 1",
            target_text="Test poem 2",
            generated_metrics={'score': 0.8},
            target_metrics={'score': 0.9},
            metric_differences={'score': 0.1},
            statistical_analysis={'mean_diff': 0.1},
            similarity_scores={'overall': 0.85},
            overall_comparison_score=0.85
        )
        
        assert result.overall_comparison_score == 0.85
        assert result.metric_differences == {'score': 0.1}
        assert result.generated_text == "Test poem 1"
        assert result.target_text == "Test poem 2"


class TestMetricStatistics:
    """Test cases for MetricStatistics."""
    
    def test_metric_statistics_creation(self):
        """Test MetricStatistics creation."""
        stats = MetricStatistics(
            mean=0.75,
            std=0.1,
            min_val=0.5,
            max_val=0.9,
            median=0.8
        )
        
        assert stats.mean == 0.75
        assert stats.std == 0.1
        assert stats.min_val == 0.5
        assert stats.max_val == 0.9
        assert stats.median == 0.8