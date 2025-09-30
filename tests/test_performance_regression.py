"""
Tests for performance regression testing functionality.
"""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime, timedelta

from src.stylometric.performance_regression import (
    RegressionThresholds,
    RegressionAlert,
    PerformanceBaseline,
    RegressionDetector,
    create_regression_tester
)
from src.stylometric.performance_monitor import (
    PerformanceMetrics,
    BenchmarkResult,
    PerformanceMonitor,
    PerformanceBenchmark
)


class TestRegressionThresholds:
    """Test cases for RegressionThresholds dataclass."""
    
    def test_default_thresholds(self):
        """Test default threshold values."""
        thresholds = RegressionThresholds()
        
        assert thresholds.latency_increase_percent == 20.0
        assert thresholds.memory_increase_percent == 15.0
        assert thresholds.success_rate_decrease_percent == 5.0
        assert thresholds.min_samples == 3


class TestPerformanceBaseline:
    """Test cases for PerformanceBaseline class."""
    
    def test_baseline_initialization_no_file(self):
        """Test baseline initialization when no file exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_file = Path(temp_dir) / "nonexistent_baseline.json"
            baseline = PerformanceBaseline(baseline_file)
            
            assert baseline.baseline_file == baseline_file
            assert baseline.baselines == {}
    
    def test_update_and_save_baseline(self):
        """Test updating and saving baselines."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            baseline = PerformanceBaseline(temp_path)
            
            # Update baseline
            metrics = {
                "average_latency_ms": 150.0,
                "average_memory_mb": 250.0,
                "success_rate": 0.95
            }
            baseline.update_baseline("test_config", metrics)
            
            # Check that baseline was updated
            assert "test_config" in baseline.baselines
            saved_metrics = baseline.baselines["test_config"]
            assert saved_metrics["average_latency_ms"] == 150.0
            assert saved_metrics["average_memory_mb"] == 250.0
            assert saved_metrics["success_rate"] == 0.95
            assert "updated_at" in saved_metrics
            
        finally:
            if temp_path.exists():
                temp_path.unlink()


class TestRegressionDetector:
    """Test cases for RegressionDetector class."""
    
    def test_detector_initialization(self):
        """Test regression detector initialization."""
        baseline = PerformanceBaseline()
        detector = RegressionDetector(baseline)
        
        assert detector.baseline_manager is baseline
        assert isinstance(detector.thresholds, RegressionThresholds)
        assert detector.alerts == []
    
    def test_latency_regression_detected(self):
        """Test detection of latency regression."""
        baseline = PerformanceBaseline()
        thresholds = RegressionThresholds(latency_increase_percent=10.0)
        detector = RegressionDetector(baseline, thresholds)
        
        # Set up baseline
        baseline.update_baseline("test_config", {
            "average_latency_ms": 100.0,
            "average_memory_mb": 200.0,
            "success_rate": 1.0
        })
        
        # Create test result with significant latency increase (25% > 10% threshold)
        metrics = [
            PerformanceMetrics(
                latency_ms=125.0,
                memory_usage_mb=200.0,
                peak_memory_mb=200.0,
                cpu_usage_percent=50.0,
                operation="test",
                model_name="test"
            ) for _ in range(3)
        ]
        
        result = BenchmarkResult(
            configuration_name="test_config",
            metrics=metrics,
            average_latency_ms=125.0,
            median_latency_ms=125.0,
            p95_latency_ms=125.0,
            average_memory_mb=200.0,
            peak_memory_mb=200.0,
            success_rate=1.0,
            total_runs=3
        )
        
        alerts = detector.check_for_regressions([result])
        assert len(alerts) == 1
        
        alert = alerts[0]
        assert alert.metric_name == "average_latency_ms"
        assert alert.current_value == 125.0
        assert alert.baseline_value == 100.0
        assert alert.change_percent == 25.0
        assert alert.configuration_name == "test_config"


def test_create_regression_tester():
    """Test creating regression tester."""
    tester = create_regression_tester()
    
    assert hasattr(tester, 'baseline_manager')
    assert hasattr(tester, 'detector')
    assert hasattr(tester, 'run_regression_test')