"""
Tests for performance monitoring functionality.
"""

import pytest
import time
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

from src.stylometric.performance_monitor import (
    PerformanceMonitor,
    PerformanceMetrics,
    BenchmarkResult,
    PerformanceBenchmark,
    MemoryMonitor,
    create_performance_monitor,
    create_benchmark_suite
)


class TestMemoryMonitor:
    """Test cases for MemoryMonitor class."""
    
    def test_memory_monitor_initialization(self):
        """Test memory monitor initialization."""
        monitor = MemoryMonitor(interval_seconds=0.05)
        
        assert monitor.interval == 0.05
        assert not monitor.monitoring
        assert monitor.peak_memory == 0.0
        assert monitor.current_memory == 0.0
        assert monitor.gpu_peak_memory == 0.0
        assert monitor.monitor_thread is None
    
    def test_memory_monitor_start_stop(self):
        """Test starting and stopping memory monitoring."""
        monitor = MemoryMonitor(interval_seconds=0.01)
        
        # Start monitoring
        monitor.start_monitoring()
        assert monitor.monitoring
        assert monitor.monitor_thread is not None
        
        # Let it run briefly
        time.sleep(0.05)
        
        # Stop monitoring
        stats = monitor.stop_monitoring()
        assert not monitor.monitoring
        assert "peak_memory_mb" in stats
        assert "current_memory_mb" in stats
        assert "gpu_peak_memory_mb" in stats
        assert stats["peak_memory_mb"] > 0  # Should have recorded some memory usage


class TestPerformanceMonitor:
    """Test cases for PerformanceMonitor class."""
    
    def test_performance_monitor_initialization(self):
        """Test performance monitor initialization."""
        monitor = PerformanceMonitor(enable_gpu_monitoring=False)
        
        assert not monitor.enable_gpu_monitoring
        assert isinstance(monitor.memory_monitor, MemoryMonitor)
        assert monitor.metrics_history == []
    
    def test_monitor_operation_context_manager(self):
        """Test the monitor_operation context manager."""
        monitor = PerformanceMonitor(enable_gpu_monitoring=False)
        
        def test_operation():
            time.sleep(0.01)  # Simulate some work
        
        with monitor.monitor_operation("test_op", "test_model", {"param": "value"}) as metrics:
            test_operation()
        
        # Check that metrics were populated
        assert metrics.latency_ms > 0
        assert metrics.memory_usage_mb > 0
        assert metrics.peak_memory_mb > 0
        assert metrics.operation == "test_op"
        assert metrics.model_name == "test_model"
        assert metrics.configuration == {"param": "value"}
        
        # Check that metrics were stored in history
        assert len(monitor.metrics_history) == 1
        assert monitor.metrics_history[0] is metrics
    
    def test_get_recent_metrics(self):
        """Test getting recent metrics."""
        monitor = PerformanceMonitor(enable_gpu_monitoring=False)
        
        # Add some test metrics
        for i in range(5):
            with monitor.monitor_operation(f"test_op_{i}"):
                time.sleep(0.001)
        
        # Test getting recent metrics
        recent = monitor.get_recent_metrics(3)
        assert len(recent) == 3
        assert recent[0].operation == "test_op_2"
        assert recent[2].operation == "test_op_4"
    
    def test_calculate_operation_stats(self):
        """Test calculating operation statistics."""
        monitor = PerformanceMonitor(enable_gpu_monitoring=False)
        
        # Add multiple metrics for the same operation
        for i in range(5):
            with monitor.monitor_operation("test_op"):
                time.sleep(0.001 * (i + 1))  # Variable sleep times
        
        stats = monitor.calculate_operation_stats("test_op")
        
        assert stats is not None
        assert stats["count"] == 5
        assert "avg_latency_ms" in stats
        assert "median_latency_ms" in stats
        assert "p95_latency_ms" in stats
        assert "min_latency_ms" in stats
        assert "max_latency_ms" in stats
        assert "avg_memory_mb" in stats
        assert "peak_memory_mb" in stats
        
        # Test with non-existent operation
        assert monitor.calculate_operation_stats("nonexistent") is None


class TestPerformanceBenchmark:
    """Test cases for PerformanceBenchmark class."""
    
    def test_benchmark_initialization(self):
        """Test benchmark initialization."""
        monitor = PerformanceMonitor(enable_gpu_monitoring=False)
        benchmark = PerformanceBenchmark(monitor)
        
        assert benchmark.monitor is monitor
        assert benchmark.benchmark_results == []
    
    def test_run_benchmark(self):
        """Test running a benchmark."""
        monitor = PerformanceMonitor(enable_gpu_monitoring=False)
        benchmark = PerformanceBenchmark(monitor)
        
        def test_function(config):
            time.sleep(config.get("sleep_time", 0.001))
        
        configurations = {
            "fast_config": {"sleep_time": 0.001},
            "slow_config": {"sleep_time": 0.005}
        }
        
        results = benchmark.run_benchmark(
            test_function,
            configurations,
            runs_per_config=3,
            warmup_runs=1
        )
        
        assert len(results) == 2
        
        # Check fast config results
        fast_result = next(r for r in results if r.configuration_name == "fast_config")
        assert fast_result.total_runs == 3
        assert fast_result.success_rate == 1.0
        assert len(fast_result.metrics) == 3
        
        # Check slow config results
        slow_result = next(r for r in results if r.configuration_name == "slow_config")
        assert slow_result.average_latency_ms > fast_result.average_latency_ms


def test_create_performance_monitor():
    """Test create_performance_monitor factory function."""
    monitor = create_performance_monitor(enable_gpu_monitoring=False)
    
    assert isinstance(monitor, PerformanceMonitor)
    assert not monitor.enable_gpu_monitoring


def test_create_benchmark_suite():
    """Test create_benchmark_suite factory function."""
    monitor = create_performance_monitor(enable_gpu_monitoring=False)
    benchmark = create_benchmark_suite(monitor)
    
    assert isinstance(benchmark, PerformanceBenchmark)
    assert benchmark.monitor is monitor