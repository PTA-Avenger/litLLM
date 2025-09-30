"""
Performance monitoring utilities for poetry generation models.

This module provides comprehensive performance monitoring including latency measurement,
memory usage tracking, benchmarking capabilities, and performance regression testing.
"""

import time
import psutil
import threading
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
from pathlib import Path
import json
import statistics
import gc
import torch
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    latency_ms: float
    memory_usage_mb: float
    peak_memory_mb: float
    cpu_usage_percent: float
    gpu_memory_mb: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    operation: str = "unknown"
    model_name: str = "unknown"
    configuration: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    configuration_name: str
    metrics: List[PerformanceMetrics]
    average_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    average_memory_mb: float
    peak_memory_mb: float
    success_rate: float
    total_runs: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

class MemoryMonitor:
    """Thread-safe memory usage monitor."""
    
    def __init__(self, interval_seconds: float = 0.1):
        """
        Initialize memory monitor.
        
        Args:
            interval_seconds: Monitoring interval in seconds
        """
        self.interval = interval_seconds
        self.monitoring = False
        self.peak_memory = 0.0
        self.current_memory = 0.0
        self.gpu_peak_memory = 0.0
        self.monitor_thread = None
        self._lock = threading.Lock()
    
    def start_monitoring(self):
        """Start memory monitoring in a separate thread."""
        with self._lock:
            if self.monitoring:
                return
            
            self.monitoring = True
            self.peak_memory = 0.0
            self.gpu_peak_memory = 0.0
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, float]:
        """
        Stop memory monitoring and return peak usage.
        
        Returns:
            Dict containing peak memory usage statistics
        """
        with self._lock:
            if not self.monitoring:
                return {"peak_memory_mb": 0.0, "gpu_peak_memory_mb": 0.0}
            
            self.monitoring = False
            
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=1.0)
            
            return {
                "peak_memory_mb": self.peak_memory,
                "current_memory_mb": self.current_memory,
                "gpu_peak_memory_mb": self.gpu_peak_memory
            }
    
    def _monitor_loop(self):
        """Memory monitoring loop."""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                # Monitor system memory
                memory_info = process.memory_info()
                current_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
                
                with self._lock:
                    self.current_memory = current_mb
                    self.peak_memory = max(self.peak_memory, current_mb)
                
                # Monitor GPU memory if available
                if torch.cuda.is_available():
                    try:
                        gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # Convert to MB
                        with self._lock:
                            self.gpu_peak_memory = max(self.gpu_peak_memory, gpu_memory)
                    except Exception:
                        pass  # GPU monitoring is optional
                
                time.sleep(self.interval)
                
            except Exception as e:
                logger.warning(f"Memory monitoring error: {e}")
                time.sleep(self.interval)


class PerformanceMonitor:
    """Main performance monitoring class."""
    
    def __init__(self, enable_gpu_monitoring: bool = True):
        """
        Initialize performance monitor.
        
        Args:
            enable_gpu_monitoring: Whether to monitor GPU metrics
        """
        self.enable_gpu_monitoring = enable_gpu_monitoring and torch.cuda.is_available()
        self.memory_monitor = MemoryMonitor()
        self.metrics_history: List[PerformanceMetrics] = []
        
    @contextmanager
    def monitor_operation(self, operation_name: str, model_name: str = "unknown", 
                         configuration: Optional[Dict[str, Any]] = None):
        """
        Context manager for monitoring an operation's performance.
        
        Args:
            operation_name: Name of the operation being monitored
            model_name: Name of the model being used
            configuration: Configuration parameters for the operation
            
        Yields:
            PerformanceMetrics: Metrics object that will be populated
        """
        # Initialize metrics
        metrics = PerformanceMetrics(
            latency_ms=0.0,
            memory_usage_mb=0.0,
            peak_memory_mb=0.0,
            cpu_usage_percent=0.0,
            operation=operation_name,
            model_name=model_name,
            configuration=configuration or {}
        )
        
        # Get initial state
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)
        initial_cpu_times = process.cpu_times()
        
        # Start monitoring
        self.memory_monitor.start_monitoring()
        start_time = time.perf_counter()
        
        # Monitor GPU if available
        initial_gpu_memory = 0.0
        if self.enable_gpu_monitoring:
            try:
                initial_gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
            except Exception:
                pass
        
        try:
            yield metrics
            
        finally:
            # Calculate elapsed time
            end_time = time.perf_counter()
            metrics.latency_ms = (end_time - start_time) * 1000
            
            # Stop memory monitoring and get peak usage
            memory_stats = self.memory_monitor.stop_monitoring()
            metrics.peak_memory_mb = memory_stats["peak_memory_mb"]
            
            # Get final memory usage
            final_memory = process.memory_info().rss / (1024 * 1024)
            metrics.memory_usage_mb = final_memory
            
            # Calculate CPU usage
            final_cpu_times = process.cpu_times()
            cpu_time_used = (final_cpu_times.user - initial_cpu_times.user + 
                           final_cpu_times.system - initial_cpu_times.system)
            wall_time = end_time - start_time
            metrics.cpu_usage_percent = (cpu_time_used / wall_time) * 100 if wall_time > 0 else 0
            
            # Get GPU metrics if available
            if self.enable_gpu_monitoring:
                try:
                    final_gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
                    metrics.gpu_memory_mb = final_gpu_memory
                    
                    # Get GPU utilization (simplified)
                    if hasattr(torch.cuda, 'utilization'):
                        metrics.gpu_utilization_percent = torch.cuda.utilization()
                except Exception:
                    pass
            
            # Store metrics
            self.metrics_history.append(metrics)
            
            # Log performance summary
            logger.info(f"Performance - {operation_name}: "
                       f"latency={metrics.latency_ms:.2f}ms, "
                       f"memory={metrics.memory_usage_mb:.1f}MB, "
                       f"peak_memory={metrics.peak_memory_mb:.1f}MB")    

    def get_recent_metrics(self, count: int = 10) -> List[PerformanceMetrics]:
        """
        Get recent performance metrics.
        
        Args:
            count: Number of recent metrics to return
            
        Returns:
            List of recent PerformanceMetrics
        """
        return self.metrics_history[-count:] if self.metrics_history else []
    
    def get_metrics_by_operation(self, operation_name: str) -> List[PerformanceMetrics]:
        """
        Get metrics filtered by operation name.
        
        Args:
            operation_name: Name of the operation to filter by
            
        Returns:
            List of PerformanceMetrics for the specified operation
        """
        return [m for m in self.metrics_history if m.operation == operation_name]
    
    def calculate_operation_stats(self, operation_name: str) -> Optional[Dict[str, float]]:
        """
        Calculate statistics for a specific operation.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Dictionary with statistical metrics or None if no data
        """
        metrics = self.get_metrics_by_operation(operation_name)
        if not metrics:
            return None
        
        latencies = [m.latency_ms for m in metrics]
        memory_usage = [m.memory_usage_mb for m in metrics]
        
        return {
            "count": len(metrics),
            "avg_latency_ms": statistics.mean(latencies),
            "median_latency_ms": statistics.median(latencies),
            "p95_latency_ms": statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "avg_memory_mb": statistics.mean(memory_usage),
            "peak_memory_mb": max(m.peak_memory_mb for m in metrics)
        }
    
    def clear_history(self):
        """Clear metrics history."""
        self.metrics_history.clear()
        logger.info("Performance metrics history cleared")


class PerformanceBenchmark:
    """Performance benchmarking utilities."""
    
    def __init__(self, monitor: PerformanceMonitor):
        """
        Initialize benchmark with a performance monitor.
        
        Args:
            monitor: PerformanceMonitor instance
        """
        self.monitor = monitor
        self.benchmark_results: List[BenchmarkResult] = []
    
    def run_benchmark(self, 
                     benchmark_function: Callable,
                     configurations: Dict[str, Dict[str, Any]],
                     runs_per_config: int = 5,
                     warmup_runs: int = 1) -> List[BenchmarkResult]:
        """
        Run performance benchmark with multiple configurations.
        
        Args:
            benchmark_function: Function to benchmark (should accept config dict)
            configurations: Dict mapping config names to config parameters
            runs_per_config: Number of runs per configuration
            warmup_runs: Number of warmup runs (not counted in results)
            
        Returns:
            List of BenchmarkResult objects
        """
        results = []
        
        for config_name, config_params in configurations.items():
            logger.info(f"Running benchmark for configuration: {config_name}")
            
            # Warmup runs
            for i in range(warmup_runs):
                try:
                    with self.monitor.monitor_operation(f"warmup_{config_name}", 
                                                      configuration=config_params):
                        benchmark_function(config_params)
                except Exception as e:
                    logger.warning(f"Warmup run {i+1} failed for {config_name}: {e}")
            
            # Actual benchmark runs
            run_metrics = []
            successful_runs = 0
            
            for run in range(runs_per_config):
                try:
                    with self.monitor.monitor_operation(f"benchmark_{config_name}",
                                                      configuration=config_params) as metrics:
                        benchmark_function(config_params)
                        run_metrics.append(metrics)
                        successful_runs += 1
                        
                except Exception as e:
                    logger.error(f"Benchmark run {run+1} failed for {config_name}: {e}")
                
                # Force garbage collection between runs
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Calculate benchmark statistics
            if run_metrics:
                latencies = [m.latency_ms for m in run_metrics]
                memory_usage = [m.memory_usage_mb for m in run_metrics]
                
                result = BenchmarkResult(
                    configuration_name=config_name,
                    metrics=run_metrics,
                    average_latency_ms=statistics.mean(latencies),
                    median_latency_ms=statistics.median(latencies),
                    p95_latency_ms=statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies),
                    average_memory_mb=statistics.mean(memory_usage),
                    peak_memory_mb=max(m.peak_memory_mb for m in run_metrics),
                    success_rate=successful_runs / runs_per_config,
                    total_runs=runs_per_config
                )
                
                results.append(result)
                self.benchmark_results.append(result)
                
                logger.info(f"Benchmark completed for {config_name}: "
                           f"avg_latency={result.average_latency_ms:.2f}ms, "
                           f"success_rate={result.success_rate:.2%}")
        
        return results  
  
    def save_benchmark_results(self, filepath: Path):
        """
        Save benchmark results to JSON file.
        
        Args:
            filepath: Path to save results
        """
        # Convert results to serializable format
        serializable_results = []
        for result in self.benchmark_results:
            result_dict = {
                "configuration_name": result.configuration_name,
                "average_latency_ms": result.average_latency_ms,
                "median_latency_ms": result.median_latency_ms,
                "p95_latency_ms": result.p95_latency_ms,
                "average_memory_mb": result.average_memory_mb,
                "peak_memory_mb": result.peak_memory_mb,
                "success_rate": result.success_rate,
                "total_runs": result.total_runs,
                "timestamp": result.timestamp,
                "metrics": [
                    {
                        "latency_ms": m.latency_ms,
                        "memory_usage_mb": m.memory_usage_mb,
                        "peak_memory_mb": m.peak_memory_mb,
                        "cpu_usage_percent": m.cpu_usage_percent,
                        "gpu_memory_mb": m.gpu_memory_mb,
                        "gpu_utilization_percent": m.gpu_utilization_percent,
                        "timestamp": m.timestamp,
                        "operation": m.operation,
                        "model_name": m.model_name,
                        "configuration": m.configuration
                    }
                    for m in result.metrics
                ]
            }
            serializable_results.append(result_dict)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Benchmark results saved to {filepath}")
    
    def load_benchmark_results(self, filepath: Path) -> List[BenchmarkResult]:
        """
        Load benchmark results from JSON file.
        
        Args:
            filepath: Path to load results from
            
        Returns:
            List of BenchmarkResult objects
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        results = []
        for result_data in data:
            # Reconstruct metrics
            metrics = []
            for metric_data in result_data["metrics"]:
                metric = PerformanceMetrics(
                    latency_ms=metric_data["latency_ms"],
                    memory_usage_mb=metric_data["memory_usage_mb"],
                    peak_memory_mb=metric_data["peak_memory_mb"],
                    cpu_usage_percent=metric_data["cpu_usage_percent"],
                    gpu_memory_mb=metric_data.get("gpu_memory_mb"),
                    gpu_utilization_percent=metric_data.get("gpu_utilization_percent"),
                    timestamp=metric_data["timestamp"],
                    operation=metric_data["operation"],
                    model_name=metric_data["model_name"],
                    configuration=metric_data["configuration"]
                )
                metrics.append(metric)
            
            # Reconstruct result
            result = BenchmarkResult(
                configuration_name=result_data["configuration_name"],
                metrics=metrics,
                average_latency_ms=result_data["average_latency_ms"],
                median_latency_ms=result_data["median_latency_ms"],
                p95_latency_ms=result_data["p95_latency_ms"],
                average_memory_mb=result_data["average_memory_mb"],
                peak_memory_mb=result_data["peak_memory_mb"],
                success_rate=result_data["success_rate"],
                total_runs=result_data["total_runs"],
                timestamp=result_data["timestamp"]
            )
            results.append(result)
        
        self.benchmark_results.extend(results)
        logger.info(f"Loaded {len(results)} benchmark results from {filepath}")
        return results


def create_performance_monitor(enable_gpu_monitoring: bool = True) -> PerformanceMonitor:
    """
    Factory function to create a performance monitor.
    
    Args:
        enable_gpu_monitoring: Whether to enable GPU monitoring
        
    Returns:
        PerformanceMonitor instance
    """
    return PerformanceMonitor(enable_gpu_monitoring=enable_gpu_monitoring)


def create_benchmark_suite(monitor: PerformanceMonitor) -> PerformanceBenchmark:
    """
    Factory function to create a benchmark suite.
    
    Args:
        monitor: PerformanceMonitor instance
        
    Returns:
        PerformanceBenchmark instance
    """
    return PerformanceBenchmark(monitor)