"""
Performance regression testing utilities.

This module provides tools for detecting performance regressions by comparing
current performance metrics against historical baselines.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import statistics
from datetime import datetime, timedelta

from .performance_monitor import PerformanceMetrics, BenchmarkResult, PerformanceBenchmark

logger = logging.getLogger(__name__)


@dataclass
class RegressionThresholds:
    """Thresholds for detecting performance regressions."""
    latency_increase_percent: float = 20.0  # Alert if latency increases by more than 20%
    memory_increase_percent: float = 15.0   # Alert if memory usage increases by more than 15%
    success_rate_decrease_percent: float = 5.0  # Alert if success rate drops by more than 5%
    min_samples: int = 3  # Minimum number of samples needed for comparison


@dataclass
class RegressionAlert:
    """Container for regression alert information."""
    metric_name: str
    current_value: float
    baseline_value: float
    change_percent: float
    threshold_percent: float
    severity: str  # "warning", "critical"
    timestamp: str
    configuration_name: str
    details: Dict[str, Any]

class PerformanceBaseline:
    """Manages performance baselines for regression testing."""
    
    def __init__(self, baseline_file: Optional[Path] = None):
        """
        Initialize performance baseline manager.
        
        Args:
            baseline_file: Path to baseline file (optional)
        """
        self.baseline_file = baseline_file or Path("performance_baseline.json")
        self.baselines: Dict[str, Dict[str, float]] = {}
        self.load_baselines()
    
    def load_baselines(self):
        """Load baselines from file if it exists."""
        if self.baseline_file.exists():
            try:
                with open(self.baseline_file, 'r') as f:
                    self.baselines = json.load(f)
                logger.info(f"Loaded performance baselines from {self.baseline_file}")
            except Exception as e:
                logger.error(f"Failed to load baselines: {e}")
                self.baselines = {}
        else:
            logger.info("No baseline file found, starting with empty baselines")
    
    def save_baselines(self):
        """Save current baselines to file."""
        try:
            with open(self.baseline_file, 'w') as f:
                json.dump(self.baselines, f, indent=2)
            logger.info(f"Saved performance baselines to {self.baseline_file}")
        except Exception as e:
            logger.error(f"Failed to save baselines: {e}")
    
    def update_baseline(self, configuration_name: str, metrics: Dict[str, float]):
        """
        Update baseline for a configuration.
        
        Args:
            configuration_name: Name of the configuration
            metrics: Dictionary of metric name to value
        """
        self.baselines[configuration_name] = {
            **metrics,
            "updated_at": datetime.now().isoformat()
        }
        self.save_baselines()
        logger.info(f"Updated baseline for configuration: {configuration_name}")
    
    def get_baseline(self, configuration_name: str) -> Optional[Dict[str, float]]:
        """
        Get baseline for a configuration.
        
        Args:
            configuration_name: Name of the configuration
            
        Returns:
            Dictionary of baseline metrics or None if not found
        """
        return self.baselines.get(configuration_name)
    
    def has_baseline(self, configuration_name: str) -> bool:
        """
        Check if baseline exists for a configuration.
        
        Args:
            configuration_name: Name of the configuration
            
        Returns:
            True if baseline exists, False otherwise
        """
        return configuration_name in self.baselines

class RegressionDetector:
    """Detects performance regressions by comparing against baselines."""
    
    def __init__(self, baseline_manager: PerformanceBaseline, 
                 thresholds: Optional[RegressionThresholds] = None):
        """
        Initialize regression detector.
        
        Args:
            baseline_manager: PerformanceBaseline instance
            thresholds: RegressionThresholds for detection sensitivity
        """
        self.baseline_manager = baseline_manager
        self.thresholds = thresholds or RegressionThresholds()
        self.alerts: List[RegressionAlert] = []
    
    def check_for_regressions(self, benchmark_results: List[BenchmarkResult]) -> List[RegressionAlert]:
        """
        Check benchmark results for performance regressions.
        
        Args:
            benchmark_results: List of BenchmarkResult objects to check
            
        Returns:
            List of RegressionAlert objects for detected regressions
        """
        alerts = []
        
        for result in benchmark_results:
            config_name = result.configuration_name
            baseline = self.baseline_manager.get_baseline(config_name)
            
            if not baseline:
                logger.info(f"No baseline found for {config_name}, skipping regression check")
                continue
            
            # Check if we have enough samples
            if len(result.metrics) < self.thresholds.min_samples:
                logger.warning(f"Insufficient samples for {config_name} regression check: "
                             f"{len(result.metrics)} < {self.thresholds.min_samples}")
                continue
            
            # Check latency regression
            latency_alert = self._check_metric_regression(
                "average_latency_ms",
                result.average_latency_ms,
                baseline.get("average_latency_ms", 0),
                self.thresholds.latency_increase_percent,
                config_name,
                {"current_median": result.median_latency_ms, "current_p95": result.p95_latency_ms}
            )
            if latency_alert:
                alerts.append(latency_alert)
            
            # Check memory regression
            memory_alert = self._check_metric_regression(
                "average_memory_mb",
                result.average_memory_mb,
                baseline.get("average_memory_mb", 0),
                self.thresholds.memory_increase_percent,
                config_name,
                {"current_peak": result.peak_memory_mb}
            )
            if memory_alert:
                alerts.append(memory_alert)
            
            # Check success rate regression (decrease is bad)
            success_rate_alert = self._check_metric_regression(
                "success_rate",
                result.success_rate,
                baseline.get("success_rate", 1.0),
                -self.thresholds.success_rate_decrease_percent,  # Negative because decrease is bad
                config_name,
                {"total_runs": result.total_runs}
            )
            if success_rate_alert:
                alerts.append(success_rate_alert)
        
        self.alerts.extend(alerts)
        return alerts    

    def _check_metric_regression(self, metric_name: str, current_value: float, 
                               baseline_value: float, threshold_percent: float,
                               config_name: str, details: Dict[str, Any]) -> Optional[RegressionAlert]:
        """
        Check if a specific metric shows regression.
        
        Args:
            metric_name: Name of the metric
            current_value: Current metric value
            baseline_value: Baseline metric value
            threshold_percent: Threshold for regression (positive for increase, negative for decrease)
            config_name: Configuration name
            details: Additional details for the alert
            
        Returns:
            RegressionAlert if regression detected, None otherwise
        """
        if baseline_value == 0:
            return None  # Can't calculate percentage change
        
        change_percent = ((current_value - baseline_value) / baseline_value) * 100
        
        # Check if change exceeds threshold
        if threshold_percent > 0:  # Looking for increases (bad)
            regression_detected = change_percent > threshold_percent
        else:  # Looking for decreases (bad)
            regression_detected = change_percent < threshold_percent
        
        if regression_detected:
            # Determine severity
            severity = "critical" if abs(change_percent) > abs(threshold_percent) * 2 else "warning"
            
            return RegressionAlert(
                metric_name=metric_name,
                current_value=current_value,
                baseline_value=baseline_value,
                change_percent=change_percent,
                threshold_percent=threshold_percent,
                severity=severity,
                timestamp=datetime.now().isoformat(),
                configuration_name=config_name,
                details=details
            )
        
        return None
    
    def get_recent_alerts(self, hours: int = 24) -> List[RegressionAlert]:
        """
        Get regression alerts from the last N hours.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of recent RegressionAlert objects
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_alerts = []
        for alert in self.alerts:
            try:
                alert_time = datetime.fromisoformat(alert.timestamp)
                if alert_time >= cutoff_time:
                    recent_alerts.append(alert)
            except ValueError:
                # Skip alerts with invalid timestamps
                continue
        
        return recent_alerts
    
    def clear_alerts(self):
        """Clear all stored alerts."""
        self.alerts.clear()
        logger.info("Cleared all regression alerts")

def create_regression_tester(baseline_file: Optional[Path] = None,
                           thresholds: Optional[RegressionThresholds] = None):
    """
    Factory function to create a performance regression tester.
    
    Args:
        baseline_file: Path to baseline file
        thresholds: Regression detection thresholds
        
    Returns:
        PerformanceRegressionTester instance
    """
    from .performance_monitor import PerformanceMonitor, PerformanceBenchmark
    
    class PerformanceRegressionTester:
        """Main class for performance regression testing."""
        
        def __init__(self, baseline_file: Optional[Path] = None,
                     thresholds: Optional[RegressionThresholds] = None):
            """
            Initialize performance regression tester.
            
            Args:
                baseline_file: Path to baseline file
                thresholds: Regression detection thresholds
            """
            self.baseline_manager = PerformanceBaseline(baseline_file)
            self.detector = RegressionDetector(self.baseline_manager, thresholds)
        
        def run_regression_test(self, benchmark: PerformanceBenchmark,
                               configurations: Dict[str, Dict[str, Any]],
                               runs_per_config: int = 5) -> Tuple[List[BenchmarkResult], List[RegressionAlert]]:
            """
            Run performance regression test.
            
            Args:
                benchmark: PerformanceBenchmark instance
                configurations: Configurations to test
                runs_per_config: Number of runs per configuration
                
            Returns:
                Tuple of (benchmark results, regression alerts)
            """
            logger.info("Running performance regression test...")
            
            # Define a simple benchmark function for regression testing
            def regression_benchmark_function(config):
                # This would typically call your actual model generation function
                # For now, we'll simulate with a sleep
                import time
                time.sleep(config.get("simulated_latency", 0.1))
            
            # Run benchmarks
            results = benchmark.run_benchmark(
                regression_benchmark_function,
                configurations,
                runs_per_config=runs_per_config,
                warmup_runs=1
            )
            
            # Check for regressions
            alerts = self.detector.check_for_regressions(results)
            
            if alerts:
                logger.warning(f"Detected {len(alerts)} performance regressions")
                for alert in alerts:
                    logger.warning(f"Regression in {alert.configuration_name}.{alert.metric_name}: "
                                 f"{alert.change_percent:.1f}% change "
                                 f"(threshold: {alert.threshold_percent:.1f}%)")
            else:
                logger.info("No performance regressions detected")
            
            return results, alerts
    
    return PerformanceRegressionTester(baseline_file, thresholds)