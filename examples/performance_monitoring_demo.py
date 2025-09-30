#!/usr/bin/env python3
"""
Performance monitoring demonstration for the stylistic poetry LLM.

This example shows how to use the performance monitoring capabilities
to track latency, memory usage, and run benchmarks on poetry generation.
"""

import sys
import time
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stylometric.performance_monitor import (
    create_performance_monitor,
    create_benchmark_suite,
    PerformanceMetrics,
    BenchmarkResult
)
from stylometric.performance_regression import (
    create_regression_tester,
    RegressionThresholds
)
from stylometric.model_interface import (
    PoetryGenerationRequest,
    GenerationConfig,
    create_poetry_model
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def simulate_poetry_generation(config):
    """
    Simulate poetry generation with different configurations.
    
    In a real implementation, this would call the actual GPT model.
    For demonstration, we simulate different latencies and memory usage.
    """
    # Simulate different generation complexities
    complexity = config.get("complexity", 1)
    temperature = config.get("temperature", 0.8)
    max_length = config.get("max_length", 100)
    
    # Simulate processing time based on complexity
    base_time = 0.1
    processing_time = base_time * complexity * (temperature + 0.5)
    time.sleep(processing_time)
    
    # Simulate memory allocation
    memory_data = [0] * (1000 * complexity * max_length // 10)
    
    # Simulate occasional failures
    if config.get("failure_rate", 0) > 0:
        import random
        if random.random() < config["failure_rate"]:
            raise RuntimeError("Simulated generation failure")
    
    return f"Generated poem with {len(memory_data)} tokens"


def demonstrate_basic_monitoring():
    """Demonstrate basic performance monitoring."""
    print("\n" + "="*60)
    print("BASIC PERFORMANCE MONITORING DEMONSTRATION")
    print("="*60)
    
    # Create performance monitor
    monitor = create_performance_monitor(enable_gpu_monitoring=False)
    
    # Monitor a single operation
    print("\n1. Monitoring a single poetry generation operation:")
    
    with monitor.monitor_operation("poetry_generation", "gpt2", {"temperature": 0.8}) as metrics:
        result = simulate_poetry_generation({"complexity": 2, "temperature": 0.8, "max_length": 150})
        print(f"   Generated: {result[:50]}...")
    
    print(f"   Latency: {metrics.latency_ms:.2f}ms")
    print(f"   Memory Usage: {metrics.memory_usage_mb:.1f}MB")
    print(f"   Peak Memory: {metrics.peak_memory_mb:.1f}MB")
    print(f"   CPU Usage: {metrics.cpu_usage_percent:.1f}%")
    
    # Monitor multiple operations
    print("\n2. Monitoring multiple operations:")
    
    for i in range(3):
        config = {"complexity": i + 1, "temperature": 0.7 + i * 0.1, "max_length": 100 + i * 50}
        with monitor.monitor_operation(f"generation_{i+1}", "gpt2", config):
            simulate_poetry_generation(config)
    
    # Show recent metrics
    recent_metrics = monitor.get_recent_metrics(5)
    print(f"\n   Recorded {len(recent_metrics)} operations")
    
    for i, metric in enumerate(recent_metrics):
        print(f"   Operation {i+1}: {metric.operation} - {metric.latency_ms:.2f}ms")
    
    # Calculate statistics
    stats = monitor.calculate_operation_stats("poetry_generation")
    if stats:
        print(f"\n3. Statistics for 'poetry_generation' operations:")
        print(f"   Count: {stats['count']}")
        print(f"   Average Latency: {stats['avg_latency_ms']:.2f}ms")
        print(f"   Median Latency: {stats['median_latency_ms']:.2f}ms")
        print(f"   P95 Latency: {stats['p95_latency_ms']:.2f}ms")
        print(f"   Average Memory: {stats['avg_memory_mb']:.1f}MB")


def demonstrate_benchmarking():
    """Demonstrate performance benchmarking."""
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARKING DEMONSTRATION")
    print("="*60)
    
    # Create monitor and benchmark suite
    monitor = create_performance_monitor(enable_gpu_monitoring=False)
    benchmark = create_benchmark_suite(monitor)
    
    # Define test configurations
    configurations = {
        "low_complexity": {
            "complexity": 1,
            "temperature": 0.7,
            "max_length": 100,
            "failure_rate": 0.0
        },
        "medium_complexity": {
            "complexity": 2,
            "temperature": 0.8,
            "max_length": 200,
            "failure_rate": 0.05
        },
        "high_complexity": {
            "complexity": 3,
            "temperature": 0.9,
            "max_length": 300,
            "failure_rate": 0.1
        }
    }
    
    print(f"\nRunning benchmark with {len(configurations)} configurations...")
    print("Configurations:")
    for name, config in configurations.items():
        print(f"  - {name}: complexity={config['complexity']}, "
              f"temp={config['temperature']}, max_len={config['max_length']}")
    
    # Run benchmark
    results = benchmark.run_benchmark(
        simulate_poetry_generation,
        configurations,
        runs_per_config=5,
        warmup_runs=2
    )
    
    # Display results
    print(f"\nBenchmark Results:")
    print("-" * 80)
    print(f"{'Configuration':<20} {'Avg Latency':<12} {'P95 Latency':<12} {'Memory':<10} {'Success Rate':<12}")
    print("-" * 80)
    
    for result in results:
        print(f"{result.configuration_name:<20} "
              f"{result.average_latency_ms:<12.2f} "
              f"{result.p95_latency_ms:<12.2f} "
              f"{result.average_memory_mb:<10.1f} "
              f"{result.success_rate:<12.2%}")
    
    # Save benchmark results
    results_file = Path("benchmark_results.json")
    benchmark.save_benchmark_results(results_file)
    print(f"\nBenchmark results saved to {results_file}")
    
    return results


def demonstrate_regression_testing(benchmark_results):
    """Demonstrate performance regression testing."""
    print("\n" + "="*60)
    print("PERFORMANCE REGRESSION TESTING DEMONSTRATION")
    print("="*60)
    
    # Create regression tester with custom thresholds
    thresholds = RegressionThresholds(
        latency_increase_percent=15.0,  # Alert if latency increases by more than 15%
        memory_increase_percent=10.0,   # Alert if memory increases by more than 10%
        success_rate_decrease_percent=3.0,  # Alert if success rate drops by more than 3%
        min_samples=3
    )
    
    tester = create_regression_tester(
        baseline_file=Path("performance_baseline.json"),
        thresholds=thresholds
    )
    
    print("\n1. Establishing performance baselines...")
    
    # Establish baselines from benchmark results
    for result in benchmark_results:
        baseline_metrics = {
            "average_latency_ms": result.average_latency_ms,
            "median_latency_ms": result.median_latency_ms,
            "p95_latency_ms": result.p95_latency_ms,
            "average_memory_mb": result.average_memory_mb,
            "peak_memory_mb": result.peak_memory_mb,
            "success_rate": result.success_rate
        }
        
        tester.baseline_manager.update_baseline(result.configuration_name, baseline_metrics)
        print(f"   Baseline established for {result.configuration_name}")
    
    print("\n2. Simulating performance regression...")
    
    # Create monitor and benchmark for regression test
    monitor = create_performance_monitor(enable_gpu_monitoring=False)
    benchmark = create_benchmark_suite(monitor)
    
    # Define configurations with intentional regressions
    regression_configurations = {
        "low_complexity": {
            "complexity": 1,
            "temperature": 0.7,
            "max_length": 100,
            "failure_rate": 0.0
        },
        "medium_complexity": {
            "complexity": 3,  # Increased complexity (regression)
            "temperature": 0.8,
            "max_length": 250,  # Increased length (regression)
            "failure_rate": 0.15  # Increased failure rate (regression)
        },
        "high_complexity": {
            "complexity": 3,
            "temperature": 0.9,
            "max_length": 300,
            "failure_rate": 0.1
        }
    }
    
    # Run regression test
    results, alerts = tester.run_regression_test(
        benchmark,
        regression_configurations,
        runs_per_config=4
    )
    
    print(f"\n3. Regression test results:")
    
    if alerts:
        print(f"\n   ⚠️  PERFORMANCE REGRESSIONS DETECTED ({len(alerts)} alerts):")
        print("-" * 60)
        
        for alert in alerts:
            severity_icon = "🔴" if alert.severity == "critical" else "🟡"
            print(f"   {severity_icon} {alert.configuration_name}.{alert.metric_name}")
            print(f"      Current: {alert.current_value:.2f}")
            print(f"      Baseline: {alert.baseline_value:.2f}")
            print(f"      Change: {alert.change_percent:+.1f}% (threshold: {alert.threshold_percent:+.1f}%)")
            print(f"      Severity: {alert.severity.upper()}")
            print()
    else:
        print("   ✅ No performance regressions detected!")
    
    # Generate and display regression report
    report = tester.generate_regression_report(alerts) if hasattr(tester, 'generate_regression_report') else None
    if report:
        print("\n4. Regression Report:")
        print("-" * 40)
        print(report)


def demonstrate_real_model_integration():
    """Demonstrate integration with actual poetry model (if available)."""
    print("\n" + "="*60)
    print("REAL MODEL INTEGRATION DEMONSTRATION")
    print("="*60)
    
    try:
        # Try to create a real poetry model
        model = create_poetry_model("gpt", "gpt2")
        
        print("\n1. Loading model...")
        if model.load_model():
            print("   ✅ Model loaded successfully")
            
            # Create performance monitor
            monitor = create_performance_monitor(enable_gpu_monitoring=False)
            
            print("\n2. Monitoring real poetry generation...")
            
            # Test different generation configurations
            test_configs = [
                GenerationConfig(max_length=50, temperature=0.7),
                GenerationConfig(max_length=100, temperature=0.8),
                GenerationConfig(max_length=150, temperature=0.9)
            ]
            
            for i, config in enumerate(test_configs):
                request = PoetryGenerationRequest(
                    prompt="Write a poem about nature",
                    poet_style="general",
                    generation_config=config
                )
                
                config_name = f"real_generation_{i+1}"
                with monitor.monitor_operation(config_name, "gpt2", config.__dict__) as metrics:
                    response = model.generate_poetry(request)
                
                if response.success:
                    print(f"   Generation {i+1}: {metrics.latency_ms:.2f}ms, "
                          f"{metrics.memory_usage_mb:.1f}MB")
                    print(f"   Generated: {response.generated_text[:100]}...")
                else:
                    print(f"   Generation {i+1} failed: {response.error_message}")
            
            # Show performance statistics
            for i in range(len(test_configs)):
                config_name = f"real_generation_{i+1}"
                stats = monitor.calculate_operation_stats(config_name)
                if stats:
                    print(f"\n   Stats for {config_name}:")
                    print(f"     Average latency: {stats['avg_latency_ms']:.2f}ms")
                    print(f"     Average memory: {stats['avg_memory_mb']:.1f}MB")
            
            # Unload model
            model.unload_model()
            print("\n   Model unloaded")
            
        else:
            print("   ❌ Failed to load model")
            
    except Exception as e:
        print(f"   ⚠️  Real model integration not available: {e}")
        print("   This is expected if transformers/torch are not installed")


def main():
    """Run all performance monitoring demonstrations."""
    print("STYLISTIC POETRY LLM - PERFORMANCE MONITORING DEMO")
    print("=" * 60)
    print("This demo shows how to monitor performance, run benchmarks,")
    print("and detect performance regressions in poetry generation.")
    
    try:
        # Run demonstrations
        demonstrate_basic_monitoring()
        benchmark_results = demonstrate_benchmarking()
        demonstrate_regression_testing(benchmark_results)
        demonstrate_real_model_integration()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETE")
        print("="*60)
        print("\nKey files created:")
        print("- benchmark_results.json: Benchmark results")
        print("- performance_baseline.json: Performance baselines")
        print("\nNext steps:")
        print("- Integrate monitoring into your poetry generation pipeline")
        print("- Set up automated regression testing in CI/CD")
        print("- Customize thresholds based on your performance requirements")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()