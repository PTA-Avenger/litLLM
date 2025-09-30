"""
Evaluation comparison system for generated vs. target poetry.

This module provides comprehensive comparison functionality including
side-by-side analysis, statistical analysis, visualization, and reporting.
"""

import math
import statistics
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from collections import defaultdict
import json

from .evaluation_metrics import QuantitativeEvaluator, EvaluationResult


@dataclass
class ComparisonResult:
    """Container for detailed comparison results."""
    generated_text: str
    target_text: str
    generated_metrics: Dict[str, Any]
    target_metrics: Dict[str, Any]
    metric_differences: Dict[str, float]
    statistical_analysis: Dict[str, Any]
    similarity_scores: Dict[str, float]
    overall_comparison_score: float


@dataclass
class MetricStatistics:
    """Statistical analysis for a specific metric."""
    metric_name: str
    generated_value: float
    target_value: float
    absolute_difference: float
    relative_difference: float
    percentage_difference: float
    z_score: Optional[float] = None
    significance_level: str = "normal"


class EvaluationComparator:
    """Main class for comparing generated and target poetry evaluations."""
    
    def __init__(self):
        """Initialize the comparator with required evaluator."""
        self.evaluator = QuantitativeEvaluator()
    
    def compare_poetry_side_by_side(self, generated_text: str, target_text: str) -> ComparisonResult:
        """
        Perform comprehensive side-by-side comparison of generated vs target poetry.
        
        Args:
            generated_text: Generated poetry text
            target_text: Target poetry text for comparison
            
        Returns:
            ComparisonResult containing detailed comparison analysis
        """
        # Get comprehensive evaluation for both texts
        generated_eval = self.evaluator.evaluate_poetry(generated_text, target_text)
        target_eval = self.evaluator.evaluate_poetry(target_text)
        
        # Extract metrics for comparison
        generated_metrics = {
            'lexical': generated_eval.lexical_metrics,
            'structural': generated_eval.structural_metrics,
            'readability': generated_eval.readability_metrics
        }
        
        target_metrics = {
            'lexical': target_eval.lexical_metrics,
            'structural': target_eval.structural_metrics,
            'readability': target_eval.readability_metrics
        }
        
        # Calculate metric differences
        metric_differences = self._calculate_metric_differences(generated_metrics, target_metrics)
        
        # Perform statistical analysis
        statistical_analysis = self._perform_statistical_analysis(generated_metrics, target_metrics)
        
        # Calculate similarity scores
        similarity_scores = self._calculate_detailed_similarity_scores(generated_metrics, target_metrics)
        
        # Calculate overall comparison score
        overall_score = self._calculate_overall_comparison_score(similarity_scores)
        
        return ComparisonResult(
            generated_text=generated_text,
            target_text=target_text,
            generated_metrics=generated_metrics,
            target_metrics=target_metrics,
            metric_differences=metric_differences,
            statistical_analysis=statistical_analysis,
            similarity_scores=similarity_scores,
            overall_comparison_score=overall_score
        )
    
    def analyze_metric_differences(self, generated_metrics: Dict[str, Any], 
                                     target_metrics: Dict[str, Any]) -> List[MetricStatistics]:
        """
        Analyze differences between generated and target metrics.
        
        Args:
            generated_metrics: Metrics from generated poetry
            target_metrics: Metrics from target poetry
            
        Returns:
            List of MetricStatistics objects with detailed analysis
        """
        statistics_list = []
        
        # Flatten metrics for comparison
        flat_generated = self._flatten_metrics(generated_metrics)
        flat_target = self._flatten_metrics(target_metrics)
        
        # Analyze each common metric
        common_metrics = set(flat_generated.keys()) & set(flat_target.keys())
        
        for metric_name in common_metrics:
            gen_value = flat_generated[metric_name]
            target_value = flat_target[metric_name]
            
            # Skip non-numeric values
            if not isinstance(gen_value, (int, float)) or not isinstance(target_value, (int, float)):
                continue
            
            # Calculate differences
            abs_diff = abs(gen_value - target_value)
            
            # Relative difference (avoiding division by zero)
            if target_value != 0:
                rel_diff = (gen_value - target_value) / target_value
                pct_diff = rel_diff * 100
            else:
                rel_diff = float('inf') if gen_value != 0 else 0.0
                pct_diff = float('inf') if gen_value != 0 else 0.0
            
            # Determine significance level
            significance = self._determine_significance_level(abs_diff, target_value)
            
            stat = MetricStatistics(
                metric_name=metric_name,
                generated_value=gen_value,
                target_value=target_value,
                absolute_difference=abs_diff,
                relative_difference=rel_diff,
                percentage_difference=pct_diff,
                significance_level=significance
            )
            
            statistics_list.append(stat)
        
        # Sort by absolute difference (descending)
        statistics_list.sort(key=lambda x: x.absolute_difference, reverse=True)
        
        return statistics_list

    def generate_comparison_visualization_data(self, comparison_result: ComparisonResult) -> Dict[str, Any]:
        """
        Generate data structure suitable for creating visualizations.
        
        Args:
            comparison_result: Result from compare_poetry_side_by_side
            
        Returns:
            Dictionary containing visualization-ready data
        """
        # Prepare data for different types of visualizations
        viz_data = {
            'bar_chart_data': self._prepare_bar_chart_data(comparison_result),
            'radar_chart_data': self._prepare_radar_chart_data(comparison_result),
            'scatter_plot_data': self._prepare_scatter_plot_data(comparison_result),
            'difference_heatmap_data': self._prepare_heatmap_data(comparison_result),
            'summary_stats': self._prepare_summary_statistics(comparison_result)
        }
        
        return viz_data
    
    def create_comprehensive_report(self, comparison_result: ComparisonResult) -> Dict[str, Any]:
        """
        Create a comprehensive evaluation report.
        
        Args:
            comparison_result: Result from compare_poetry_side_by_side
            
        Returns:
            Dictionary containing comprehensive report data
        """
        # Analyze metric differences
        metric_stats = self.analyze_metric_differences(
            comparison_result.generated_metrics,
            comparison_result.target_metrics
        )
        
        # Generate visualization data
        viz_data = self.generate_comparison_visualization_data(comparison_result)
        
        # Create summary insights
        insights = self._generate_insights(comparison_result, metric_stats)
        
        # Prepare the comprehensive report
        report = {
            'executive_summary': {
                'overall_similarity': comparison_result.overall_comparison_score,
                'total_metrics_compared': len(metric_stats),
                'significant_differences': len([s for s in metric_stats if s.significance_level == 'high']),
                'key_insights': insights['key_insights']
            },
            'detailed_comparison': {
                'generated_text': comparison_result.generated_text,
                'target_text': comparison_result.target_text,
                'generated_metrics': comparison_result.generated_metrics,
                'target_metrics': comparison_result.target_metrics
            },
            'statistical_analysis': {
                'metric_statistics': [asdict(stat) for stat in metric_stats],
                'similarity_scores': comparison_result.similarity_scores,
                'statistical_summary': comparison_result.statistical_analysis
            },
            'visualization_data': viz_data,
            'insights_and_recommendations': insights,
            'metadata': {
                'report_version': '1.0',
                'comparison_timestamp': None,  # Would be set by caller
                'evaluation_method': 'quantitative_comparison'
            }
        }
        
        return report
    
    def _calculate_metric_differences(self, generated_metrics: Dict[str, Any], 
                                      target_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate differences between metric sets."""
        differences = {}
        
        flat_generated = self._flatten_metrics(generated_metrics)
        flat_target = self._flatten_metrics(target_metrics)
        
        common_metrics = set(flat_generated.keys()) & set(flat_target.keys())
        
        for metric in common_metrics:
            gen_val = flat_generated[metric]
            target_val = flat_target[metric]
            
            if isinstance(gen_val, (int, float)) and isinstance(target_val, (int, float)):
                differences[metric] = gen_val - target_val
        
        return differences
    
    def _perform_statistical_analysis(self, generated_metrics: Dict[str, Any], 
                                    target_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical analysis on metric differences."""
        flat_generated = self._flatten_metrics(generated_metrics)
        flat_target = self._flatten_metrics(target_metrics)
        
        # Calculate basic statistics
        differences = []
        absolute_differences = []
        relative_differences = []
        
        common_metrics = set(flat_generated.keys()) & set(flat_target.keys())
        
        for metric in common_metrics:
            gen_val = flat_generated[metric]
            target_val = flat_target[metric]
            
            if isinstance(gen_val, (int, float)) and isinstance(target_val, (int, float)):
                diff = gen_val - target_val
                abs_diff = abs(diff)
                
                differences.append(diff)
                absolute_differences.append(abs_diff)
                
                if target_val != 0:
                    rel_diff = diff / target_val
                    relative_differences.append(rel_diff)
        
        # Calculate statistics
        analysis = {}
        
        if differences:
            analysis['mean_difference'] = statistics.mean(differences)
            analysis['median_difference'] = statistics.median(differences)
            analysis['std_difference'] = statistics.stdev(differences) if len(differences) > 1 else 0.0
            
        if absolute_differences:
            analysis['mean_absolute_difference'] = statistics.mean(absolute_differences)
            analysis['median_absolute_difference'] = statistics.median(absolute_differences)
            
        if relative_differences:
            analysis['mean_relative_difference'] = statistics.mean(relative_differences)
            analysis['median_relative_difference'] = statistics.median(relative_differences)
        
        analysis['total_metrics_compared'] = len(common_metrics)
        
        return analysis
    
    def _calculate_detailed_similarity_scores(self, generated_metrics: Dict[str, Any], 
                                              target_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate detailed similarity scores for different metric categories."""
        similarity_scores = {}
        
        # Calculate similarity for each category
        for category in ['lexical', 'structural', 'readability']:
            if category in generated_metrics and category in target_metrics:
                similarity = self.evaluator._calculate_metric_similarity(
                    generated_metrics[category],
                    target_metrics[category]
                )
                similarity_scores[f'{category}_similarity'] = similarity
        
        # Calculate overall similarity
        category_similarities = [score for score in similarity_scores.values()]
        if category_similarities:
            similarity_scores['overall_similarity'] = statistics.mean(category_similarities)
        else:
            similarity_scores['overall_similarity'] = 0.0
        
        return similarity_scores
    
    def _calculate_overall_comparison_score(self, similarity_scores: Dict[str, float]) -> float:
        """Calculate overall comparison score."""
        return similarity_scores.get('overall_similarity', 0.0)
    
    def _flatten_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Union[int, float]]:
        """Flatten nested metrics dictionary."""
        flat = {}
        
        for category, category_metrics in metrics.items():
            if isinstance(category_metrics, dict):
                for metric_name, value in category_metrics.items():
                    flat_key = f"{category}_{metric_name}"
                    flat[flat_key] = value
            else:
                flat[category] = category_metrics
        
        return flat
    
    def _determine_significance_level(self, abs_difference: float, target_value: float) -> str:
        """Determine significance level of difference."""
        if target_value == 0:
            return "high" if abs_difference > 0 else "none"
        
        relative_diff = abs_difference / abs(target_value)
        
        if relative_diff < 0.05:  # Less than 5% difference
            return "low"
        elif relative_diff < 0.15:  # Less than 15% difference
            return "medium"
        else:
            return "high"
    
    def _prepare_bar_chart_data(self, comparison_result: ComparisonResult) -> Dict[str, Any]:
        """Prepare data for bar chart visualization."""
        flat_generated = self._flatten_metrics(comparison_result.generated_metrics)
        flat_target = self._flatten_metrics(comparison_result.target_metrics)
        
        metrics = []
        generated_values = []
        target_values = []
        
        common_metrics = set(flat_generated.keys()) & set(flat_target.keys())
        
        for metric in sorted(common_metrics):
            gen_val = flat_generated[metric]
            target_val = flat_target[metric]
            
            if isinstance(gen_val, (int, float)) and isinstance(target_val, (int, float)):
                metrics.append(metric)
                generated_values.append(gen_val)
                target_values.append(target_val)
        
        return {
            'metrics': metrics,
            'generated_values': generated_values,
            'target_values': target_values,
            'chart_type': 'grouped_bar'
        }
    
    def _prepare_radar_chart_data(self, comparison_result: ComparisonResult) -> Dict[str, Any]:
        """Prepare data for radar chart visualization."""
        # Use similarity scores for radar chart
        categories = []
        similarity_values = []
        
        for category, score in comparison_result.similarity_scores.items():
            if category != 'overall_similarity':
                categories.append(category.replace('_similarity', '').title())
                similarity_values.append(score)
        
        return {
            'categories': categories,
            'similarity_scores': similarity_values,
            'chart_type': 'radar'
        }
    
    def _prepare_scatter_plot_data(self, comparison_result: ComparisonResult) -> Dict[str, Any]:
        """Prepare data for scatter plot visualization."""
        flat_generated = self._flatten_metrics(comparison_result.generated_metrics)
        flat_target = self._flatten_metrics(comparison_result.target_metrics)
        
        x_values = []  # Target values
        y_values = []  # Generated values
        labels = []
        
        common_metrics = set(flat_generated.keys()) & set(flat_target.keys())
        
        for metric in common_metrics:
            gen_val = flat_generated[metric]
            target_val = flat_target[metric]
            
            if isinstance(gen_val, (int, float)) and isinstance(target_val, (int, float)):
                x_values.append(target_val)
                y_values.append(gen_val)
                labels.append(metric)
        
        return {
            'x_values': x_values,
            'y_values': y_values,
            'labels': labels,
            'chart_type': 'scatter',
            'x_label': 'Target Values',
            'y_label': 'Generated Values'
        }
    
    def _prepare_heatmap_data(self, comparison_result: ComparisonResult) -> Dict[str, Any]:
        """Prepare data for difference heatmap visualization."""
        # Create a matrix of relative differences
        categories = ['lexical', 'structural', 'readability']
        metrics_by_category = defaultdict(list)
        
        # Group metrics by category
        for category in categories:
            if category in comparison_result.generated_metrics:
                for metric_name in comparison_result.generated_metrics[category].keys():
                    metrics_by_category[category].append(metric_name)
        
        # Calculate relative differences matrix
        heatmap_data = []
        row_labels = []
        col_labels = categories
        
        for category in categories:
            if category in metrics_by_category:
                for metric in metrics_by_category[category]:
                    gen_val = comparison_result.generated_metrics[category].get(metric, 0)
                    target_val = comparison_result.target_metrics[category].get(metric, 0)
                    
                    if isinstance(gen_val, (int, float)) and isinstance(target_val, (int, float)):
                        if target_val != 0:
                            rel_diff = (gen_val - target_val) / target_val
                        else:
                            rel_diff = 1.0 if gen_val != 0 else 0.0
                        
                        row_labels.append(f"{category}_{metric}")
                        # Create row with difference for this category, 0 for others
                        row = [0.0] * len(categories)
                        row[categories.index(category)] = rel_diff
                        heatmap_data.append(row)
        
        return {
            'data_matrix': heatmap_data,
            'row_labels': row_labels,
            'col_labels': col_labels,
            'chart_type': 'heatmap',
            'color_scale': 'diverging'
        }
    
    def _prepare_summary_statistics(self, comparison_result: ComparisonResult) -> Dict[str, Any]:
        """Prepare summary statistics for visualization."""
        return {
            'overall_similarity': comparison_result.overall_comparison_score,
            'similarity_breakdown': comparison_result.similarity_scores,
            'statistical_summary': comparison_result.statistical_analysis,
            'total_metrics': len(self._flatten_metrics(comparison_result.generated_metrics))
        }
    
    def _generate_insights(self, comparison_result: ComparisonResult, 
                           metric_stats: List[MetricStatistics]) -> Dict[str, Any]:
        """Generate insights and recommendations from comparison results."""
        insights = {
            'key_insights': [],
            'strengths': [],
            'areas_for_improvement': [],
            'recommendations': []
        }
        
        # Overall similarity insight
        overall_sim = comparison_result.overall_comparison_score
        if overall_sim >= 0.8:
            insights['key_insights'].append("Generated poetry shows high similarity to target style")
            insights['strengths'].append("Strong overall stylistic alignment")
        elif overall_sim >= 0.6:
            insights['key_insights'].append("Generated poetry shows moderate similarity to target style")
        else:
            insights['key_insights'].append("Generated poetry shows low similarity to target style")
            insights['areas_for_improvement'].append("Overall stylistic alignment needs improvement")
        
        # Category-specific insights
        for category, score in comparison_result.similarity_scores.items():
            if category.endswith('_similarity') and category != 'overall_similarity':
                category_name = category.replace('_similarity', '')
                
                if score >= 0.8:
                    insights['strengths'].append(f"Strong {category_name} similarity")
                elif score < 0.5:
                    insights['areas_for_improvement'].append(f"Weak {category_name} similarity")
                    insights['recommendations'].append(f"Focus on improving {category_name} features")
        
        # Metric-specific insights
        high_diff_metrics = [stat for stat in metric_stats if stat.significance_level == 'high']
        if high_diff_metrics:
            top_diff = high_diff_metrics[0]  # Already sorted by difference
            insights['areas_for_improvement'].append(
                f"Largest difference in {top_diff.metric_name} "
                f"({top_diff.percentage_difference:.1f}% difference)"
            )
        
        # Generate recommendations
        if overall_sim < 0.7:
            insights['recommendations'].append("Consider additional fine-tuning with more target-specific data")
        
        if len(high_diff_metrics) > 3:
            insights['recommendations'].append("Multiple metrics show significant differences - review training approach")
        
        return insights


# Convenience functions for direct use
def compare_poetry_texts(generated_text: str, target_text: str) -> ComparisonResult:
    """Compare two poetry texts using default comparator."""
    comparator = EvaluationComparator()
    return comparator.compare_poetry_side_by_side(generated_text, target_text)


def generate_comparison_report(generated_text: str, target_text: str) -> Dict[str, Any]:
    """Generate comprehensive comparison report."""
    comparator = EvaluationComparator()
    comparison_result = comparator.compare_poetry_side_by_side(generated_text, target_text)
    return comparator.create_comprehensive_report(comparison_result)


def analyze_poetry_differences(generated_text: str, target_text: str) -> List[MetricStatistics]:
    """Analyze differences between generated and target poetry."""
    comparator = EvaluationComparator()
    comparison_result = comparator.compare_poetry_side_by_side(generated_text, target_text)
    return comparator.analyze_metric_differences(
        comparison_result.generated_metrics,
        comparison_result.target_metrics
    )