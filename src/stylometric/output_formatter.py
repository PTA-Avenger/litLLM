"""
Enhanced output formatting and display for poetry generation results.

This module provides comprehensive formatting capabilities for displaying
generated poetry with stylistic analysis, comparison results, and saving
options with evaluation metrics.
"""

import json
import textwrap
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
from dataclasses import asdict

from .evaluation_comparison import ComparisonResult, MetricStatistics


class PoetryOutputFormatter:
    """Enhanced formatter for poetry generation output and analysis."""
    
    def __init__(self, width: int = 80, indent: str = "  "):
        """
        Initialize the formatter.
        
        Args:
            width: Maximum line width for text wrapping
            indent: Indentation string for nested content
        """
        self.width = width
        self.indent = indent
        self.double_indent = indent * 2
    
    def format_poem_display(self, poem_text: str, title: Optional[str] = None) -> str:
        """
        Format poem text for attractive display.
        
        Args:
            poem_text: The poem text to format
            title: Optional title for the poem
            
        Returns:
            Formatted poem display string
        """
        lines = []
        
        # Add decorative border
        border = "═" * min(self.width, 60)
        lines.append(border)
        
        # Add title if provided
        if title:
            title_line = f" {title.upper()} "
            padding = (len(border) - len(title_line)) // 2
            lines.append("═" * padding + title_line + "═" * (len(border) - padding - len(title_line)))
            lines.append(border)
        
        # Add poem content with proper spacing
        lines.append("")
        poem_lines = poem_text.strip().split('\n')
        
        for line in poem_lines:
            if line.strip():  # Non-empty line
                # Center shorter lines, left-align longer ones
                if len(line.strip()) < self.width - 10:
                    centered_line = line.strip().center(self.width - 4)
                    lines.append(f"  {centered_line}")
                else:
                    # Wrap long lines
                    wrapped = textwrap.fill(line.strip(), width=self.width - 4)
                    for wrapped_line in wrapped.split('\n'):
                        lines.append(f"  {wrapped_line}")
            else:  # Empty line (stanza break)
                lines.append("")
        
        lines.append("")
        lines.append(border)
        
        return '\n'.join(lines)    

    def format_stylistic_analysis(self, analysis_results: Dict[str, Any], 
                                  detailed: bool = False) -> str:
        """
        Format stylistic analysis results for display.
        
        Args:
            analysis_results: Analysis results dictionary
            detailed: Whether to show detailed analysis
            
        Returns:
            Formatted analysis display string
        """
        lines = []
        
        # Header
        lines.append("┌" + "─" * (self.width - 2) + "┐")
        lines.append("│" + " STYLISTIC ANALYSIS ".center(self.width - 2) + "│")
        lines.append("├" + "─" * (self.width - 2) + "┤")
        
        # Basic metrics
        lines.append("│ BASIC METRICS" + " " * (self.width - 16) + "│")
        lines.append("├" + "─" * (self.width - 2) + "┤")
        
        basic_metrics = [
            ("Word count", analysis_results.get('word_count', 'N/A')),
            ("Line count", analysis_results.get('line_count', 'N/A')),
            ("Stanza count", analysis_results.get('stanza_count', 'N/A')),
            ("Average words per line", self._format_number(analysis_results.get('avg_words_per_line', 'N/A'))),
        ]
        
        for label, value in basic_metrics:
            line = f"│ {label:<25} {str(value):>10} │"
            lines.append(line + " " * (self.width - len(line) - 1) + "│")
        
        # Lexical metrics
        if detailed or any(key in analysis_results for key in ['ttr', 'lexical_density']):
            lines.append("├" + "─" * (self.width - 2) + "┤")
            lines.append("│ LEXICAL METRICS" + " " * (self.width - 18) + "│")
            lines.append("├" + "─" * (self.width - 2) + "┤")
            
            lexical_metrics = [
                ("Type-Token Ratio", self._format_number(analysis_results.get('ttr', 'N/A'))),
                ("Lexical density", self._format_number(analysis_results.get('lexical_density', 'N/A'))),
                ("Avg word length", self._format_number(analysis_results.get('avg_word_length', 'N/A'))),
            ]
            
            for label, value in lexical_metrics:
                line = f"│ {label:<25} {str(value):>10} │"
                lines.append(line + " " * (self.width - len(line) - 1) + "│")
        
        # Structural metrics
        if detailed or any(key in analysis_results for key in ['syllable_count', 'avg_syllables_per_line']):
            lines.append("├" + "─" * (self.width - 2) + "┤")
            lines.append("│ STRUCTURAL METRICS" + " " * (self.width - 21) + "│")
            lines.append("├" + "─" * (self.width - 2) + "┤")
            
            structural_metrics = [
                ("Total syllables", analysis_results.get('syllable_count', 'N/A')),
                ("Avg syllables/line", self._format_number(analysis_results.get('avg_syllables_per_line', 'N/A'))),
                ("Rhyme scheme", analysis_results.get('rhyme_scheme', 'N/A')),
                ("Meter pattern", analysis_results.get('meter_pattern', 'N/A')),
            ]
            
            for label, value in structural_metrics:
                line = f"│ {label:<25} {str(value):>10} │"
                lines.append(line + " " * (self.width - len(line) - 1) + "│")
        
        lines.append("└" + "─" * (self.width - 2) + "┘")
        
        return '\n'.join(lines)
    
    def format_comparison_analysis(self, comparison_result: ComparisonResult) -> str:
        """
        Format comparison analysis results for display.
        
        Args:
            comparison_result: Comparison result from evaluation
            
        Returns:
            Formatted comparison display string
        """
        lines = []
        
        # Header
        lines.append("┌" + "─" * (self.width - 2) + "┐")
        lines.append("│" + " STYLE COMPARISON ANALYSIS ".center(self.width - 2) + "│")
        lines.append("├" + "─" * (self.width - 2) + "┤")
        
        # Overall similarity
        overall_sim = comparison_result.overall_comparison_score
        sim_bar = self._create_progress_bar(overall_sim, width=20)
        lines.append(f"│ Overall Similarity: {overall_sim:.3f} {sim_bar} │")
        lines.append("├" + "─" * (self.width - 2) + "┤")
        
        # Category similarities
        lines.append("│ CATEGORY SIMILARITIES" + " " * (self.width - 24) + "│")
        lines.append("├" + "─" * (self.width - 2) + "┤")
        
        for category, score in comparison_result.similarity_scores.items():
            if category != 'overall_similarity':
                category_name = category.replace('_similarity', '').replace('_', ' ').title()
                score_bar = self._create_progress_bar(score, width=15)
                line = f"│ {category_name:<15} {score:.3f} {score_bar} │"
                lines.append(line + " " * (self.width - len(line) - 1) + "│")
        
        # Top differences
        if hasattr(comparison_result, 'metric_differences'):
            lines.append("├" + "─" * (self.width - 2) + "┤")
            lines.append("│ NOTABLE DIFFERENCES" + " " * (self.width - 22) + "│")
            lines.append("├" + "─" * (self.width - 2) + "┤")
            
            # Sort differences by absolute value
            sorted_diffs = sorted(
                comparison_result.metric_differences.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]  # Top 5 differences
            
            for metric, diff in sorted_diffs:
                metric_name = metric.replace('_', ' ').title()[:20]
                diff_str = f"{diff:+.3f}" if isinstance(diff, (int, float)) else str(diff)
                line = f"│ {metric_name:<20} {diff_str:>10} │"
                lines.append(line + " " * (self.width - len(line) - 1) + "│")
        
        lines.append("└" + "─" * (self.width - 2) + "┘")
        
        return '\n'.join(lines) 
   
    def format_generation_metadata(self, metadata: Dict[str, Any]) -> str:
        """
        Format generation metadata for display.
        
        Args:
            metadata: Generation metadata dictionary
            
        Returns:
            Formatted metadata display string
        """
        lines = []
        
        lines.append("┌" + "─" * (self.width - 2) + "┐")
        lines.append("│" + " GENERATION DETAILS ".center(self.width - 2) + "│")
        lines.append("├" + "─" * (self.width - 2) + "┤")
        
        # Format key metadata
        metadata_items = [
            ("Model", metadata.get('model_name', 'N/A')),
            ("Temperature", self._format_number(metadata.get('temperature', 'N/A'))),
            ("Max length", metadata.get('max_length', 'N/A')),
            ("Generation time", f"{metadata.get('generation_time_ms', 0):.0f}ms"),
            ("Tokens generated", metadata.get('tokens_generated', 'N/A')),
        ]
        
        for label, value in metadata_items:
            line = f"│ {label:<20} {str(value):>15} │"
            lines.append(line + " " * (self.width - len(line) - 1) + "│")
        
        lines.append("└" + "─" * (self.width - 2) + "┘")
        
        return '\n'.join(lines)
    
    def create_comprehensive_output(self, poem_text: str, analysis_results: Dict[str, Any],
                                    generation_metadata: Optional[Dict[str, Any]] = None,
                                    comparison_result: Optional[ComparisonResult] = None,
                                    poet_style: Optional[str] = None,
                                    prompt: Optional[str] = None) -> str:
        """
        Create comprehensive formatted output combining all elements.
        
        Args:
            poem_text: Generated poem text
            analysis_results: Stylistic analysis results
            generation_metadata: Generation metadata
            comparison_result: Optional comparison results
            poet_style: Optional poet style used
            prompt: Optional original prompt
            
        Returns:
            Complete formatted output string
        """
        sections = []
        
        # Header with prompt and style info
        if prompt or poet_style:
            header_lines = []
            header_lines.append("═" * self.width)
            header_lines.append(" POETRY GENERATION RESULTS ".center(self.width))
            header_lines.append("═" * self.width)
            
            if prompt:
                header_lines.append(f"Prompt: {prompt}")
            if poet_style:
                header_lines.append(f"Style: {poet_style.replace('_', ' ').title()}")
            
            header_lines.append("═" * self.width)
            sections.append('\n'.join(header_lines))
        
        # Generated poem
        poem_title = f"Generated in {poet_style.replace('_', ' ').title()} Style" if poet_style else "Generated Poem"
        sections.append(self.format_poem_display(poem_text, poem_title))
        
        # Stylistic analysis
        sections.append(self.format_stylistic_analysis(analysis_results, detailed=True))
        
        # Comparison analysis if available
        if comparison_result:
            sections.append(self.format_comparison_analysis(comparison_result))
        
        # Generation metadata if available
        if generation_metadata:
            sections.append(self.format_generation_metadata(generation_metadata))
        
        return '\n\n'.join(sections)   
 
    def save_results_with_formatting(self, output_path: Path, poem_text: str,
                                     analysis_results: Dict[str, Any],
                                     generation_config: Dict[str, Any],
                                     generation_metadata: Dict[str, Any],
                                     comparison_result: Optional[ComparisonResult] = None,
                                     prompt: Optional[str] = None,
                                     poet_style: Optional[str] = None,
                                     theme: Optional[str] = None,
                                     form: Optional[str] = None) -> None:
        """
        Save results with comprehensive formatting and metadata.
        
        Args:
            output_path: Path to save the results
            poem_text: Generated poem text
            analysis_results: Stylistic analysis results
            generation_config: Generation configuration
            generation_metadata: Generation metadata
            comparison_result: Optional comparison results
            prompt: Original prompt
            poet_style: Poet style used
            theme: Theme specified
            form: Poetic form specified
        """
        timestamp = datetime.now()
        
        if output_path.suffix.lower() == '.json':
            # Save as structured JSON
            output_data = {
                'metadata': {
                    'timestamp': timestamp.isoformat(),
                    'prompt': prompt,
                    'poet_style': poet_style,
                    'theme': theme,
                    'form': form,
                    'generation_config': generation_config,
                    'generation_metadata': generation_metadata
                },
                'results': {
                    'generated_text': poem_text,
                    'analysis_results': analysis_results,
                    'comparison_result': asdict(comparison_result) if comparison_result else None
                }
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        elif output_path.suffix.lower() in ['.txt', '.md']:
            # Save as formatted text
            content_lines = []
            
            # Header
            content_lines.append("=" * 80)
            content_lines.append("STYLISTIC POETRY GENERATION RESULTS")
            content_lines.append("=" * 80)
            content_lines.append(f"Generated: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            content_lines.append("")
            
            # Parameters
            content_lines.append("GENERATION PARAMETERS")
            content_lines.append("-" * 40)
            if prompt:
                content_lines.append(f"Prompt: {prompt}")
            if poet_style:
                content_lines.append(f"Poet Style: {poet_style}")
            if theme:
                content_lines.append(f"Theme: {theme}")
            if form:
                content_lines.append(f"Form: {form}")
            
            for key, value in generation_config.items():
                content_lines.append(f"{key.replace('_', ' ').title()}: {value}")
            content_lines.append("")
            
            # Generated poem
            content_lines.append(self.format_poem_display(poem_text))
            content_lines.append("")
            
            # Analysis
            content_lines.append(self.format_stylistic_analysis(analysis_results, detailed=True))
            content_lines.append("")
            
            # Comparison if available
            if comparison_result:
                content_lines.append(self.format_comparison_analysis(comparison_result))
                content_lines.append("")
            
            # Generation details
            content_lines.append(self.format_generation_metadata(generation_metadata))
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(content_lines))
        
        else:
            # Default to plain text
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"Generated: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
                if prompt:
                    f.write(f"Prompt: {prompt}\n")
                if poet_style:
                    f.write(f"Style: {poet_style}\n")
                f.write("\n" + "="*50 + "\n")
                f.write(poem_text)
                f.write("\n" + "="*50 + "\n")
                
                if analysis_results:
                    f.write("\nAnalysis:\n")
                    for key, value in analysis_results.items():
                        f.write(f"{key}: {value}\n")
    
    def _format_number(self, value: Any, decimals: int = 3) -> str:
        """Format numeric values for display."""
        if isinstance(value, (int, float)):
            if isinstance(value, float):
                return f"{value:.{decimals}f}"
            else:
                return str(value)
        return str(value)
    
    def _create_progress_bar(self, value: float, width: int = 20, 
                             filled_char: str = "█", empty_char: str = "░") -> str:
        """Create a visual progress bar for similarity scores."""
        if not isinstance(value, (int, float)) or value < 0:
            return empty_char * width
        
        filled_width = int(min(value, 1.0) * width)
        empty_width = width - filled_width
        
        return filled_char * filled_width + empty_char * empty_width


# Convenience functions
def format_poetry_output(poem_text: str, analysis_results: Dict[str, Any],
                         **kwargs) -> str:
    """Convenience function for formatting poetry output."""
    formatter = PoetryOutputFormatter()
    return formatter.create_comprehensive_output(poem_text, analysis_results, **kwargs)


def save_poetry_results(output_path: Path, poem_text: str, analysis_results: Dict[str, Any],
                        **kwargs) -> None:
    """Convenience function for saving poetry results."""
    formatter = PoetryOutputFormatter()
    formatter.save_results_with_formatting(output_path, poem_text, analysis_results, **kwargs)