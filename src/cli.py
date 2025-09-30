"""
Command-line interface for poetry generation.

This module provides a CLI for generating poetry with various stylistic controls
and parameters using the Stylistic Poetry LLM Framework.
"""

import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any
import click
from datetime import datetime
import textwrap

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from stylometric.model_interface import (
    create_poetry_model, 
    PoetryGenerationRequest, 
    GenerationConfig
)
from stylometric.evaluation_metrics import QuantitativeEvaluator
from stylometric.evaluation_comparison import EvaluationComparator, generate_comparison_report
from stylometric.poet_profile import PoetProfile
from stylometric.output_formatter import PoetryOutputFormatter, format_poetry_output, save_poetry_results
from stylometric.performance_monitor import create_performance_monitor
from config.settings import config_manager
from utils.logging import initialize_logging, get_logger
from utils.error_integration import (
    get_system_error_handler, initialize_error_handling, 
    handle_with_recovery, safe_execute
)
from utils.exceptions import create_user_friendly_error_message


# Available poet styles
AVAILABLE_POETS = {
    'emily_dickinson': 'Emily Dickinson - dashes, slant rhyme, contemplative themes',
    'walt_whitman': 'Walt Whitman - free verse, cataloging, expansive themes',
    'edgar_allan_poe': 'Edgar Allan Poe - dark themes, consistent rhyme, haunting atmosphere',
    'general': 'General poetry style'
}

# Available poetic forms
AVAILABLE_FORMS = [
    'sonnet', 'haiku', 'free_verse', 'ballad', 'limerick', 'villanelle', 'ghazal'
]


def validate_poet_style(ctx, param, value):
    """Validate poet style parameter."""
    if value is None:
        return value
    
    if value not in AVAILABLE_POETS:
        available = ', '.join(AVAILABLE_POETS.keys())
        raise click.BadParameter(f"Invalid poet style. Available options: {available}")
    
    return value


def validate_form(ctx, param, value):
    """Validate poetic form parameter."""
    if value is None:
        return value
    
    if value not in AVAILABLE_FORMS:
        available = ', '.join(AVAILABLE_FORMS)
        raise click.BadParameter(f"Invalid form. Available options: {available}")
    
    return value


def validate_temperature(ctx, param, value):
    """Validate temperature parameter."""
    if value is not None and (value < 0.0 or value > 2.0):
        raise click.BadParameter("Temperature must be between 0.0 and 2.0")
    return value


def validate_top_p(ctx, param, value):
    """Validate top_p parameter."""
    if value is not None and (value < 0.0 or value > 1.0):
        raise click.BadParameter("top_p must be between 0.0 and 1.0")
    return value


def validate_max_length(ctx, param, value):
    """Validate max_length parameter."""
    if value is not None and (value < 10 or value > 1000):
        raise click.BadParameter("max_length must be between 10 and 1000")
    return value


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Path to configuration file')
@click.option('--log-level', '-l', default='INFO', help='Logging level')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--no-recovery', is_flag=True, help='Disable error recovery (fail fast)')
@click.pass_context
def cli(ctx, config, log_level, verbose, no_recovery):
    """Stylistic Poetry LLM Framework - Generate poetry in the style of renowned poets."""
    
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Set log level based on verbose flag
    if verbose:
        log_level = 'DEBUG'
    
    # Initialize logging and error handling
    initialize_logging(log_level)
    initialize_error_handling(log_errors=True, enable_recovery=not no_recovery)
    logger = get_logger('cli')
    
    def load_configuration():
        """Load configuration with error handling."""
        if config:
            config_path = Path(config)
            return config_manager.load_config(config_path)
        else:
            return config_manager.get_config()
    
    # Load configuration with error handling
    config_result = safe_execute(
        "configuration loading",
        load_configuration,
        fallback_value=None,
        reraise=False
    )
    
    if config_result is None:
        click.echo("Warning: Using default configuration due to loading error", err=True)
        ctx.obj['config'] = config_manager.get_default_config()
    else:
        ctx.obj['config'] = config_result
    
    # Store error handling preferences
    ctx.obj['enable_recovery'] = not no_recovery
    
    logger.info("CLI initialized successfully")


@cli.command()
@click.argument('prompt', required=True)
@click.option('--poet', '-p', 
              callback=validate_poet_style,
              help=f'Poet style to emulate. Options: {", ".join(AVAILABLE_POETS.keys())}')
@click.option('--form', '-f',
              callback=validate_form,
              help=f'Poetic form. Options: {", ".join(AVAILABLE_FORMS)}')
@click.option('--theme', '-t', help='Thematic focus for the poem')
@click.option('--model', '-m', default='gpt2', help='Model to use for generation (default: gpt2)')
@click.option('--temperature', callback=validate_temperature, type=float, default=0.8,
              help='Sampling temperature (0.0-2.0, default: 0.8)')
@click.option('--top-p', callback=validate_top_p, type=float, default=0.9,
              help='Top-p sampling parameter (0.0-1.0, default: 0.9)')
@click.option('--top-k', type=int, default=50,
              help='Top-k sampling parameter (default: 50)')
@click.option('--max-length', callback=validate_max_length, type=int, default=200,
              help='Maximum length of generated text (10-1000, default: 200)')
@click.option('--output', '-o', type=click.Path(), help='Save output to file')
@click.option('--analyze', is_flag=True, help='Include stylistic analysis of generated poetry')
@click.option('--compare', type=click.Path(exists=True), help='Compare generated poetry with target text file')
@click.option('--format', 'output_format', type=click.Choice(['simple', 'enhanced', 'detailed']), 
              default='enhanced', help='Output formatting style (default: enhanced)')
@click.option('--no-sample', is_flag=True, help='Disable sampling (use greedy decoding)')
@click.option('--monitor-performance', is_flag=True, help='Enable performance monitoring and reporting')
@click.pass_context
def generate(ctx, prompt, poet, form, theme, model, temperature, top_p, top_k, 
             max_length, output, analyze, compare, output_format, no_sample, monitor_performance):
    """Generate poetry based on a prompt and stylistic parameters.
    
    PROMPT: The text prompt to inspire the poem generation.
    
    Examples:
    
    \b
    # Generate a poem about nature in Emily Dickinson's style
    poetry-cli generate "the quiet forest" --poet emily_dickinson
    
    \b
    # Generate a sonnet about love with analysis
    poetry-cli generate "eternal love" --form sonnet --analyze
    
    \b
    # Generate with custom parameters and save to file
    poetry-cli generate "city lights" --temperature 1.2 --output poem.txt
    
    \b
    # Generate with performance monitoring
    poetry-cli generate "ocean waves" --monitor-performance --analyze
    """
    logger = get_logger('generate')
    config = ctx.obj['config']
    
    try:
        # Validate prompt
        if not prompt.strip():
            raise click.BadParameter("Prompt cannot be empty")
        
        # Display generation parameters
        click.echo("=== Poetry Generation ===")
        click.echo(f"Prompt: {prompt}")
        if poet:
            click.echo(f"Poet style: {poet} ({AVAILABLE_POETS[poet]})")
        if form:
            click.echo(f"Form: {form}")
        if theme:
            click.echo(f"Theme: {theme}")
        click.echo(f"Model: {model}")
        click.echo(f"Temperature: {temperature}")
        click.echo()
        
        # Create generation configuration
        gen_config = GenerationConfig(
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=not no_sample
        )
        
        # Create generation request
        request = PoetryGenerationRequest(
            prompt=prompt,
            poet_style=poet,
            theme=theme,
            form=form,
            generation_config=gen_config
        )
        
        # Initialize model
        click.echo("Loading model...")
        poetry_model = create_poetry_model(model_type="gpt", model_name=model)
        
        if not poetry_model.load_model():
            raise click.ClickException(f"Failed to load model: {model}")
        
        logger.info(f"Model {model} loaded successfully")
        
        try:
            # Initialize performance monitor if requested
            performance_monitor = None
            performance_metrics = None
            
            if monitor_performance:
                performance_monitor = create_performance_monitor(enable_gpu_monitoring=True)
                click.echo("Performance monitoring enabled")
            
            # Generate poetry with optional performance monitoring
            click.echo("Generating poetry...")
            
            if performance_monitor:
                with performance_monitor.monitor_operation(
                    "poetry_generation", 
                    model, 
                    gen_config.__dict__
                ) as metrics:
                    response = poetry_model.generate_poetry(request)
                    performance_metrics = metrics
            else:
                response = poetry_model.generate_poetry(request)
            
            if not response.success:
                raise click.ClickException(f"Generation failed: {response.error_message}")
            
            # Perform analysis if requested or if enhanced formatting is used
            analysis_results = None
            comparison_result = None
            
            if analyze or output_format in ['enhanced', 'detailed']:
                try:
                    evaluator = QuantitativeEvaluator()
                    analysis_results = evaluator.calculate_lexical_metrics(response.generated_text)
                    
                    # Add additional metrics for enhanced display
                    if output_format == 'detailed':
                        # Get more comprehensive analysis
                        full_eval = evaluator.evaluate_poetry(response.generated_text)
                        analysis_results.update({
                            'structural_metrics': full_eval.structural_metrics,
                            'readability_metrics': full_eval.readability_metrics
                        })
                    
                except Exception as e:
                    logger.warning(f"Analysis failed: {e}")
                    analysis_results = {'error': str(e)}
            
            # Perform comparison if target file provided
            if compare:
                try:
                    with open(compare, 'r', encoding='utf-8') as f:
                        target_text = f.read().strip()
                    
                    comparator = EvaluationComparator()
                    comparison_result = comparator.compare_poetry_side_by_side(
                        response.generated_text, target_text
                    )
                    
                    click.echo(f"\nComparing with target text from: {compare}")
                    
                except Exception as e:
                    logger.warning(f"Comparison failed: {e}")
                    click.echo(f"Comparison failed: {e}")
            
            # Display performance metrics if monitoring was enabled
            if performance_metrics:
                click.echo("\n" + "="*50)
                click.echo("PERFORMANCE METRICS")
                click.echo("="*50)
                click.echo(f"Generation latency: {performance_metrics.latency_ms:.2f}ms")
                click.echo(f"Memory usage: {performance_metrics.memory_usage_mb:.1f}MB")
                click.echo(f"Peak memory: {performance_metrics.peak_memory_mb:.1f}MB")
                click.echo(f"CPU usage: {performance_metrics.cpu_usage_percent:.1f}%")
                if performance_metrics.gpu_memory_mb:
                    click.echo(f"GPU memory: {performance_metrics.gpu_memory_mb:.1f}MB")
                if performance_metrics.gpu_utilization_percent:
                    click.echo(f"GPU utilization: {performance_metrics.gpu_utilization_percent:.1f}%")
                click.echo("="*50)
            
            # Display results based on format choice
            if output_format == 'simple':
                # Simple format - just the poem
                click.echo("\n" + "="*50)
                click.echo("GENERATED POEM")
                click.echo("="*50)
                click.echo(response.generated_text)
                click.echo("="*50)
                
                if analysis_results and 'error' not in analysis_results:
                    click.echo(f"\nBasic Analysis:")
                    click.echo(f"Words: {analysis_results.get('word_count', 'N/A')}")
                    click.echo(f"Lines: {analysis_results.get('line_count', 'N/A')}")
                    if 'ttr' in analysis_results:
                        click.echo(f"TTR: {analysis_results['ttr']:.3f}")
            
            else:
                # Enhanced or detailed format
                formatter = PoetryOutputFormatter()
                
                # Add performance metrics to generation metadata if available
                enhanced_metadata = response.generation_metadata or {}
                if performance_metrics:
                    enhanced_metadata['performance_metrics'] = {
                        'latency_ms': performance_metrics.latency_ms,
                        'memory_usage_mb': performance_metrics.memory_usage_mb,
                        'peak_memory_mb': performance_metrics.peak_memory_mb,
                        'cpu_usage_percent': performance_metrics.cpu_usage_percent,
                        'gpu_memory_mb': performance_metrics.gpu_memory_mb,
                        'gpu_utilization_percent': performance_metrics.gpu_utilization_percent,
                        'timestamp': performance_metrics.timestamp
                    }
                
                formatted_output = formatter.create_comprehensive_output(
                    poem_text=response.generated_text,
                    analysis_results=analysis_results or {},
                    generation_metadata=enhanced_metadata,
                    comparison_result=comparison_result,
                    poet_style=poet,
                    prompt=prompt
                )
                
                click.echo("\n" + formatted_output)
            
            # Save output if requested
            if output:
                output_path = Path(output)
                
                try:
                    formatter = PoetryOutputFormatter()
                    
                    # Include performance metrics in saved output if available
                    save_metadata = response.generation_metadata or {}
                    if performance_metrics:
                        save_metadata['performance_metrics'] = {
                            'latency_ms': performance_metrics.latency_ms,
                            'memory_usage_mb': performance_metrics.memory_usage_mb,
                            'peak_memory_mb': performance_metrics.peak_memory_mb,
                            'cpu_usage_percent': performance_metrics.cpu_usage_percent,
                            'gpu_memory_mb': performance_metrics.gpu_memory_mb,
                            'gpu_utilization_percent': performance_metrics.gpu_utilization_percent,
                            'timestamp': performance_metrics.timestamp
                        }
                    
                    formatter.save_results_with_formatting(
                        output_path=output_path,
                        poem_text=response.generated_text,
                        analysis_results=analysis_results or {},
                        generation_config=gen_config.__dict__,
                        generation_metadata=save_metadata,
                        comparison_result=comparison_result,
                        prompt=prompt,
                        poet_style=poet,
                        theme=theme,
                        form=form
                    )
                    
                    click.echo(f"\nOutput saved to: {output_path}")
                    logger.info(f"Output saved to {output_path}")
                    
                except Exception as e:
                    logger.error(f"Failed to save output: {e}")
                    click.echo(f"Failed to save output: {e}", err=True)
            
            logger.info("Poetry generation completed successfully")
            
        finally:
            # Clean up model
            poetry_model.unload_model()
            logger.info("Model unloaded")
    
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise click.ClickException(str(e))


@cli.command()
def list_poets():
    """List available poet styles."""
    click.echo("=== Available Poet Styles ===")
    for key, description in AVAILABLE_POETS.items():
        click.echo(f"  {key:<20} - {description}")


@cli.command()
def list_forms():
    """List available poetic forms."""
    click.echo("=== Available Poetic Forms ===")
    for form in AVAILABLE_FORMS:
        click.echo(f"  {form}")


@cli.command()
@click.option('--poet', '-p', callback=validate_poet_style,
              help='Show example for specific poet style')
def examples(poet):
    """Show usage examples."""
    click.echo("=== Usage Examples ===")
    click.echo()
    
    if poet:
        # Show examples for specific poet
        poet_examples = {
            'emily_dickinson': [
                'poetry-cli generate "the quiet garden" --poet emily_dickinson',
                'poetry-cli generate "death and immortality" --poet emily_dickinson --analyze'
            ],
            'walt_whitman': [
                'poetry-cli generate "America the beautiful" --poet walt_whitman',
                'poetry-cli generate "the open road" --poet walt_whitman --form free_verse'
            ],
            'edgar_allan_poe': [
                'poetry-cli generate "midnight ravens" --poet edgar_allan_poe',
                'poetry-cli generate "lost love" --poet edgar_allan_poe --theme melancholy'
            ],
            'general': [
                'poetry-cli generate "spring morning" --poet general',
                'poetry-cli generate "friendship" --poet general --form sonnet'
            ]
        }
        
        click.echo(f"Examples for {poet} style:")
        for example in poet_examples.get(poet, []):
            click.echo(f"  {example}")
    else:
        # Show general examples
        examples_list = [
            "# Basic generation",
            "poetry-cli generate \"the ocean waves\"",
            "",
            "# With poet style",
            "poetry-cli generate \"nature's beauty\" --poet emily_dickinson",
            "",
            "# With form specification",
            "poetry-cli generate \"eternal love\" --form sonnet",
            "",
            "# With analysis and output file",
            "poetry-cli generate \"city lights\" --analyze --output poem.txt",
            "",
            "# Custom generation parameters",
            "poetry-cli generate \"dreams\" --temperature 1.2 --max-length 150",
            "",
            "# Multiple options combined",
            "poetry-cli generate \"autumn leaves\" --poet walt_whitman --theme nostalgia --analyze"
        ]
        
        for example in examples_list:
            click.echo(example)


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--compare', type=click.Path(exists=True), help='Compare with another poem file')
@click.option('--output', '-o', type=click.Path(), help='Save analysis to file')
@click.option('--format', 'output_format', type=click.Choice(['simple', 'enhanced', 'detailed']), 
              default='enhanced', help='Output formatting style (default: enhanced)')
@click.pass_context
def analyze_poem(ctx, input_file, compare, output, output_format):
    """Analyze an existing poem file for stylistic features.
    
    INPUT_FILE: Path to the poem text file to analyze.
    
    Examples:
    
    \b
    # Analyze a poem file
    poetry-cli analyze-poem poem.txt
    
    \b
    # Compare two poems
    poetry-cli analyze-poem poem1.txt --compare poem2.txt
    
    \b
    # Save detailed analysis to file
    poetry-cli analyze-poem poem.txt --format detailed --output analysis.json
    """
    logger = get_logger('analyze_poem')
    
    try:
        # Read input poem
        with open(input_file, 'r', encoding='utf-8') as f:
            poem_text = f.read().strip()
        
        if not poem_text:
            raise click.BadParameter("Input file is empty")
        
        click.echo(f"Analyzing poem from: {input_file}")
        
        # Perform comprehensive analysis
        evaluator = QuantitativeEvaluator()
        analysis_results = evaluator.calculate_lexical_metrics(poem_text)
        
        # Get full evaluation for detailed analysis
        if output_format == 'detailed':
            full_eval = evaluator.evaluate_poetry(poem_text)
            analysis_results.update({
                'structural_metrics': full_eval.structural_metrics,
                'readability_metrics': full_eval.readability_metrics
            })
        
        # Perform comparison if requested
        comparison_result = None
        if compare:
            try:
                with open(compare, 'r', encoding='utf-8') as f:
                    target_text = f.read().strip()
                
                comparator = EvaluationComparator()
                comparison_result = comparator.compare_poetry_side_by_side(poem_text, target_text)
                
                click.echo(f"Comparing with: {compare}")
                
            except Exception as e:
                logger.warning(f"Comparison failed: {e}")
                click.echo(f"Comparison failed: {e}")
        
        # Display results
        if output_format == 'simple':
            click.echo("\n=== Analysis Results ===")
            click.echo(f"Word count: {analysis_results.get('word_count', 'N/A')}")
            click.echo(f"Line count: {analysis_results.get('line_count', 'N/A')}")
            if 'ttr' in analysis_results:
                click.echo(f"Type-Token Ratio: {analysis_results['ttr']:.3f}")
            if 'lexical_density' in analysis_results:
                click.echo(f"Lexical density: {analysis_results['lexical_density']:.3f}")
        else:
            formatter = PoetryOutputFormatter()
            
            formatted_output = formatter.create_comprehensive_output(
                poem_text=poem_text,
                analysis_results=analysis_results,
                comparison_result=comparison_result,
                prompt=f"Analysis of {Path(input_file).name}"
            )
            
            click.echo("\n" + formatted_output)
        
        # Save output if requested
        if output:
            output_path = Path(output)
            
            try:
                formatter = PoetryOutputFormatter()
                formatter.save_results_with_formatting(
                    output_path=output_path,
                    poem_text=poem_text,
                    analysis_results=analysis_results,
                    generation_config={'analysis_mode': True},
                    generation_metadata={'source_file': str(input_file)},
                    comparison_result=comparison_result,
                    prompt=f"Analysis of {Path(input_file).name}"
                )
                
                click.echo(f"\nAnalysis saved to: {output_path}")
                logger.info(f"Analysis saved to {output_path}")
                
            except Exception as e:
                logger.error(f"Failed to save analysis: {e}")
                click.echo(f"Failed to save analysis: {e}", err=True)
        
        logger.info("Poem analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise click.ClickException(str(e))


@cli.command()
@click.pass_context
def info(ctx):
    """Display system information and available options."""
    config = ctx.obj['config']
    
    click.echo("=== Stylistic Poetry LLM Framework ===")
    click.echo("Version: 0.1.0")
    click.echo()
    
    click.echo("=== Available Poet Styles ===")
    for key, description in AVAILABLE_POETS.items():
        click.echo(f"  {key:<20} - {description}")
    click.echo()
    
    click.echo("=== Available Forms ===")
    for form in AVAILABLE_FORMS:
        click.echo(f"  {form}")
    click.echo()
    
    click.echo("=== Configuration ===")
    click.echo(f"Default model: {config.model.base_model_name}")
    click.echo(f"Device: {config.device}")
    click.echo(f"Data directory: {config.data.data_dir}")


def main():
    """Main entry point for the CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled by user", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()