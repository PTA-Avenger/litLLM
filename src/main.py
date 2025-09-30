"""Main entry point for the Stylistic Poetry LLM Framework."""

import sys
from pathlib import Path
import click

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import config_manager
from utils.logging import initialize_logging, get_logger
from utils.exceptions import PoetryLLMError, ConfigurationError


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Path to configuration file')
@click.option('--log-level', '-l', default='INFO', help='Logging level')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, config, log_level, verbose):
    """Stylistic Poetry LLM Framework - Generate poetry in the style of renowned poets."""
    
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Set log level based on verbose flag
    if verbose:
        log_level = 'DEBUG'
    
    # Initialize logging
    initialize_logging(log_level)
    logger = get_logger('main')
    
    try:
        # Load configuration
        if config:
            config_path = Path(config)
            ctx.obj['config'] = config_manager.load_config(config_path)
        else:
            ctx.obj['config'] = config_manager.get_config()
        
        logger.info("Stylistic Poetry LLM Framework initialized")
        logger.debug(f"Configuration loaded from: {config_manager.config_path}")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise ConfigurationError(f"Initialization failed: {e}")


@cli.command()
@click.pass_context
def info(ctx):
    """Display system information and configuration."""
    logger = get_logger('info')
    config = ctx.obj['config']
    
    click.echo("=== Stylistic Poetry LLM Framework ===")
    click.echo(f"Version: 0.1.0")
    click.echo(f"Configuration file: {config_manager.config_path}")
    click.echo()
    
    click.echo("=== Configuration ===")
    click.echo(f"Base model: {config.model.base_model_name}")
    click.echo(f"Device: {config.device}")
    click.echo(f"Log level: {config.log_level}")
    click.echo(f"Data directory: {config.data.data_dir}")
    click.echo(f"Output directory: {config.data.output_dir}")
    click.echo()
    
    click.echo("=== Directories ===")
    directories = ['src', 'tests', 'data', 'models', 'config', 'logs']
    for directory in directories:
        path = Path(directory)
        status = "✓" if path.exists() else "✗"
        click.echo(f"{status} {directory}/")
    
    logger.info("System information displayed")


@cli.command()
@click.option('--check-deps', is_flag=True, help='Check if all dependencies are installed')
@click.pass_context
def validate(ctx, check_deps):
    """Validate system setup and configuration."""
    logger = get_logger('validate')
    config = ctx.obj['config']
    
    click.echo("=== Validating System Setup ===")
    
    # Check directory structure
    required_dirs = ['src', 'tests', 'data', 'models', 'config']
    missing_dirs = []
    
    for directory in required_dirs:
        path = Path(directory)
        if path.exists():
            click.echo(f"✓ Directory {directory}/ exists")
        else:
            click.echo(f"✗ Directory {directory}/ missing")
            missing_dirs.append(directory)
    
    # Check configuration
    try:
        # Validate configuration by accessing key properties
        _ = config.model.base_model_name
        _ = config.training.learning_rate
        _ = config.data.data_dir
        click.echo("✓ Configuration is valid")
    except Exception as e:
        click.echo(f"✗ Configuration validation failed: {e}")
        return
    
    # Check dependencies if requested
    if check_deps:
        click.echo("\n=== Checking Dependencies ===")
        required_packages = [
            'torch', 'transformers', 'nltk', 'pyphen', 'pronouncing',
            'numpy', 'pandas', 'scikit-learn', 'pydantic', 'pyyaml', 'click'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
                click.echo(f"✓ {package}")
            except ImportError:
                click.echo(f"✗ {package} (not installed)")
                missing_packages.append(package)
        
        if missing_packages:
            click.echo(f"\nMissing packages: {', '.join(missing_packages)}")
            click.echo("Run: pip install -r requirements.txt")
    
    if missing_dirs:
        click.echo(f"\nMissing directories: {', '.join(missing_dirs)}")
        click.echo("These will be created automatically when needed.")
    
    if not missing_dirs and (not check_deps or not missing_packages):
        click.echo("\n✓ System validation passed!")
        logger.info("System validation completed successfully")
    else:
        logger.warning("System validation found issues")


def main():
    """Main entry point."""
    try:
        cli()
    except PoetryLLMError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled by user", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()