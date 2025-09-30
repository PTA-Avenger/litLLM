#!/usr/bin/env python3
"""
Entry point script for the Poetry CLI.

This script provides a convenient way to run the poetry generation CLI
from the project root directory.
"""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

# Import and run the CLI
from cli import main

if __name__ == '__main__':
    main()