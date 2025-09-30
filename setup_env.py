#!/usr/bin/env python3
"""Setup script to create virtual environment and install dependencies."""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return result
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"Command: {command}")
        print(f"Error: {e.stderr}")
        return None


def main():
    """Main setup function."""
    print("=== Stylistic Poetry LLM Framework Setup ===")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("✗ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Create virtual environment
    venv_path = Path("venv")
    if not venv_path.exists():
        result = run_command(f"{sys.executable} -m venv venv", "Creating virtual environment")
        if not result:
            sys.exit(1)
    else:
        print("✓ Virtual environment already exists")
    
    # Determine activation script path
    if os.name == 'nt':  # Windows
        activate_script = venv_path / "Scripts" / "activate.bat"
        pip_path = venv_path / "Scripts" / "pip.exe"
    else:  # macOS/Linux
        activate_script = venv_path / "bin" / "activate"
        pip_path = venv_path / "bin" / "pip"
    
    # Install requirements
    if Path("requirements.txt").exists():
        result = run_command(f"{pip_path} install -r requirements.txt", "Installing requirements")
        if not result:
            sys.exit(1)
    else:
        print("✗ requirements.txt not found")
        sys.exit(1)
    
    # Install package in development mode
    result = run_command(f"{pip_path} install -e .", "Installing package in development mode")
    if not result:
        sys.exit(1)
    
    print("\n=== Setup Complete ===")
    print("To activate the virtual environment:")
    
    if os.name == 'nt':  # Windows
        print("  venv\\Scripts\\activate")
    else:  # macOS/Linux
        print("  source venv/bin/activate")
    
    print("\nTo validate the installation:")
    print("  python -m src.main validate --check-deps")
    
    print("\nTo view system information:")
    print("  python -m src.main info")


if __name__ == "__main__":
    main()