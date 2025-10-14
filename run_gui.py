#!/usr/bin/env python3
"""
Launcher script for the MLR Working Party GUI Application.
This script sets up the environment and launches the GUI.
"""

import sys
import os

# Add the 02_code directory to the path
project_root = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.join(project_root, '02_code')
sys.path.insert(0, code_dir)

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = {
        'PyQt6': 'PyQt6',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'scipy': 'scipy'
    }
    
    missing_packages = []
    
    for package, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("ERROR: Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install them using:")
        print("  pip install -r requirements.txt")
        print("\nOr install individually:")
        print(f"  pip install {' '.join(missing_packages)}")
        return False
    
    return True


def main():
    """Main entry point for the launcher."""
    print("MLR Working Party GUI Application")
    print("=" * 50)
    
    # Check dependencies
    print("\nChecking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    print("All dependencies found!")
    
    # Import and run the GUI
    print("\nLaunching GUI application...")
    try:
        from mlr_gui_app import main as gui_main
        gui_main()
    except Exception as e:
        print(f"\nERROR: Failed to launch GUI application:")
        print(f"  {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
