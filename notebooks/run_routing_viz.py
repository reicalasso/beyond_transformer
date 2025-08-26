#!/usr/bin/env python3
"""
Run script for routing visualization notebook
"""

import subprocess
import sys
import os

def main():
    """Launch the routing visualization notebook"""
    notebook_path = "notebooks/routing_viz.ipynb"
    
    # Check if notebook exists
    if not os.path.exists(notebook_path):
        print(f"Error: Notebook not found at {notebook_path}")
        return 1
    
    # Try to launch Jupyter notebook
    try:
        print(f"Launching {notebook_path}...")
        subprocess.run(["jupyter", "notebook", notebook_path], check=True)
        return 0
    except FileNotFoundError:
        print("Error: Jupyter notebook not found. Please install it with:")
        print("  pip install jupyter")
        return 1
    except subprocess.CalledProcessError as e:
        print(f"Error launching notebook: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())