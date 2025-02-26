"""
Install required packages for the evaluation script
"""
import os
import subprocess
import sys
from pathlib import Path

def install(package):
    """
    Install a pip python package
    
    Args:
        package (str): Package name with version
    """
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install required packages
install("numpy>=1.19.0")
install("scikit-learn>=0.24.0")

from .main import evaluate