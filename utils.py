import sys
import subprocess
import warnings
from typing import List

# Suppress warnings
warnings.filterwarnings("ignore")

def install_missing_packages(packages: List[str]):
    """
    Checks if required packages are installed and installs them if missing.

    Args:
        packages (List[str]): A list of package names to check and install.
    """
    for package in packages:
        try:
            # Special handling for scikit-learn as its import name is 'sklearn'
            if package == 'scikit-learn':
                __import__('sklearn')
            else:
                __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])