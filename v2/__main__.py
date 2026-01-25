#!/usr/bin/env python3
"""
Allow running Hexify v2 as a module.

Usage:
    python -m v2 input.png -o output.png
    python -m v2 --help
"""

from .cli import main

if __name__ == "__main__":
    import sys
    sys.exit(main())
