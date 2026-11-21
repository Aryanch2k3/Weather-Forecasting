"""Deprecated compatibility entry point. Fake grid generation is disabled.

Use scripts/prepare_data.py to convert original real CSV source records into
packed 8-column weather rows. This program intentionally never generates data.
"""
import sys

if __name__ == '__main__':
    sys.exit('Synthetic grid generation has been removed. Use the included Delhi Open-Meteo CSV in data/ '
             'and run: .\\.venv\\Scripts\\python.exe scripts\\prepare_data.py --dataset delhi')
