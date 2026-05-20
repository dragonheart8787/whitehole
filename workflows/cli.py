"""Backward-compatible CLI shim — delegates to package entry point.

Prefer: whitesearch ...  (after pip install -e .)
Or:     python -m whitesearch.cli ...
"""

from whitesearch.cli import main

if __name__ == "__main__":
    main()
