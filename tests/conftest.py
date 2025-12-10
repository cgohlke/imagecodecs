# imagecodecs/tests/conftest.py

"""Pytest configuration."""

import os
import sys

if os.environ.get('VSCODE_CWD'):
    # work around pytest not using PYTHONPATH in VSCode
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    )


def pytest_report_header(config: object) -> str:
    """Return pytest report header."""
    try:
        import imagecodecs
        from imagecodecs import _imagecodecs

        return (
            f'Python {sys.version.splitlines()[0]}\n'
            f'packagedir: {imagecodecs.__path__[0]}\n'
            f'version: {imagecodecs.version()}\n'
            f'dependencies: {_imagecodecs.version()}'
        )
    except Exception as exc:
        return f'pytest_report_header failed: {exc!s}'
