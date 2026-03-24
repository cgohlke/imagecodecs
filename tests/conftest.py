# imagecodecs/tests/conftest.py

"""Pytest configuration."""

from __future__ import annotations

import glob
import os
import pathlib
import sys
from typing import Any

DATA_PATH = pathlib.Path(os.path.dirname(__file__)) / 'data'

if os.environ.get('VSCODE_CWD'):
    # work around pytest not using PYTHONPATH in VSCode
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    )


def datafiles(pathname: str, base: str | None = None) -> Any:
    """Return path to data file(s)."""
    if base is None:
        base = str(DATA_PATH)
    path = os.path.join(base, *pathname.split('/'))
    if any(i in path for i in '*?'):
        return glob.glob(path)
    return path


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
