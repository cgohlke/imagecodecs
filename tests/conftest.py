# imagecodecs/tests/conftest.py
# type: ignore

import os
import sys

if os.environ.get('VSCODE_CWD'):
    # work around pytest not using PYTHONPATH in VSCode
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    )


def pytest_report_header(config, start_path):
    try:
        pyversion = f'Python {sys.version.splitlines()[0]}'
        import imagecodecs
        from imagecodecs import _imagecodecs

        return '{}\npackagedir: {}\nversion: {}\ndependencies: {}'.format(
            pyversion,
            imagecodecs.__path__[0],
            imagecodecs.version(),
            _imagecodecs.version(),
        )
    except Exception as exc:
        return f'pytest_report_header failed: {exc!s}'
