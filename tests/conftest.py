# imagecodecs/tests/conftest.py


def pytest_report_header(config):
    import sys

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
