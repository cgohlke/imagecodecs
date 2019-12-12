# -*- coding: utf-8 -*-
# imagecodecs/conftest.py


def pytest_report_header(config):
    import sys
    import os

    try:
        pyversion = 'Python %s' % sys.version.splitlines()[0]
        if (
            'imagecodecs_lite' in os.getcwd() or
            os.path.exists(os.path.join(os.path.dirname(__file__), '..',
                                        'imagecodecs_lite'))
        ):
            import imagecodecs_lite

            return '%s\npackagedir: %s\nversion: %s' % (
                pyversion,
                imagecodecs_lite.__path__[0],
                imagecodecs_lite.version(),
            )
        else:
            import imagecodecs
            from imagecodecs import imagecodecs as imagecodecs_py

            return '%s\npackagedir: %s\nversion: %s\nversions Py: %s' % (
                pyversion,
                imagecodecs.__path__[0],
                imagecodecs.version(),
                imagecodecs_py.version(),
            )
    except Exception as e:
        return 'pytest_report_header failed: %s' % str(e)
