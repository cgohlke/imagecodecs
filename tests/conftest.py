# -*- coding: utf-8 -*-
# imagecodecs/conftest.py


def pytest_report_header(config):
    import os

    try:
        if 'imagecodecs_lite' in os.path.abspath(__file__):
            import imagecodecs_lite

            return 'packagedir: %s\nversion: %s' % (
                imagecodecs_lite.__path__[0],
                imagecodecs_lite.version(),
            )
        else:
            import imagecodecs
            from imagecodecs import imagecodecs as imagecodecs_py

            return 'packagedir: %s\nversion: %s\nversions Py: %s' % (
                imagecodecs.__path__[0],
                imagecodecs.version(),
                imagecodecs_py.version(),
            )
    except Exception as e:
        return 'pytest_report_header failed: %s' % str(e)
