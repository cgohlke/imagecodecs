# -*- coding: utf-8 -*-
# imagecodecs/conftest.py

def pytest_report_header(config):
    try:
        import imagecodecs
        from imagecodecs import imagecodecs as imagecodecs_py
        return 'versions C: %s\nversions Py: %s' % (
            imagecodecs.version(), imagecodecs_py.version())
    except Exception:
        pass
