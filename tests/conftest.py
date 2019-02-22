# -*- coding: utf-8 -*-
# imagecodecs/conftest.py

def pytest_report_header(config):
    import os
    print(__file__)
    try:
        if 'imagecodecs_lite' in os.path.abspath(__file__):
            import imagecodecs_lite
            return 'versions: %s' % (imagecodecs_lite.version())
        else:
            import imagecodecs
            from imagecodecs import imagecodecs as imagecodecs_py
            return 'versions C: %s\nversions Py: %s' % (
                imagecodecs.version(), imagecodecs_py.version())
    except Exception as e:
        return 'pytest_report_header failed: %s' % str(e)
