# -*- coding: utf-8 -*-
# imagecodecs/conftest.py

def pytest_report_header(config):
    try:
        import imagecodecs
        return 'versions: ' + imagecodecs.version(str)
    except Exception:
        pass
