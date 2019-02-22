# -*- coding: utf-8 -*-
# imagecodecs/setup.py

"""Imagecodecs package setuptools script."""

import sys
import re

import numpy

from setuptools import setup, Extension
from Cython.Distutils import build_ext

buildnumber = ''  # 'post0'

with open('imagecodecs/_imagecodecs.pyx') as fh:
    code = fh.read()

version = re.search(r"__version__ = '(.*?)'", code).groups()[0]

version += ('.' + buildnumber) if buildnumber else ''

description = re.search(r'"""(.*)\.(?:\r\n|\r|\n)', code).groups()[0]

readme = re.search(r'(?:\r\n|\r|\n){2}"""(.*)"""(?:\r\n|\r|\n){2}__version__',
                   code, re.MULTILINE | re.DOTALL).groups()[0]

readme = '\n'.join([description, '=' * len(description)]
                   + readme.splitlines()[1:])

license = re.search(r'(# Copyright.*?(?:\r\n|\r|\n))(?:\r\n|\r|\n)+""', code,
                    re.MULTILINE | re.DOTALL).groups()[0]

license = license.replace('# ', '').replace('#', '')

if 'sdist' in sys.argv:
    with open('LICENSE', 'w') as fh:
        fh.write(license)
    with open('README.rst', 'w') as fh:
        fh.write(readme)
    numpy_required = '1.11.3'

else:
    numpy_required = numpy.__version__


sources = [
    'imagecodecs/opj_color.c',
    'imagecodecs/jpeg_sof3.cpp',
    'imagecodecs/_imagecodecs.pyx',
]

include_dirs = [
    numpy.get_include(),
    'imagecodecs',
]

try:
    # running in Windows development environment
    import _inclib  # noqa
    libraries = [
        'zlib', 'lz4', 'webp', 'png', 'jxrlib', 'jpeg', 'lzf', 'libbz2',
        'libblosc', 'snappy', 'zstd_static', 'lzma-static', 'openjp2',
        'lcms2']
    define_macros = [('WIN32', 1), ('LZMA_API_STATIC', 1),
                     ('OPJ_STATIC', 1), ('OPJ_HAVE_LIBLCMS2', 1),
                     ('CHARLS_STATIC', 1)]
    libraries_jpeg12 = ['jpeg12']
    if sys.version_info < (3, 5):
        # clarls-2.0 not compatible with msvc 9 or 10
        libraries_jpegls = []
    else:
        libraries_jpegls = ['charls']
    libraries_zfp = ['zfp']
    openmp_args = ['/openmp']

except ImportError:
    # this works with most recent Debian
    libraries = ['jpeg', 'lz4', 'zstd', 'lzma', 'bz2', 'png', 'webp', 'blosc',
                 'openjp2', 'jxrglue', 'jpegxr', 'lcms2', 'z']
    include_dirs.extend(
        ['/usr/include/jxrlib',
         '/usr/include/openjpeg-2.1',
         '/usr/include/openjpeg-2.2',
         '/usr/include/openjpeg-2.3'])
    define_macros = [('OPJ_HAVE_LIBLCMS2', 1)]
    if sys.platform == 'win32':
        define_macros.append(('WIN32', 1), ('CHARLS_STATIC', 1))
    else:
        libraries.append('m')
    libraries_jpeg12 = []  # 'jpeg12'
    libraries_jpegls = []  # 'charls'
    libraries_zfp = []  # 'zfp'
    openmp_args = ['-fopenmp']


if 'lzf' not in libraries and 'liblzf' not in libraries:
    # use liblzf sources from sdist
    sources.extend(['liblzf-3.6/lzf_c.c', 'liblzf-3.6/lzf_d.c'])
    include_dirs.append('liblzf-3.6')


ext_modules = [
    Extension(
        'imagecodecs._imagecodecs_lite',
        ['imagecodecs/imagecodecs.c', 'imagecodecs/_imagecodecs_lite.pyx'],
        include_dirs=[numpy.get_include(), 'imagecodecs'],
        libraries=[] if sys.platform == 'win32' else ['m'],
    ),
    Extension(
        'imagecodecs._imagecodecs',
        sources,
        include_dirs=include_dirs,
        libraries=libraries,
        define_macros=define_macros,
    )
]

if libraries_jpeg12:
    ext_modules += [
        Extension(
            'imagecodecs._jpeg12',
            ['imagecodecs/_jpeg12.pyx'],
            include_dirs=[numpy.get_include(), 'imagecodecs'],
            libraries=libraries_jpeg12,
            define_macros=[('BITS_IN_JSAMPLE', 12)],
        )
    ]

if libraries_jpegls:
    ext_modules += [
        Extension(
            'imagecodecs._jpegls',
            ['imagecodecs/_jpegls.pyx'],
            include_dirs=[numpy.get_include(), 'imagecodecs'],
            libraries=libraries_jpegls,
            define_macros=define_macros,
        )
    ]

if libraries_zfp:
    ext_modules += [
        Extension(
            'imagecodecs._zfp',
            ['imagecodecs/_zfp.pyx'],
            include_dirs=[numpy.get_include(), 'imagecodecs'],
            libraries=libraries_zfp,
            define_macros=define_macros,
            extra_compile_args = openmp_args
        )
    ]

setup_args = dict(
    name='imagecodecs',
    version=version,
    description=description,
    long_description=readme,
    author='Christoph Gohlke',
    author_email='cgohlke@uci.edu',
    url='https://www.lfd.uci.edu/~gohlke/',
    python_requires='>=2.7',
    install_requires=['numpy>=%s' % numpy_required],
    extras_require={'all': ['matplotlib>=2.2', 'tifffile>=2019.1.1']},
    tests_require=['pytest', 'tifffile', 'blosc', 'zstd', 'lz4',
                   'python-lzf', 'scikit-image'],
    packages=['imagecodecs'],
    package_data={'imagecodecs': ['licenses/*']},
    entry_points={
        'console_scripts': ['imagecodecs=imagecodecs.__main__:main']},
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    license='BSD',
    zip_safe=False,
    platforms=['any'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: BSD License',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: C',
        'Programming Language :: Cython',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        ],
    )

if '--universal' in sys.argv:
    del setup_args['ext_modules']
    del setup_args['cmdclass']
    del setup_args['package_data']

setup(**setup_args)
