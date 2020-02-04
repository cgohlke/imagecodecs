# imagecodecs/setup.py

"""Imagecodecs package setuptools script."""

import sys
import os
import re

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext

buildnumber = ''  # e.g 'pre1' or 'post1'

base_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(base_dir, 'imagecodecs/imagecodecs.py')) as fh:
    code = fh.read().replace('\r', '')

version = re.search(r"__version__ = '(.*?)'", code).groups()[0]

version += ('.' + buildnumber) if buildnumber else ''

description = re.search(r'"""(.*)\.(?:\r\n|\r|\n)', code).groups()[0]

readme = re.search(r'(?:\r\n|\r|\n){2}"""(.*)"""(?:\r\n|\r|\n){2}__version__',
                   code, re.MULTILINE | re.DOTALL).groups()[0]

readme = '\n'.join([description, '=' * len(description)] +
                   readme.splitlines()[1:])

license = re.search(r'(# Copyright.*?(?:\r\n|\r|\n))(?:\r\n|\r|\n)+""', code,
                    re.MULTILINE | re.DOTALL).groups()[0]

license = license.replace('# ', '').replace('#', '')

if 'sdist' in sys.argv:
    # update README, LICENSE, and CHANGES files

    with open('README.rst', 'w') as fh:
        fh.write(readme)

    license = re.search(r'(# Copyright.*?(?:\r\n|\r|\n))(?:\r\n|\r|\n)+""',
                        code, re.MULTILINE | re.DOTALL).groups()[0]

    license = license.replace('# ', '').replace('#', '')

    with open('LICENSE', 'w') as fh:
        fh.write('BSD 3-Clause License\n\n')
        fh.write(license)

    revisions = re.search(r'(?:\r\n|\r|\n){2}(Revisions.*)   \.\.\.', readme,
                          re.MULTILINE | re.DOTALL).groups()[0].strip()

    with open('CHANGES.rst', 'r') as fh:
        old = fh.read()

    d = revisions.splitlines()[-1]
    old = old.split(d)[-1]
    with open('CHANGES.rst', 'w') as fh:
        fh.write(revisions.strip())
        fh.write(old)

###############################################################################

INCLUDE_DIRS = ['imagecodecs']
LIBRARY_DIRS = []
LIBRARIES = []
DEFINE_MACROS = []
EXTRA_COMPILE_ARGS = []

if sys.platform == 'win32':
    DEFINE_MACROS.append(('WIN32', 1))
    OPENMP_ARGS = ['/openmp']
else:
    LIBRARIES.append('m')
    OPENMP_ARGS = [] if os.environ.get('SKIP_OMP', False) else ['-fopenmp']

EXTENSIONS = {
    'shared': dict(),
    'imcd': dict(sources=['imagecodecs/imcd.c']),
    'aec': dict(libraries=['aec']),
    'bitshuffle': dict(
        sources=[
            'bitshuffle-0.3.5/bitshuffle_core.c',
            'bitshuffle-0.3.5/iochain.c',
            ],
        include_dirs=['bitshuffle-0.3.5']
        ),
    'blosc': dict(libraries=['blosc']),
    'brotli': dict(libraries=['brotlienc', 'brotlidec', 'brotlicommon']),
    'bz2': dict(libraries=['bz2']),
    'gif': dict(libraries=['gif']),
    'jpeg2k': dict(
        sources=['imagecodecs/opj_color.c'],
        libraries=['openjp2', 'lcms2'],
        define_macros=[('OPJ_HAVE_LIBLCMS2', 1)]
        ),
    'jpeg8': dict(libraries=['jpeg']),
    'jpeg12': dict(
        libraries=['jpeg12'],
        define_macros=[('BITS_IN_JSAMPLE', 12)]
        ),
    'jpegls': dict(libraries=['charls']),
    'jpegsof3': dict(sources=['imagecodecs/jpegsof3.cpp']),
    'jpegxl': dict(libraries=['brunslidec-c', 'brunslienc-c']),
    'jpegxr': dict(libraries=['jpegxr', 'jxrglue']),
    'lz4': dict(libraries=['lz4']),
    'lzf': dict(
        sources=['liblzf-3.6/lzf_c.c', 'liblzf-3.6/lzf_d.c'],
        include_dirs=['liblzf-3.6']
        ),
    'lzma': dict(libraries=['lzma']),
    'png': dict(libraries=['png', 'z']),
    'snappy': dict(libraries=['snappy']),
    # 'szip': dict(libraries=['libaec']),
    'tiff': dict(libraries=['tiff']),
    'webp': dict(libraries=['webp']),
    'zfp': dict(libraries=['zfp'], extra_compile_args=OPENMP_ARGS),
    'zlib': dict(libraries=['z']),
    'zopfli': dict(libraries=['zopfli']),
    'zstd': dict(libraries=['zstd']),
    # 'template': dict(
    #     sources=[],
    #     include_dirs=[],
    #     library_dirs=[],
    #     libraries=[],
    #     define_macros=[],
    #     extra_compile_args=[],
    #     ),
}

if os.environ.get('COMPUTERNAME', '').startswith('CG-'):
    # Windows development environment. Use static libraries
    from _inclib import INCLIB
    INCLUDE_DIRS.append(INCLIB)
    LIBRARY_DIRS.append(INCLIB)

    EXTENSIONS['aec'] = dict(libraries=['libaec'])
    EXTENSIONS['bz2'] = dict(libraries=['libbz2'])
    EXTENSIONS['lzf'] = dict(libraries=['lzf'])
    EXTENSIONS['gif'] = dict(libraries=['libgif'])
    # EXTENSIONS['szip'] = dict(libraries=['libaec'])
    EXTENSIONS['zstd'] = dict(libraries=['zstd_static'])
    EXTENSIONS['jpegls']['define_macros'] = [('CHARLS_STATIC', 1)]
    EXTENSIONS['jpeg2k']['define_macros'] += [('OPJ_STATIC', 1)]
    EXTENSIONS['jpeg2k']['include_dirs'] = [INCLIB + 'openjpeg-2.3']
    EXTENSIONS['jpegxr']['include_dirs'] = [INCLIB + 'jxrlib']
    EXTENSIONS['blosc'] = dict(
        libraries=['libblosc', 'zlib', 'lz4', 'snappy', 'zstd_static']
    )
    EXTENSIONS['brotli'] = dict(
        libraries=[
            'brotlienc-static', 'brotlidec-static', 'brotlicommon-static'
        ]
    )
    EXTENSIONS['lzma'] = dict(
        libraries=['lzma-static'],
        define_macros=[('LZMA_API_STATIC', 1)]
    )
    EXTENSIONS['tiff'] = dict(
        libraries=[
            'tiff', 'z', 'jpeg', 'png', 'webp', 'zstd_static', 'lzma-static',
        ],
        define_macros=[('LZMA_API_STATIC', 1)]
    )
    EXTENSIONS['jpegxl'] = dict(
        libraries=[
            'brunslidec-c', 'brunslienc-c',
            # static linking
            'brunslidec-static', 'brunslienc-static', 'brunslicommon-static',
            # vendored brotli currently used for compressing metadata
            'brunsli_brotlidec-static',
            'brunsli_brotlienc-static',
            'brunsli_brotlicommon-static',
        ]
    )

elif os.environ.get('LD_LIBRARY_PATH', os.environ.get('LIBRARY_PATH', '')):
    # Czaki's CI environment
    del EXTENSIONS['jpeg12']

    base_path = os.environ.get(
        'BASE_PATH', os.path.dirname(os.path.abspath(__file__))
    )
    include_base_path = os.path.join(
        base_path, 'build_utils/libs_build/include'
    )
    LIBRARY_DIRS = [
        x for x in os.environ.get(
            'LD_LIBRARY_PATH', os.environ.get('LIBRARY_PATH', '')
        ).split(':') if x
    ]

    if os.path.exists(include_base_path):
        INCLUDE_DIRS.append(include_base_path)
        for el in os.listdir(include_base_path):
            path_to_dir = os.path.join(include_base_path, el)
            if os.path.isdir(path_to_dir):
                INCLUDE_DIRS.append(path_to_dir)
        jxr_path = os.path.join(include_base_path, 'libjxr')
        if os.path.exists(jxr_path):
            jpegxr_include_dirs = [jxr_path]
            for el in os.listdir(jxr_path):
                path_to_dir = os.path.join(jxr_path, el)
                if os.path.isdir(path_to_dir):
                    jpegxr_include_dirs.append(path_to_dir)
            EXTENSIONS['jpegxr']['include_dirs'] = jpegxr_include_dirs

    for dir_path in INCLUDE_DIRS:
        if os.path.exists(os.path.join(dir_path, 'charls', 'charls.h')):
            break
    else:
        del EXTENSIONS['jpegls']

    for dir_path in INCLUDE_DIRS:
        if os.path.exists(os.path.join(dir_path, 'zfp.h')):
            break
    else:
        del EXTENSIONS['zfp']

    EXTENSIONS['jpegxr']['define_macros'] = [('__ANSI__', 1)]

else:
    # Most recent Debian
    del EXTENSIONS['jpeg12']  # jpeg12 library not available
    del EXTENSIONS['jpegls']  # CharLS 2.1 library not available
    del EXTENSIONS['jpegxl']  # Brunsli library not available
    del EXTENSIONS['zfp']  # ZFP library not available

    if sys.platform != 'win32':
        EXTENSIONS['jpeg2k']['include_dirs'] = ['/usr/include/openjpeg-2.3']
        EXTENSIONS['jpegxr']['include_dirs'] = ['/usr/include/jxrlib']
        EXTENSIONS['jpegxr']['define_macros'] = [('__ANSI__', 1)]

###############################################################################

# Use precompiled c files if Cython if not installed
# Work around "Cython in setup_requires doesn't work"
# https://github.com/pypa/setuptools/issues/1317
try:
    import Cython  # noqa
    ext = '.pyx'
except ImportError:
    ext = '.c'


def extension(name):
    e = EXTENSIONS[name]
    return Extension(
        f'imagecodecs._{name}',
        sources=[f'imagecodecs/_{name}' + ext] + e.get('sources', []),
        include_dirs=INCLUDE_DIRS + e.get('include_dirs', []),
        library_dirs=LIBRARY_DIRS + e.get('library_dirs', []),
        libraries=LIBRARIES + e.get('libraries', []),
        define_macros=DEFINE_MACROS + e.get('define_macros', []),
        extra_compile_args=EXTRA_COMPILE_ARGS + e.get('extra_compile_args', [])
    )


class build_ext(_build_ext):
    """Customize build of extensions.

    Delay import numpy until building extensions.
    Add numpy include directory to include_dirs.
    Add user options to skip building specific extensions.

    """

    user_options = _build_ext.user_options + (
        [('lite', None, 'only build the _imcd extension')] +
        [(f'skip-{name}', None, f'do not build the _{name} extension')
         for name in EXTENSIONS]
        )

    def initialize_options(self):
        for name in EXTENSIONS:
            setattr(self, f'skip_{name}', False)
        self.lite = False
        _build_ext.initialize_options(self)

    def finalize_options(self):
        _build_ext.finalize_options(self)

        # add numpy include directory
        # delay import of numpy until setup_requires are installed
        # prevent numpy from detecting setup process
        if isinstance(__builtins__, dict):
            __builtins__['__NUMPY_SETUP__'] = False
        else:
            setattr(__builtins__, '__NUMPY_SETUP__', False)
        import numpy
        self.include_dirs.append(numpy.get_include())

        # remove extensions based on user_options
        for ext in self.extensions.copy():
            name = ext.name.rsplit('_', 1)[-1]
            if (
                (self.lite and name not in ('imcd', 'shared')) or
                getattr(self, f'skip_{name}', False)
            ):
                print(f'skipping {ext.name!r} extension (deselected)')
                self.extensions.remove(ext)


setup(
    name='imagecodecs',
    version=version,
    description=description,
    long_description=readme,
    author='Christoph Gohlke',
    author_email='cgohlke@uci.edu',
    url='https://www.lfd.uci.edu/~gohlke/',
    python_requires='>=3.6',
    install_requires=['numpy>=1.15'],
    setup_requires=['setuptools>=18.0', 'numpy>=1.15'],  # 'cython>=0.29.14'
    extras_require={'all': ['matplotlib>=3.1', 'tifffile>=2019.7.2']},
    tests_require=['pytest', 'tifffile', 'czifile', 'blosc', 'zstd', 'lz4',
                   'python-lzf', 'bitshuffle', 'zopflipy'],  # zfpy
    packages=['imagecodecs'],
    package_data={'imagecodecs': ['licenses/*']},
    entry_points={
        'console_scripts': ['imagecodecs=imagecodecs.__main__:main']},
    ext_modules=[extension(name) for name in sorted(EXTENSIONS)],
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
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
)
