# imagecodecs/setup.py

"""Imagecodecs package setuptools script."""

import sys
import os
import re
import shutil

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext

try:
    import pip
    from packaging.version import parse
    import platform

    if parse(pip.__version__) < parse('19.3') and platform.system() == 'Linux':
        print('Installing imagecodecs wheels requires pip >= 19.3')
except ImportError:
    pass

buildnumber = ''  # e.g 'pre1' or 'post1'

base_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(base_dir, 'imagecodecs/imagecodecs.py')) as fh:
    code = fh.read().replace('\r', '')

version = re.search(r"__version__ = '(.*?)'", code).groups()[0]

version += ('.' + buildnumber) if buildnumber else ''

description = re.search(r'"""(.*)\.(?:\r\n|\r|\n)', code).groups()[0]

readme = re.search(
    r'(?:\r\n|\r|\n){2}"""(.*)"""(?:\r\n|\r|\n){2}__version__',
    code,
    re.MULTILINE | re.DOTALL,
).groups()[0]

readme = '\n'.join(
    [description, '=' * len(description)] + readme.splitlines()[1:]
)

license = re.search(
    r'(# Copyright.*?(?:\r\n|\r|\n))(?:\r\n|\r|\n)+""',
    code,
    re.MULTILINE | re.DOTALL,
).groups()[0]

license = license.replace('# ', '').replace('#', '')

if 'sdist' in sys.argv:
    # update README, LICENSE, and CHANGES files

    with open('README.rst', 'w') as fh:
        fh.write(readme)

    license = re.search(
        r'(# Copyright.*?(?:\r\n|\r|\n))(?:\r\n|\r|\n)+""',
        code,
        re.MULTILINE | re.DOTALL,
    ).groups()[0]

    license = license.replace('# ', '').replace('#', '')

    with open('LICENSE', 'w') as fh:
        fh.write('BSD 3-Clause License\n\n')
        fh.write(license)

    revisions = (
        re.search(
            r'(?:\r\n|\r|\n){2}(Revisions.*)   \.\.\.',
            readme,
            re.MULTILINE | re.DOTALL,
        )
        .groups()[0]
        .strip()
    )

    with open('CHANGES.rst', 'r') as fh:
        old = fh.read()

    old = old.split(revisions.splitlines()[-1])[-1]
    with open('CHANGES.rst', 'w') as fh:
        fh.write(revisions.strip())
        fh.write(old)


def ext(**kwargs):
    """Return Extension arguments."""
    d = dict(
        sources=[],
        include_dirs=[],
        library_dirs=[],
        libraries=[],
        define_macros=[],
        extra_compile_args=[],
        extra_link_args=[],
        cython_compile_env={},
    )
    d.update(kwargs)
    return d


OPTIONS = {
    'cythonize': sys.version_info >= (3, 10) or 'PyPy' in sys.version,
    'include_dirs': ['imagecodecs'],
    'library_dirs': [],
    'libraries': ['m'] if sys.platform != 'win32' else [],
    'define_macros': [('WIN32', 1)] if sys.platform == 'win32' else [],
    'extra_compile_args': [],
    'extra_link_args': [],
}

EXTENSIONS = {
    'shared': ext(cython_compile_env={'IS_PYPY': 'PyPy' in sys.version}),
    'imcd': ext(sources=['imagecodecs/imcd.c']),
    'aec': ext(libraries=['aec']),
    'avif': ext(libraries=['avif']),
    # 'exr': ext(
    #     sources=['3rdparty/tinyexr/tinyexr.cc'],
    #     include_dirs=['3rdparty/tinyexr'],
    # ),
    'bitshuffle': ext(
        sources=[
            '3rdparty/bitshuffle/bitshuffle_core.c',
            '3rdparty/bitshuffle/iochain.c',
        ],
        include_dirs=['3rdparty/bitshuffle'],
    ),
    'blosc': ext(libraries=['blosc']),
    'brotli': ext(libraries=['brotlienc', 'brotlidec', 'brotlicommon']),
    'brunsli': ext(libraries=['brunslidec-c', 'brunslienc-c']),
    'bz2': ext(libraries=['bz2']),
    'deflate': ext(libraries=['deflate']),
    'gif': ext(libraries=['gif']),
    'jpeg2k': ext(
        sources=['3rdparty/openjpeg/color.c'],
        include_dirs=['3rdparty/openjpeg'],
        libraries=['openjp2', 'lcms2'],
    ),
    'jpeg8': ext(
        libraries=['jpeg'], cython_compile_env={'HAVE_LIBJPEG_TURBO': True}
    ),
    'jpeg12': ext(
        libraries=['jpeg12'],
        define_macros=[('BITS_IN_JSAMPLE', 12)],
        cython_compile_env={'HAVE_LIBJPEG_TURBO': True},
    ),
    'jpegls': ext(libraries=['charls']),
    'jpegsof3': ext(sources=['imagecodecs/jpegsof3.cpp']),
    'jpegxl': ext(libraries=['jxl', 'jxl_dec', 'jxl_threads']),
    'jpegxr': ext(
        libraries=['jpegxr', 'jxrglue'],
        define_macros=[('__ANSI__', 1)] if sys.platform != 'win32' else [],
    ),
    'lerc': ext(libraries=['lerc']),
    'ljpeg': ext(
        sources=['3rdparty/liblj92/lj92.c'], include_dirs=['3rdparty/liblj92']
    ),
    'lz4': ext(libraries=['lz4']),
    'lz4f': ext(libraries=['lz4']),
    'lzf': ext(
        sources=['3rdparty/liblzf/lzf_c.c', '3rdparty/liblzf/lzf_d.c'],
        include_dirs=['3rdparty/liblzf'],
    ),
    'lzma': ext(libraries=['lzma']),
    'pglz': ext(
        sources=['3rdparty/postgresql/pg_lzcompress.c'],
        include_dirs=['3rdparty/postgresql'],
    ),
    'png': ext(libraries=['png', 'z']),
    'snappy': ext(libraries=['snappy']),
    # 'szip': ext(libraries=['libaec']),
    'tiff': ext(libraries=['tiff']),
    'webp': ext(libraries=['webp']),
    'zfp': ext(libraries=['zfp']),
    'zlib': ext(libraries=['z']),
    'zopfli': ext(libraries=['zopfli']),
    'zstd': ext(libraries=['zstd']),
}


def customize_build_default(EXTENSIONS, OPTIONS):
    """Customize build for common platforms: recent Debian, arch..."""
    import platform

    del EXTENSIONS['avif']  # libavif library not commonly available
    del EXTENSIONS['jpeg12']  # jpeg12 requires custom build
    del EXTENSIONS['lerc']  # LERC library not commonly available
    del EXTENSIONS['lz4f']  # requires static linking

    if 'arch' not in platform.platform():
        del EXTENSIONS['jpegls']  # CharLS 2.1 library not commonly available
        del EXTENSIONS['jpegxl']  # jpeg-xl library not commonly available
        del EXTENSIONS['brunsli']  # Brunsli library not commonly available
        del EXTENSIONS['zfp']  # ZFP library not commonly available

    if sys.platform == 'win32':
        EXTENSIONS['bz2']['libraries'] = ['libbz2']
    else:
        EXTENSIONS['jpeg2k']['include_dirs'].extend(
            ('/usr/include/openjpeg-2.3', '/usr/include/openjpeg-2.4')
        )
        EXTENSIONS['jpegxr']['include_dirs'].append('/usr/include/jxrlib')
        EXTENSIONS['zopfli']['include_dirs'].append('/usr/include/zopfli')


def customize_build_cg(EXTENSIONS, OPTIONS):
    """Customize build for Windows development environment with static libs."""
    from _inclib import INCLIB

    OPTIONS['include_dirs'].append(INCLIB)
    OPTIONS['library_dirs'].append(INCLIB)

    EXTENSIONS['avif']['libraries'] = [
        'avif',
        'aom',
        'libdav1d',
        'rav1e',
        'Ws2_32',
        'Advapi32',
        'Userenv',
    ]
    EXTENSIONS['aec']['libraries'] = ['libaec']
    EXTENSIONS['bz2']['libraries'] = ['libbz2']
    EXTENSIONS['lzf']['libraries'] = ['lzf']
    EXTENSIONS['gif']['libraries'] = ['libgif']
    # EXTENSIONS['szip']['libraries'] = ['libaec']
    EXTENSIONS['deflate']['libraries'] = ['libdeflatestatic']

    EXTENSIONS['zstd']['libraries'] = ['zstd_static']
    EXTENSIONS['jpegls']['define_macros'].append(('CHARLS_STATIC', 1))
    EXTENSIONS['jpeg2k']['define_macros'].append(('OPJ_STATIC', 1))
    EXTENSIONS['jpegxr']['include_dirs'].append(INCLIB + 'jxrlib')
    # EXTENSIONS['exr']['define_macros'].append(('TINYEXR_USE_OPENMP', 1))
    # EXTENSIONS['exr']['extra_compile_args'] = ['/openmp']
    EXTENSIONS['zfp']['extra_compile_args'] = ['/openmp']
    EXTENSIONS['blosc']['libraries'] = [
        'libblosc',
        'zlib',
        'lz4',
        'snappy',
        'zstd_static',
    ]
    EXTENSIONS['brotli']['libraries'] = [
        'brotlienc-static',
        'brotlidec-static',
        'brotlicommon-static',
    ]
    EXTENSIONS['lzma']['libraries'] = ['lzma-static']
    EXTENSIONS['lzma']['define_macros'].append(('LZMA_API_STATIC', 1))
    EXTENSIONS['tiff']['define_macros'].append(('LZMA_API_STATIC', 1))
    EXTENSIONS['tiff']['libraries'] = [
        'tiff',
        'z',
        'jpeg',
        'png',
        'webp',
        'zstd_static',
        'lzma-static',
        'libdeflatestatic',
        'lerc',
    ]
    EXTENSIONS['jpegxl']['define_macros'].extend(
        (('JXL_STATIC_DEFINE', 1), ('JXL_THREADS_STATIC_DEFINE', 1))
    )
    EXTENSIONS['jpegxl']['libraries'] = [
        'jxl-static',
        'jxl_dec-static',
        'jxl_extras-static',
        'jxl_threads-static',
        'jxl_brotlienc-static',
        'jxl_brotlidec-static',
        'jxl_brotlicommon-static',
        'jxl_hwy',
        'jxl_lodepng',
        'jxl_lskcms',
        'jxl_sjpeg',
    ]
    EXTENSIONS['brunsli']['libraries'] = [
        'brunslidec-c',
        'brunslienc-c',
        # static linking
        'brunslidec-static',
        'brunslienc-static',
        'brunslicommon-static',
        # vendored brotli currently used for compressing metadata
        'brunsli_brotlidec-static',
        'brunsli_brotlienc-static',
        'brunsli_brotlicommon-static',
    ]


def customize_build_ci(EXTENSIONS, OPTIONS):
    """Customize build for Czaki's CI environment."""

    if not os.environ.get('SKIP_OMP', False):
        if sys.platform == 'darwin':
            EXTENSIONS['zfp']['extra_compile_args'].append('-Xpreprocessor')
            EXTENSIONS['zfp']['extra_link_args'].append('-lomp')
        EXTENSIONS['zfp']['extra_compile_args'].append('-fopenmp')

    base_path = os.environ.get(
        'BASE_PATH', os.path.dirname(os.path.abspath(__file__))
    )
    include_base_path = os.path.join(
        base_path, 'build_utils', 'libs_build', 'include'
    )
    OPTIONS['library_dirs'] = [
        x
        for x in os.environ.get(
            'LD_LIBRARY_PATH', os.environ.get('LIBRARY_PATH', '')
        ).split(':')
        if x
    ]

    EXTENSIONS['zopfli']['include_dirs'].append(
        os.path.join(include_base_path, 'zopfli')
    )

    if os.path.exists(include_base_path):
        OPTIONS['include_dirs'].append(include_base_path)
        for el in os.listdir(include_base_path):
            path_to_dir = os.path.join(include_base_path, el)
            if os.path.isdir(path_to_dir):
                OPTIONS['include_dirs'].append(path_to_dir)
        jxr_path = os.path.join(include_base_path, 'libjxr')
        if os.path.exists(jxr_path):
            jpegxr_include_dirs = [jxr_path]
            for el in os.listdir(jxr_path):
                path_to_dir = os.path.join(jxr_path, el)
                if os.path.isdir(path_to_dir):
                    jpegxr_include_dirs.append(path_to_dir)
            EXTENSIONS['jpegxr']['include_dirs'] = jpegxr_include_dirs

    for dir_path in OPTIONS['include_dirs']:
        if os.path.exists(os.path.join(dir_path, 'jxl', 'types.h')):
            break
    else:
        del EXTENSIONS['jpegxl']

    libjpeg12_base_path = os.path.join(
        base_path, 'build_utils', 'libs_build', 'libjpeg12'
    )
    if os.path.exists(libjpeg12_base_path):
        EXTENSIONS['jpeg12']['libraries'] = ['jpeg12']
        EXTENSIONS['jpeg12']['include_dirs'] = [
            os.path.join(libjpeg12_base_path, 'include')
        ]
    else:
        del EXTENSIONS['jpeg12']

    if os.environ.get('IMCD_SKIP_JPEG12', False):
        # all tests fail on macOS; likely conflict with jpeg 8-bit dll
        del EXTENSIONS['jpeg12']

    for dir_path in OPTIONS['include_dirs']:
        if os.path.exists(os.path.join(dir_path, 'avif', 'avif.h')):
            break
    else:
        del EXTENSIONS['avif']

    for dir_path in OPTIONS['include_dirs']:
        if os.path.exists(os.path.join(dir_path, 'charls', 'charls.h')):
            break
    else:
        del EXTENSIONS['jpegls']

    for dir_path in OPTIONS['include_dirs']:
        if os.path.exists(os.path.join(dir_path, 'zfp.h')):
            break
    else:
        del EXTENSIONS['zfp']

    for dir_path in OPTIONS['include_dirs']:
        if os.path.exists(os.path.join(dir_path, 'Lerc_c_api.h')):
            break
    else:
        del EXTENSIONS['lerc']


def customize_build_cf(EXTENSIONS, OPTIONS):
    """Customize build for conda-forge."""

    del EXTENSIONS['avif']
    del EXTENSIONS['jpeg12']
    del EXTENSIONS['jpegxl']

    # build the jpeg8 extension against libjpeg v9 instead of libjpeg-turbo
    OPTIONS['cythonize'] = True
    EXTENSIONS['jpeg8']['cython_compile_env']['HAVE_LIBJPEG_TURBO'] = False

    EXTENSIONS['lerc']['libraries'] = ['Lerc']

    if sys.platform == 'win32':
        del EXTENSIONS['brunsli']  # brunsli not stable on conda-forge

        EXTENSIONS['lz4f']['libraries'] = ['liblz4']
        EXTENSIONS['bz2']['libraries'] = ['bzip2']
        EXTENSIONS['jpeg2k']['include_dirs'] += [
            os.path.join(
                os.environ['LIBRARY_INC'], 'openjpeg-' + os.environ['openjpeg']
            )
        ]
        EXTENSIONS['deflate']['libraries'] = ['libdeflate']
        EXTENSIONS['jpegls']['libraries'] = ['charls-2-x64']
        EXTENSIONS['lz4']['libraries'] = ['liblz4']
        EXTENSIONS['lzma']['libraries'] = ['liblzma']
        EXTENSIONS['png']['libraries'] = ['libpng', 'z']
        EXTENSIONS['webp']['libraries'] = ['libwebp']
        EXTENSIONS['zopfli']['include_dirs'] = [
            os.path.join(os.environ['LIBRARY_INC'], 'zopfli')
        ]
        EXTENSIONS['jpegxr']['include_dirs'] = [
            os.path.join(os.environ['LIBRARY_INC'], 'jxrlib')
        ]
        EXTENSIONS['jpegxr']['libraries'] = ['libjpegxr', 'libjxrglue']
    else:
        EXTENSIONS['zopfli']['include_dirs'] = [
            os.path.join(os.environ['PREFIX'], 'include', 'zopfli')
        ]
        EXTENSIONS['jpegxr']['include_dirs'] = [
            os.path.join(os.environ['PREFIX'], 'include', 'jxrlib')
        ]
        EXTENSIONS['jpegxr']['libraries'] = ['jpegxr', 'jxrglue']


def customize_build_macports(EXTENSIONS, OPTIONS):
    """Customize build for MacPorts."""

    del EXTENSIONS['avif']
    del EXTENSIONS['brunsli']
    del EXTENSIONS['deflate']
    del EXTENSIONS['jpeg12']
    del EXTENSIONS['jpegls']
    del EXTENSIONS['jpegxl']
    del EXTENSIONS['jpegxr']
    del EXTENSIONS['lerc']
    del EXTENSIONS['lz4f']
    del EXTENSIONS['zfp']

    EXTENSIONS['aec']['library_dirs'] = ['%PREFIX%/lib/libaec/lib']
    EXTENSIONS['aec']['include_dirs'] = ['%PREFIX%/lib/libaec/include']
    EXTENSIONS['gif']['include_dirs'] = ['%PREFIX%/include/giflib5']
    EXTENSIONS['jpeg2k']['include_dirs'].extend(
        ('%PREFIX%/include/openjpeg-2.3', '%PREFIX%/include/openjpeg-2.4')
    )
    EXTENSIONS['jpeg8']['cython_compile_env']['HAVE_LIBJPEG_TURBO'] = False
    OPTIONS['cythonize'] = True


def customize_build_mingw(EXTENSIONS, OPTIONS):
    """Customize build for mingw-w64."""

    del EXTENSIONS['brunsli']
    del EXTENSIONS['jpeg12']
    del EXTENSIONS['jpegxl']
    del EXTENSIONS['lerc']
    del EXTENSIONS['zfp']

    EXTENSIONS['jpeg2k']['include_dirs'].extend(
        (
            sys.prefix + '/include/openjpeg-2.3',
            sys.prefix + '/include/openjpeg-2.4',
        )
    )
    EXTENSIONS['jpegxr']['include_dirs'].append(sys.prefix + '/include/jxrlib')


# customize builds based on environment
try:
    from imagecodecs_distributor_setup import customize_build
except ImportError:
    if os.environ.get('COMPUTERNAME', '').startswith('CG-'):
        customize_build = customize_build_cg
    elif os.environ.get('CONDA_BUILD', ''):
        customize_build = customize_build_cf
    elif shutil.which('port'):
        customize_build = customize_build_macports
    elif os.environ.get('LD_LIBRARY_PATH', os.environ.get('LIBRARY_PATH', '')):
        customize_build = customize_build_ci
    elif os.name == 'nt' and 'GCC' in sys.version:
        customize_build = customize_build_mingw
    else:
        customize_build = customize_build_default

customize_build(EXTENSIONS, OPTIONS)

# use precompiled c files if Cython is not installed
# work around "Cython in setup_requires doesn't work"
# https://github.com/pypa/setuptools/issues/1317
try:
    import Cython  # noqa

    EXT = '.pyx'
except ImportError:
    if OPTIONS['cythonize']:
        raise
    Cython = None
    EXT = '.c'


class build_ext(_build_ext):
    """Customize build of extensions.

    Delay importing numpy until building extensions.
    Add numpy include directory to include_dirs.
    Skip building deselected extensions.
    Cythonize with compile time macros.

    """

    user_options = _build_ext.user_options + (
        [('lite', None, 'only build the _imcd extension')]
        + [
            (f'skip-{name}', None, f'do not build the _{name} extension')
            for name in EXTENSIONS
        ]
    )

    def initialize_options(self):
        for name in EXTENSIONS:
            setattr(self, f'skip_{name}', False)
        self.lite = False
        _build_ext.initialize_options(self)

    def finalize_options(self):
        _build_ext.finalize_options(self)

        # remove extensions based on user_options
        for ext in self.extensions.copy():
            name = ext.name.rsplit('_', 1)[-1]
            if (self.lite and name not in ('imcd', 'shared')) or getattr(
                self, f'skip_{name}', False
            ):
                print(f'skipping {ext.name!r} extension (deselected)')
                self.extensions.remove(ext)

        # add numpy include directory
        # delay import of numpy until setup_requires are installed
        # prevent numpy from detecting setup process
        if isinstance(__builtins__, dict):
            __builtins__['__NUMPY_SETUP__'] = False
        else:
            setattr(__builtins__, '__NUMPY_SETUP__', False)
        import numpy

        self.include_dirs.append(numpy.get_include())

        # Cythonize with compile time macros
        if Cython is not None and self.distribution.ext_modules:
            from Cython.Build.Dependencies import cythonize

            for i, ext in enumerate(self.extensions):
                name = ext.name.rsplit('_', 1)[-1]
                cyenv = EXTENSIONS[name].get('cython_compile_env', {})
                if OPTIONS['cythonize'] or cyenv:
                    cythonize(
                        ext,
                        include_path=ext.include_dirs,
                        compile_time_env=cyenv,
                        force=OPTIONS['cythonize'],
                    )


def extension(name):
    """Return setuptools Extension."""
    ext = EXTENSIONS[name]
    return Extension(
        f'imagecodecs._{name}',
        sources=[f'imagecodecs/_{name}' + EXT] + ext['sources'],
        include_dirs=OPTIONS['include_dirs'] + ext['include_dirs'],
        library_dirs=OPTIONS['library_dirs'] + ext['library_dirs'],
        libraries=OPTIONS['libraries'] + ext['libraries'],
        define_macros=OPTIONS['define_macros'] + ext['define_macros'],
        extra_compile_args=(
            OPTIONS['extra_compile_args'] + ext['extra_compile_args']
        ),
    )


setup(
    name='imagecodecs',
    version=version,
    description=description,
    long_description=readme,
    author='Christoph Gohlke',
    author_email='cgohlke@uci.edu',
    license='BSD',
    url='https://www.lfd.uci.edu/~gohlke/',
    project_urls={
        'Bug Tracker': 'https://github.com/cgohlke/imagecodecs/issues',
        'Source Code': 'https://github.com/cgohlke/imagecodecs',
        # 'Documentation': 'https://',
    },
    python_requires='>=3.7',
    install_requires=['numpy>=1.15.1'],
    setup_requires=['setuptools>=18.0', 'numpy>=1.15.1'],  # 'cython>=0.29.21'
    extras_require={
        'all': ['matplotlib>=3.2', 'tifffile>=2021.1.11', 'numcodecs']
    },
    tests_require=[
        'pytest',
        'tifffile',
        'czifile',
        'blosc',
        'zstd',
        'lz4',
        'python-lzf',
        'bitshuffle',
        'zopflipy',
        'zarr',
        'numcodecs'
        # 'zfpy',
        # 'brotli',
        # 'deflate',
        # 'pytinyexr',
    ],
    packages=['imagecodecs'],
    package_data={'imagecodecs': ['licenses/*']},
    entry_points={
        'console_scripts': ['imagecodecs=imagecodecs.__main__:main']
    },
    ext_modules=[extension(name) for name in sorted(EXTENSIONS)],
    cmdclass={'build_ext': build_ext},
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
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
)
