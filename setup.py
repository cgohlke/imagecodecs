# imagecodecs/setup.py

"""Imagecodecs package Setuptools script."""

import os
import re
import shutil
import sys

import Cython  # noqa
import numpy
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext

buildnumber = ''  # e.g. 'pre1' or 'post1'

DEBUG = bool(os.environ.get('IMAGECODECS_DEBUG', False))

base_dir = os.path.dirname(os.path.abspath(__file__))


def search(pattern, code, flags=0):
    # return first match for pattern in code
    match = re.search(pattern, code, flags)
    if match is None:
        raise ValueError(f'{pattern!r} not found')
    return match.groups()[0]


with open(
    os.path.join(base_dir, 'imagecodecs/imagecodecs.py'), encoding='utf-8'
) as fh:
    code = fh.read().replace('\r\n', '\n').replace('\r', '\n')

version = search(r"__version__ = '(.*?)'", code).replace('.x.x', '.dev0')
version += ('.' + buildnumber) if buildnumber else ''

description = search(r'"""(.*)\.(?:\r\n|\r|\n)', code)

readme = search(
    r'(?:\r\n|\r|\n){2}r"""(.*)"""(?:\r\n|\r|\n){2}from __future__',
    code,
    re.MULTILINE | re.DOTALL,
)
readme = '\n'.join(
    [description, '=' * len(description)] + readme.splitlines()[1:]
)

if 'sdist' in sys.argv:
    # update README, LICENSE, and CHANGES files

    with open('README.rst', 'w', encoding='utf-8') as fh:
        fh.write(readme)

    license = search(
        r'(# Copyright.*?(?:\r\n|\r|\n))(?:\r\n|\r|\n)+r""',
        code,
        re.MULTILINE | re.DOTALL,
    )
    license = license.replace('# ', '').replace('#', '')

    with open('LICENSE', 'w', encoding='utf-8') as fh:
        fh.write('BSD 3-Clause License\n\n')
        fh.write(license)

    revisions = search(
        r'(?:\r\n|\r|\n){2}(Revisions.*)- …',
        readme,
        re.MULTILINE | re.DOTALL,
    ).strip()

    with open('CHANGES.rst', encoding='utf-8') as fh:
        old = fh.read()

    old = old.split(revisions.splitlines()[-1])[-1]
    with open('CHANGES.rst', 'w', encoding='utf-8') as fh:
        fh.write(revisions.strip())
        fh.write(old)


def ext(**kwargs):
    """Return Extension arguments."""
    d: dict[str, object] = dict(
        sources=[],
        include_dirs=[],
        library_dirs=[],
        libraries=[],
        define_macros=[],
        extra_compile_args=[],
        extra_link_args=[],
        depends=[],
        cython_compile_time_env={},
    )
    d.update(kwargs)
    return d


OPTIONS = {
    'include_dirs': ['imagecodecs'],
    'library_dirs': [],
    'libraries': ['m'] if sys.platform != 'win32' else [],
    'define_macros': [
        # ('CYTHON_TRACE_NOGIL', '1'),
        # ('CYTHON_LIMITED_API', '1'),
        # ('Py_LIMITED_API', '1'),
        ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'),
    ]
    + [('WIN32', 1)]  # type: ignore
    if sys.platform == 'win32'
    else [],
    'extra_compile_args': ['/Zi', '/Od'] if DEBUG else [],
    'extra_link_args': ['-debug:full'] if DEBUG else [],
    'depends': ['imagecodecs/_shared.pxd'],
    'cython_compile_time_env': {},
}

EXTENSIONS = {
    'shared': ext(
        cython_compile_time_env={'IS_PYPY': 'pypy' in sys.version.lower()},
    ),
    'imcd': ext(sources=['imagecodecs/imcd.c']),
    'aec': ext(libraries=['aec']),
    'apng': ext(libraries=['png']),
    'avif': ext(libraries=['avif']),
    'bcn': ext(
        include_dirs=['3rdparty/bcdec'],
        define_macros=[('BCDEC_STATIC', 1), ('BCDEC_IMPLEMENTATION', 1)],
    ),
    'bitshuffle': ext(
        sources=[
            '3rdparty/bitshuffle/bitshuffle_core.c',
            '3rdparty/bitshuffle/iochain.c',
        ],
        include_dirs=['3rdparty/bitshuffle'],
    ),
    'blosc': ext(libraries=['blosc']),
    'blosc2': ext(libraries=['blosc2']),
    'bmp': ext(),
    'brotli': ext(libraries=['brotlienc', 'brotlidec', 'brotlicommon']),
    'brunsli': ext(libraries=['brunslidec-c', 'brunslienc-c']),
    'bz2': ext(libraries=['bz2']),
    'cms': ext(libraries=['lcms2']),
    'deflate': ext(libraries=['deflate']),
    # 'exr': ext(
    #     sources=['3rdparty/tinyexr/tinyexr.cc'],
    #     include_dirs=['3rdparty/tinyexr'],
    # ),
    'gif': ext(libraries=['gif']),
    'h5checksum': ext(
        sources=['3rdparty/hdf5/h5checksum.c'],
        include_dirs=['3rdparty/hdf5'],
    ),
    'heif': ext(libraries=['heif']),
    'jetraw': ext(libraries=['jetraw', 'dpcore']),
    'jpeg2k': ext(
        sources=['3rdparty/openjpeg/color.c'],
        include_dirs=['3rdparty/openjpeg'],
        libraries=['openjp2', 'lcms2'],
    ),
    'jpeg8': ext(
        sources=['imagecodecs/_jpeg8_legacy.pyx'],
        libraries=['jpeg'],
    ),
    'jpegls': ext(libraries=['charls']),
    'jpegsof3': ext(
        sources=['3rdparty/jpegsof3/jpegsof3.cpp'],
        include_dirs=['3rdparty/jpegsof3'],
    ),
    'jpegxl': ext(libraries=['jxl', 'jxl_threads']),
    'jpegxr': ext(
        libraries=['jpegxr', 'jxrglue'],
        define_macros=[('__ANSI__', 1)] if sys.platform != 'win32' else [],
    ),
    'lerc': ext(libraries=['Lerc']),
    'ljpeg': ext(
        sources=['3rdparty/liblj92/lj92.c'], include_dirs=['3rdparty/liblj92']
    ),
    'lz4': ext(libraries=['lz4']),
    'lz4f': ext(libraries=['lz4']),
    'lzf': ext(
        sources=['3rdparty/liblzf/lzf_c.c', '3rdparty/liblzf/lzf_d.c'],
        include_dirs=['3rdparty/liblzf'],
    ),
    'lzfse': ext(libraries=['lzfse'], sources=['imagecodecs/imcd.c']),
    'lzham': ext(libraries=['lzham']),
    'lzma': ext(libraries=['lzma']),
    'lzo': ext(libraries=['lzokay-c', 'lzokay']),
    'mozjpeg': ext(libraries=['mozjpeg']),
    # 'nvjpeg': ext(libraries=['nvjpeg', 'cuda']),
    # 'nvjpeg2k': ext(libraries=['nvjpeg2k', 'cuda']),
    'pglz': ext(
        sources=['3rdparty/postgresql/pg_lzcompress.c'],
        include_dirs=['3rdparty/postgresql'],
    ),
    'png': ext(libraries=['png']),
    'qoi': ext(
        include_dirs=['3rdparty/qoi'],
        define_macros=[('QOI_IMPLEMENTATION', 1)],
    ),
    'quantize': ext(
        sources=['3rdparty/netcdf-c/nc4var.c'],
        include_dirs=['3rdparty/netcdf-c'],
    ),
    'rgbe': ext(
        sources=['3rdparty/rgbe/rgbe.c', 'imagecodecs/imcd.c'],
        include_dirs=['3rdparty/rgbe'],
    ),
    'rcomp': ext(
        sources=['3rdparty/cfitsio/ricecomp.c'],
        include_dirs=['3rdparty/cfitsio'],
    ),
    'snappy': ext(libraries=['snappy']),
    'sperr': ext(libraries=['SPERR']),
    'spng': ext(
        sources=['3rdparty/libspng/spng.c'],
        include_dirs=['3rdparty/libspng'],
        define_macros=[('SPNG_STATIC', 1)],
        libraries=['z'],
    ),
    'szip': ext(libraries=['sz']),
    'tiff': ext(libraries=['tiff']),
    'webp': ext(libraries=['webp', 'webpdemux']),
    'zfp': ext(libraries=['zfp']),
    'zlib': ext(libraries=['z']),
    'zlibng': ext(libraries=['z-ng']),
    'zopfli': ext(libraries=['zopfli']),
    'zstd': ext(libraries=['zstd']),
}


def customize_build_default(EXTENSIONS, OPTIONS):
    """Customize build for common platforms: recent Debian, arch..."""
    import platform

    del EXTENSIONS['apng']  # apng patch not commonly available
    del EXTENSIONS['avif']  # libavif library not commonly available
    del EXTENSIONS['blosc2']  # c-blosc2 library not commonly available
    # del EXTENSIONS['heif']  # LGPL/GPL
    del EXTENSIONS['jetraw']  # commercial
    del EXTENSIONS['lerc']  # LERC library not commonly available
    del EXTENSIONS['lz4f']  # requires static linking
    del EXTENSIONS['lzfse']  # lzfse not commonly available
    del EXTENSIONS['lzham']  # lzham not commonly available
    del EXTENSIONS['lzo']  # lzokay not commonly available
    del EXTENSIONS['mozjpeg']  # Win32 only
    del EXTENSIONS['sperr']  # sperr not commonly available
    del EXTENSIONS['zlibng']  # zlib-ng library not commonly available

    if 'arch' not in platform.platform():
        del EXTENSIONS['jpegls']  # CharLS 2.1 library not commonly available
        del EXTENSIONS['jpegxl']  # jpeg-xl library not commonly available
        del EXTENSIONS['brunsli']  # Brunsli library not commonly available
        del EXTENSIONS['zfp']  # ZFP library not commonly available

    if sys.platform == 'win32':
        EXTENSIONS['bz2']['libraries'] = ['libbz2']
    else:
        EXTENSIONS['jpeg2k']['include_dirs'].extend(
            (
                '/usr/include/openjpeg-2.3',
                '/usr/include/openjpeg-2.4',
                '/usr/include/openjpeg-2.5',
            )
        )
        EXTENSIONS['jpegxr']['include_dirs'].append('/usr/include/jxrlib')
        EXTENSIONS['zopfli']['include_dirs'].append('/usr/include/zopfli')


def customize_build_cgohlke(EXTENSIONS, OPTIONS):
    """Customize build for Windows development environment with static libs."""
    INCLIB = os.environ.get('INCLIB', '.')

    OPTIONS['include_dirs'].append(os.path.join(INCLIB, 'lib'))
    OPTIONS['library_dirs'].append(os.path.join(INCLIB, 'include'))

    dlls: list[str] = []  # 'heif.dll'
    if '64 bit' in sys.version:
        for dll in dlls:
            shutil.copyfile(
                os.path.join(INCLIB, 'bin', dll), 'imagecodecs/' + dll
            )
    else:
        # del EXTENSIONS['nvjpeg2k']
        del EXTENSIONS['jetraw']
        del EXTENSIONS['heif']
        del EXTENSIONS['sperr']
        for dll in dlls:
            try:
                os.remove('imagecodecs/' + dll)
            except FileNotFoundError:
                pass
    if 'ARM64' in sys.version:
        del EXTENSIONS['jetraw']

    # EXTENSIONS['exr']['define_macros'].append(('TINYEXR_USE_OPENMP', 1))
    # EXTENSIONS['exr']['extra_compile_args'] = ['/openmp']

    if not os.environ.get('USE_JPEG8_LEGACY', False):
        # use libjpeg-turbo 3
        EXTENSIONS['jpeg8']['sources'] = []

    EXTENSIONS['mozjpeg']['include_dirs'] = [
        os.path.join(INCLIB, 'include', 'mozjpeg')
    ]
    EXTENSIONS['mozjpeg']['libraries'] = ['mozjpeg-static']

    EXTENSIONS['avif']['libraries'] = [
        'avif',
        'aom',
        'libdav1d',
        'rav1e',
        'SvtAv1Enc',
        'SvtAv1Dec',
        'Ws2_32',
        'Advapi32',
        'Userenv',
        'Bcrypt',
        'ntdll',
    ]
    EXTENSIONS['szip']['libraries'] = ['szip-static']
    EXTENSIONS['cms']['libraries'] = ['lcms2_static']
    EXTENSIONS['aec']['libraries'] = ['aec-static']
    EXTENSIONS['bz2']['libraries'] = ['libbz2']
    EXTENSIONS['lzf']['libraries'] = ['lzf']
    EXTENSIONS['gif']['libraries'] = ['libgif']
    EXTENSIONS['webp']['libraries'] = [
        'libwebp',
        'libwebpdemux',
        'libsharpyuv',
    ]

    # link with static zlib-ng compatibility mode library
    EXTENSIONS['png']['libraries'] = ['png', 'zlibstatic-ng-compat']
    EXTENSIONS['apng']['libraries'] = ['png', 'zlibstatic-ng-compat']

    EXTENSIONS['lzham']['libraries'] = ['lzhamlib', 'lzhamcomp', 'lzhamdecomp']

    EXTENSIONS['deflate']['libraries'] = ['deflatestatic']
    EXTENSIONS['zlibng']['libraries'] = ['zlibstatic-ng']
    EXTENSIONS['zstd']['libraries'] = ['zstd_static']
    EXTENSIONS['lerc']['define_macros'].append(('LERC_STATIC', 1))
    EXTENSIONS['jpegls']['define_macros'].append(('CHARLS_STATIC', 1))
    EXTENSIONS['jpeg2k']['define_macros'].append(('OPJ_STATIC', 1))
    EXTENSIONS['jpeg2k']['include_dirs'].append(
        os.path.join(INCLIB, 'include', 'openjpeg-2.5')
    )
    EXTENSIONS['jpegxr']['include_dirs'].append(
        os.path.join(INCLIB, 'include', 'jxrlib')
    )
    EXTENSIONS['zopfli']['include_dirs'].append(
        os.path.join(INCLIB, 'include', 'zopfli')
    )
    EXTENSIONS['zfp']['extra_compile_args'] = ['/openmp']
    EXTENSIONS['blosc']['libraries'] = [
        'libblosc',
        'zlib',
        'lz4',
        'snappy',
        'zstd_static',
    ]
    EXTENSIONS['blosc2']['libraries'] = [
        'libblosc2',
        'zlibstatic-ng',
        'lz4',
        'zstd_static',
    ]
    EXTENSIONS['lzma']['libraries'] = ['liblzma']
    EXTENSIONS['lzma']['define_macros'].append(('LZMA_API_STATIC', 1))
    EXTENSIONS['tiff']['define_macros'].extend(
        (('LZMA_API_STATIC', 1), ('LERC_STATIC', 1))
    )
    EXTENSIONS['tiff']['libraries'] = [
        'tiff',
        'jpeg',
        'png',
        'zlibstatic-ng-compat',
        'libwebp',
        'libsharpyuv',
        'zstd_static',
        'liblzma',
        'deflatestatic',
        'lerc',
    ]
    EXTENSIONS['jpegxl']['define_macros'].extend(
        (('JXL_STATIC_DEFINE', 1), ('JXL_THREADS_STATIC_DEFINE', 1))
    )
    EXTENSIONS['jpegxl']['libraries'] = [
        'jxl',
        'jxl_cms',
        'jxl_extras_codec',
        'jxl_threads',
        'brotlienc',
        'brotlidec',
        'brotlicommon',
        'hwy',
        'lcms2_static',
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


def customize_build_cibuildwheel(EXTENSIONS, OPTIONS):
    """Customize build for Czaki's cibuildwheel environment."""

    del EXTENSIONS['heif']  # Win32 only
    del EXTENSIONS['jetraw']  # commercial
    del EXTENSIONS['mozjpeg']  # Win32 only

    EXTENSIONS['jpeg8']['sources'] = []  # use libjpeg-turbo 3

    EXTENSIONS['lzham']['libraries'] = ['lzhamdll']
    if sys.platform == 'darwin':
        del EXTENSIONS['lzham']

    if not os.environ.get('SKIP_OMP', False):
        if sys.platform == 'darwin':
            EXTENSIONS['zfp']['extra_compile_args'].append('-Xpreprocessor')
            EXTENSIONS['zfp']['extra_link_args'].append('-lomp')
        EXTENSIONS['zfp']['extra_compile_args'].append('-fopenmp')

    OPTIONS['library_dirs'] = [
        x
        for x in os.environ.get(
            'LD_LIBRARY_PATH', os.environ.get('LIBRARY_PATH', '')
        ).split(':')
        if x
    ]

    base_path = os.environ.get(
        'BASE_PATH', os.path.dirname(os.path.abspath(__file__))
    )
    include_base_path = os.path.join(
        base_path, 'build_utils', 'libs_build', 'include'
    )

    OPTIONS['include_dirs'].append(include_base_path)
    for el in os.listdir(include_base_path):
        path_to_dir = os.path.join(include_base_path, el)
        if os.path.isdir(path_to_dir):
            OPTIONS['include_dirs'].append(path_to_dir)

    for dir_path in OPTIONS['include_dirs']:
        if os.path.exists(os.path.join(dir_path, 'jxl', 'types.h')):
            break
    else:
        del EXTENSIONS['jpegxl']


def customize_build_condaforge(EXTENSIONS, OPTIONS):
    """Customize build for conda-forge."""

    del EXTENSIONS['apng']
    del EXTENSIONS['heif']
    del EXTENSIONS['jetraw']  # commercial
    del EXTENSIONS['jpegxl']
    del EXTENSIONS['lzfse']
    del EXTENSIONS['lzham']
    del EXTENSIONS['lzo']
    del EXTENSIONS['mozjpeg']  # Win32 only
    del EXTENSIONS['sperr']
    del EXTENSIONS['zlibng']

    EXTENSIONS['jpeg8']['sources'] = []  # use libjpeg-turbo 3

    if sys.platform == 'win32':
        del EXTENSIONS['brunsli']  # brunsli not stable on conda-forge

        EXTENSIONS['lz4f']['libraries'] = ['liblz4']
        EXTENSIONS['bz2']['libraries'] = ['bzip2']
        EXTENSIONS['jpeg2k']['include_dirs'] += [
            os.path.join(
                os.environ['LIBRARY_INC'], 'openjpeg-' + os.environ['openjpeg']
            )
        ]
        EXTENSIONS['jpegls']['libraries'] = ['charls-2-x64']
        EXTENSIONS['lz4']['libraries'] = ['liblz4']
        EXTENSIONS['lzma']['libraries'] = ['liblzma']
        EXTENSIONS['png']['libraries'] = ['libpng', 'z']
        EXTENSIONS['webp']['libraries'] = ['libwebp', 'libwebpdemux']
        EXTENSIONS['zopfli']['include_dirs'] = [
            os.path.join(os.environ['LIBRARY_INC'], 'zopfli')
        ]
        EXTENSIONS['jpegxr']['include_dirs'] = [
            os.path.join(os.environ['LIBRARY_INC'], 'jxrlib')
        ]
        EXTENSIONS['jpegxr']['libraries'] = ['libjpegxr', 'libjxrglue']
        EXTENSIONS['szip']['libraries'] = ['szip']
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

    del EXTENSIONS['apng']
    del EXTENSIONS['avif']
    del EXTENSIONS['blosc2']
    del EXTENSIONS['brunsli']
    del EXTENSIONS['deflate']
    del EXTENSIONS['heif']
    del EXTENSIONS['jetraw']  # commercial
    del EXTENSIONS['jpegls']
    del EXTENSIONS['jpegxl']
    del EXTENSIONS['jpegxr']
    del EXTENSIONS['lerc']
    del EXTENSIONS['lz4f']
    del EXTENSIONS['lzfse']
    del EXTENSIONS['lzham']
    del EXTENSIONS['lzo']
    del EXTENSIONS['mozjpeg']  # Win32 only
    del EXTENSIONS['sperr']
    del EXTENSIONS['zfp']
    del EXTENSIONS['zlibng']

    # uncomment if building with libjpeg-turbo 3
    # EXTENSIONS['jpeg8']['sources'] = []

    EXTENSIONS['szip']['library_dirs'] = ['%PREFIX%/lib/libaec/lib']
    EXTENSIONS['szip']['include_dirs'] = ['%PREFIX%/lib/libaec/include']
    EXTENSIONS['aec']['library_dirs'] = ['%PREFIX%/lib/libaec/lib']
    EXTENSIONS['aec']['include_dirs'] = ['%PREFIX%/lib/libaec/include']
    EXTENSIONS['gif']['include_dirs'] = ['%PREFIX%/include/giflib5']
    EXTENSIONS['jpeg2k']['include_dirs'].extend(
        (
            '%PREFIX%/include/openjpeg-2.3',
            '%PREFIX%/include/openjpeg-2.4',
            '%PREFIX%/include/openjpeg-2.5',
        )
    )


def customize_build_mingw(EXTENSIONS, OPTIONS):
    """Customize build for mingw-w64."""

    del EXTENSIONS['brunsli']
    del EXTENSIONS['heif']
    del EXTENSIONS['jetraw']  # commercial
    del EXTENSIONS['lzfse']
    del EXTENSIONS['lzham']
    del EXTENSIONS['lzo']
    del EXTENSIONS['mozjpeg']  # Win32 only
    del EXTENSIONS['sperr']
    del EXTENSIONS['zfp']
    del EXTENSIONS['zlibng']

    EXTENSIONS['jpeg8']['sources'] = []  # use libjpeg-turbo 3
    EXTENSIONS['jpeg2k']['include_dirs'].extend(
        (
            sys.prefix + '/include/openjpeg-2.3',
            sys.prefix + '/include/openjpeg-2.4',
            sys.prefix + '/include/openjpeg-2.5',
        )
    )
    EXTENSIONS['jpegxr']['include_dirs'].append(sys.prefix + '/include/jxrlib')


if 'sdist' not in sys.argv:
    # customize builds based on environment
    try:
        from imagecodecs_distributor_setup import (  # type: ignore
            customize_build,
        )
    except ImportError:
        if os.environ.get('COMPUTERNAME', '').startswith('CG-'):
            customize_build = customize_build_cgohlke
        elif os.environ.get('IMAGECODECS_CIBW', ''):
            customize_build = customize_build_cibuildwheel
        elif os.environ.get('CONDA_BUILD', ''):
            customize_build = customize_build_condaforge
        elif shutil.which('port'):
            customize_build = customize_build_macports
        elif os.name == 'nt' and 'GCC' in sys.version:
            customize_build = customize_build_mingw
        else:
            customize_build = customize_build_default

    customize_build(EXTENSIONS, OPTIONS)


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
            if (self.lite and name not in {'imcd', 'shared'}) or getattr(
                self, f'skip_{name}', False
            ):
                print(f'skipping {ext.name!r} extension (deselected)')
                self.extensions.remove(ext)

        self.include_dirs.append(numpy.get_include())


def extension(name):
    """Return setuptools Extension."""
    opt = EXTENSIONS[name]
    sources = opt['sources']
    fname = f'imagecodecs/_{name}'
    if all(not n.startswith(fname) for n in sources):
        sources = [fname + '.pyx'] + sources
    ext = Extension(
        f'imagecodecs._{name}',
        sources=sources,
        **{
            key: (OPTIONS[key] + opt[key])
            for key in (
                'include_dirs',
                'library_dirs',
                'libraries',
                'define_macros',
                'extra_compile_args',
                'extra_link_args',
                'depends',
            )
        },
    )
    ext.cython_compile_time_env = {
        **OPTIONS['cython_compile_time_env'],  # type: ignore
        **opt['cython_compile_time_env'],
    }
    # ext.force = OPTIONS['cythonize'] or opt['cythonize']
    return ext


setup(
    name='imagecodecs',
    version=version,
    license='BSD',
    description=description,
    long_description=readme,
    long_description_content_type='text/x-rst',
    author='Christoph Gohlke',
    author_email='cgohlke@cgohlke.com',
    url='https://www.cgohlke.com',
    project_urls={
        'Bug Tracker': 'https://github.com/cgohlke/imagecodecs/issues',
        'Source Code': 'https://github.com/cgohlke/imagecodecs',
        # 'Documentation': 'https://',
    },
    python_requires='>=3.9',
    install_requires=['numpy'],
    # setup_requires=['setuptools', 'numpy', 'cython'],
    extras_require={'all': ['matplotlib', 'tifffile', 'numcodecs']},
    tests_require=[
        'pytest',
        'tifffile',
        'czifile',
        'blosc; platform_python_implementation!="PyPy"',
        'blosc2; platform_python_implementation!="PyPy"',
        'zstd',
        'lz4',
        'pyliblzfse',
        'python-lzf',
        'python-snappy',
        'bitshuffle',  # git+https://github.com/cgohlke/bitshuffle@patch-1
        'zopflipy',
        'zarr',
        'numcodecs',
        # 'bz2',
        # 'zfpy',
        # 'brotli',
        # 'deflate',
        # 'pytinyexr',
    ],
    packages=['imagecodecs'],
    package_data={'imagecodecs': ['*.pyi', 'py.typed', 'licenses/*']},
    entry_points={
        'console_scripts': ['imagecodecs=imagecodecs.__main__:main']
    },
    ext_modules=[extension(name) for name in sorted(EXTENSIONS)],
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
    platforms=['any'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: BSD License',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: C',
        'Programming Language :: Cython',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
)
