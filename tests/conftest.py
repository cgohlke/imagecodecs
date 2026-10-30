# imagecodecs/tests/conftest.py

# Copyright (c) Christoph Gohlke
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE file in the project root for details.

"""Pytest configuration."""

from __future__ import annotations

import glob
import mmap
import os
import pathlib
import sys
from typing import Any

import numpy

collect_ignore = ['test_corpus.py', 'data']

DATA_PATH = pathlib.Path(os.path.dirname(__file__)) / 'data'

if os.environ.get('VSCODE_CWD'):
    # work around pytest not using PYTHONPATH in VSCode
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    )


def pytest_report_header(config: object) -> str:
    """Return pytest report header."""
    try:
        import imagecodecs
        from imagecodecs import _imagecodecs

        return (
            f'Python {sys.version.splitlines()[0]}\n'
            f'packagedir: {imagecodecs.__path__[0]}\n'
            f'version: {imagecodecs.version()}\n'
            f'dependencies: {_imagecodecs.version()}'
        )
    except Exception as exc:
        return f'pytest_report_header failed: {exc!s}'


def readfile(filename: str, /, *, memmap: bool = False) -> Any:
    """Return content of data file."""
    data: Any
    with open(datafiles(filename), 'rb') as fh:
        if memmap:
            data = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        else:
            data = fh.read()
    return data


def datafiles(pathname: str, base: str | None = None) -> Any:
    """Return path to data file(s)."""
    if base is None:
        base = str(DATA_PATH)
    path = os.path.join(base, *pathname.split('/'))
    if any(i in path for i in '*?'):
        return glob.glob(path)
    return path


def image_data(
    itype: str,
    /,
    dtype: numpy.typing.DTypeLike,
    *,
    planar: bool = False,
    frames: bool = False,
) -> numpy.ndarray[Any, Any]:
    """Return test image array."""
    data = TEST_DATA if frames else TEST_DATA[0]

    match itype:
        case 'rgb' | 'view':
            data = data[..., [0, 2, 4]]
        case 'rgba':
            data = data[..., [0, 2, 4, -1]]
        case 'cmyk':
            data = data[..., [0, 2, 4, 6]]
        case 'cmyka':
            data = data[..., [0, 2, 4, 6, -1]]
        case 'gray':
            data = data[..., 0:1]
        case 'graya':
            data = data[..., [0, -1]]
        case 'channels':
            data = data[..., :-1]
        case 'channelsa':
            data = data[..., :]
        case 'line':
            data = data[0:1, :, 0:1]
        case 'stack':
            # TODO: remove this
            assert not frames
            assert not planar
            data = numpy.moveaxis(data, -1, 0)
        case _:
            msg = f'{itype=} not found'
            raise ValueError(msg)

    if planar:
        data = numpy.moveaxis(data, -1, -3)

    data = data.copy()

    dtype = numpy.dtype(dtype)
    if dtype.char == '?':
        data = data > data.mean()
    elif dtype.kind in 'iu':
        iinfo = numpy.iinfo(dtype)
        if dtype.kind == 'u':
            data *= iinfo.max + 1
        else:
            data *= (iinfo.max - iinfo.min) / 2
            data -= 0.5
        data = numpy.rint(data)
        data = numpy.clip(data, iinfo.min, iinfo.max)
    elif dtype.kind != 'f':
        msg = f'{dtype=} not supported'
        raise NotImplementedError(msg)

    data = data.astype(dtype)

    if dtype == 'uint16':
        # 12-bit
        data //= 16

    if itype == 'view':
        assert not frames
        assert not planar
        shape = data.shape
        temp = numpy.zeros((shape[0] + 5, shape[1] + 5, shape[2]), dtype)
        temp[2 : 2 + shape[0], 3 : 3 + shape[1], :] = data
        data = temp[2 : 2 + shape[0], 3 : 3 + shape[1], :]

    return numpy.asarray(data)


# (32, 31, 9) float64
TEST_DATA: numpy.ndarray[Any, Any] = numpy.load(datafiles('testdata.npy'))
