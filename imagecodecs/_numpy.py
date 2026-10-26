# imagecodecs/_numpy.py

# Copyright (c) Christoph Gohlke
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE file in the project root for details.

"""NONE and NUMPY codecs."""

from __future__ import annotations

import contextlib
import io
from typing import TYPE_CHECKING

import numpy

if TYPE_CHECKING:
    from collections.abc import Buffer
    from typing import IO, Any

    from numpy.typing import ArrayLike, NDArray

__all__ = [
    'NONE',
    'NUMPY',
    'NoneError',
    'NumpyError',
    'none_check',
    'none_decode',
    'none_encode',
    'none_info',
    'none_version',
    'numpy_check',
    'numpy_decode',
    'numpy_encode',
    'numpy_info',
    'numpy_version',
]


class NONE:
    """NONE codec constants."""

    available = True
    """NONE codec is available."""


class NoneError(RuntimeError):
    """NONE codec error."""


def none_version() -> str:
    """Return empty version string."""
    return ''


def none_check(data: Any, /) -> None:
    """Return None."""
    return


def none_info(data: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """Return empty info dictionary."""
    return {'_codec': 'none'}


def none_decode(data: Any, *args: Any, **kwargs: Any) -> Any:
    """Return data unchanged."""
    return data


def none_encode(data: Any, *args: Any, **kwargs: Any) -> Any:
    """Return data unchanged."""
    return data


class NUMPY:
    """NUMPY codec constants."""

    available = True
    """NUMPY codec is available."""


class NumpyError(RuntimeError):
    """NumPy codec error."""


def numpy_version() -> str:
    """Return Numpy library version string."""
    return f'numpy {numpy.__version__}'


def numpy_check(data: Buffer, /) -> bool:
    """Return whether data is NPY or NPZ encoded or None if unknown."""
    with io.BytesIO(data) as fh:
        data = fh.read(64)
    magic = b'\x93NUMPY'
    return data.startswith(magic) or (data.startswith(b'PK') and magic in data)


def numpy_decode(
    data: Buffer,
    /,
    index: int = 0,
    *,
    out: NDArray[Any] | None = None,
    **kwargs: Any,
) -> NDArray[Any]:
    """Return decoded NPY or NPZ data."""
    with io.BytesIO(data) as fh:
        try:
            result = numpy.load(fh, **kwargs)
        except ValueError as exc:
            msg = 'not a numpy array'
            raise ValueError(msg) from exc
        if hasattr(result, 'files'):
            with contextlib.suppress(Exception):
                index = result.files[index]
            result = result[index]
    return result  # type: ignore[no-any-return]


def numpy_info(data: Buffer, /, index: int | str = 0) -> dict[str, Any]:
    """Return information about NPY or NPZ encoded array."""
    import zipfile

    from numpy.lib import format as npy

    def header(fh: IO[bytes], /) -> dict[str, Any]:
        if npy.read_magic(fh) == (1, 0):
            shape, _, dtype = npy.read_array_header_1_0(fh)
            version = 1
        else:
            shape, _, dtype = npy.read_array_header_2_0(fh)
            version = 2
        return {
            '_codec': 'numpy',
            'shape': shape,
            'dtype': dtype,
            'version': version,
        }

    with io.BytesIO(data) as fh:
        if fh.read(2) != b'PK':
            fh.seek(0)
            return header(fh) | {'npz': False}
        fh.seek(0)
        with zipfile.ZipFile(fh) as zf:
            name = (
                zf.namelist()[int(index)]
                if isinstance(index, int | numpy.integer)
                else f'{index}.npy'
            )
            with zf.open(name) as f:
                return header(f) | {'npz': True}


def numpy_encode(
    data: ArrayLike,
    /,
    level: int | None = None,
    *,
    out: int | bytearray | None = None,
) -> bytes:
    """Return NPY or NPZ encoded data."""
    with io.BytesIO() as fh:
        if level:
            numpy.savez_compressed(fh, data)
        else:
            numpy.save(fh, data)
        fh.seek(0)
        return fh.read()
