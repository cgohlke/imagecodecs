# imagecodecs/typing.py

# Copyright (c) Christoph Gohlke
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE file in the project root for details.

"""Type aliases for annotating imagecodecs function arguments and results."""

__all__ = [
    'ArrayDecoder',
    'ArrayEncoder',
    'ArrayFilter',
    'ByteOrder',
    'BytesDecoder',
    'BytesEncoder',
    'BytesOut',
    'BytesResult',
    'CodecCheck',
    'CodecInfo',
    'CodecVersion',
    'EnumLike',
]

import enum
from collections.abc import Buffer, Callable
from typing import Any, Concatenate, Literal

from numpy.typing import ArrayLike, NDArray

type BytesOut = int | type[bytes | bytearray] | bytearray | memoryview
"""Output argument of functions returning bytes-like objects.

- int: size of new bytes object to return.
- bytes or bytearray type: type of new object to return.
- bytearray or memoryview: writable buffer to write to.

"""

type BytesResult = bytes | bytearray | memoryview
"""Bytes-like object returned by functions accepting BytesOut."""

type EnumLike[E: enum.Enum] = E | int | str
"""Enum member, its integer value, or its case-insensitive name."""

type ByteOrder = Literal['>', '<', '=']
"""Byte order: big-endian, little-endian, or native."""

type CodecVersion = Callable[[], str]
"""Function returning codec library version, for example, zlib_version."""

type CodecCheck = Callable[[Buffer], bool | None]
"""Function returning whether data is in codec format, like png_check."""

type CodecInfo = Callable[Concatenate[Buffer, ...], dict[str, Any]]
"""Function returning information about encoded data, like png_info."""

type BytesEncoder = Callable[Concatenate[Buffer, ...], BytesResult]
"""Function encoding bytes to bytes, for example, zlib_encode."""

type BytesDecoder = Callable[Concatenate[Buffer, ...], BytesResult]
"""Function decoding bytes to bytes, for example, zlib_decode."""

type ArrayEncoder = Callable[Concatenate[ArrayLike, ...], BytesResult]
"""Function encoding array to bytes, for example, png_encode."""

type ArrayDecoder = Callable[Concatenate[Buffer, ...], NDArray[Any]]
"""Function decoding bytes to array, for example, png_decode."""

type ArrayFilter = Callable[Concatenate[NDArray[Any], ...], NDArray[Any]]
"""Function transforming array to array, for example, delta_encode."""
