# imagecodecs/_bcn.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2023-2025, Christoph Gohlke
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""BCN and DDS codecs for the imagecodecs package."""

include '_shared.pxi'

from bcdec cimport *


class BCN:
    """BCn codec constants."""

    available = True

    class FORMAT(enum.IntEnum):
        """BCn compression format."""

        BC1 = 1  # DXT1
        BC2 = 2  # DXT3
        BC3 = 3  # DXT5
        BC4 = 4  # BC4_UNORM
        BC5 = 5  # BC5_UNORM
        BC6HU = 6  # BC6H_UF16
        BC6HS = -6  # BC6H_SF16
        BC7 = 7  # BC7_UNORM


class BcnError(RuntimeError):
    """BCn codec exceptions."""


def bcn_version():
    """Return bcdec library version string."""
    return f'bcdec {BCDEC_VERSION_MAJOR}.{BCDEC_VERSION_MINOR}'


def bcn_check(const uint8_t[::1] data):
    """Return whether data is BCn encoded."""
    return None


def bcn_encode(data, out=None):
    """Return BCn encoded data (not implemented)."""
    raise NotImplementedError('bcn_encode')


def bcn_decode(data, format, shape=None, out=None):
    """Return decoded BCn data."""
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        ssize_t srcsize = src.nbytes
        ssize_t width, height, i, ret
        char *psrc = NULL
        char *pdst = NULL
        int ndim, bcn

    if data is out:
        raise ValueError('cannot decode in-place')

    if not 128 < srcsize <= 2147483647:
        raise ValueError(f'input size {srcsize} out of bounds')

    if shape is None and hasattr(out, 'shape'):
        shape = out.shape
    elif shape is None:
        #  TODO: calculate output size from data size
        raise TypeError(
            "bcn_decode missing required argument 'shape' or 'out'"
        )
    ndim = <int> len(shape)

    bcn = format
    if bcn == 1 or bcn == 2 or bcn == 3 or bcn == 7:
        if ndim < 2 or shape[ndim - 1] != 4:
            raise ValueError(f'invalid {shape=!r} for BC{bcn}')
        width = shape[ndim - 2]
        height = 1
        for i in range(ndim - 2):
            height *= shape[i]
        dtype = numpy.uint8
    elif bcn == 4:
        if ndim < 1:
            raise ValueError(f'invalid {shape=!r} for BC{bcn}')
        width = shape[ndim - 1]
        height = 1
        for i in range(ndim - 1):
            height *= shape[i]
        dtype = numpy.uint8
    elif bcn == 5:
        if ndim < 2 or shape[ndim - 1] != 2:
            raise ValueError(f'invalid {shape=!r} for BC{bcn}')
        width = shape[ndim - 2]
        height = 1
        for i in range(ndim - 2):
            height *= shape[i]
        dtype = numpy.uint8
    elif bcn == 6 or bcn == -6:
        if ndim < 2 or shape[ndim - 1] != 3:
            raise ValueError(f'invalid {shape=!r} for BC{bcn}')
        width = shape[ndim - 2]
        height = 1
        for i in range(ndim - 2):
            height *= shape[i]
        dtype = numpy.float16
    else:
        raise BcnError(f'BC{bcn} not supported')

    if width % 4 or height % 4:
        raise ValueError(f'invalid {shape=!r} for BC{bcn}')

    out = _create_array(out, shape, dtype)
    dst = out
    pdst = <char*> dst.data
    psrc = <char*> &src[0]

    with nogil:
        ret = _bcn_decode(
            pdst,
            psrc,
            srcsize,
            width,
            height,
            bcn
        )

    if ret <= 0:
        raise BcnError(f'_bcn_decode returned {ret!r}')

    return out


cdef ssize_t _bcn_decode(
    char* pdst,
    char* psrc,
    ssize_t srcsize,
    ssize_t width,
    ssize_t height,
    int bcn,
) nogil:
    # Decode BCn encoded array.
    # TODO: move this function to C file
    cdef:
        int pitch
        ssize_t size, i, j

    height = height // 4
    size = height * width // 4

    if bcn == 1:
        # DXT1
        if srcsize < size * BCDEC_BC1_BLOCK_SIZE:
            return srcsize - size * BCDEC_BC1_BLOCK_SIZE
        pitch = <int> (width * 4)
        for j in range(height):
            i = 0
            while i < width:
                i += 4
                bcdec_bc1(psrc, pdst, pitch)
                psrc += BCDEC_BC1_BLOCK_SIZE
                pdst += 16
            pdst += pitch * 3

    elif bcn == 2:
        # DXT3
        if srcsize < size * BCDEC_BC2_BLOCK_SIZE:
            return srcsize - size * BCDEC_BC2_BLOCK_SIZE
        pitch = <int> (width * 4)
        for j in range(height):
            i = 0
            while i < width:
                i += 4
                bcdec_bc2(psrc, pdst, pitch)
                psrc += BCDEC_BC2_BLOCK_SIZE
                pdst += 16
            pdst += pitch * 3

    elif bcn == 3:
        # DXT5
        if srcsize < size * BCDEC_BC3_BLOCK_SIZE:
            return srcsize - size * BCDEC_BC3_BLOCK_SIZE
        pitch = <int> (width * 4)
        for j in range(height):
            i = 0
            while i < width:
                i += 4
                bcdec_bc3(psrc, pdst, pitch)
                psrc += BCDEC_BC3_BLOCK_SIZE
                pdst += 16
            pdst += pitch * 3

    elif bcn == 4:
        # BC4_UNORM
        if srcsize < size * BCDEC_BC4_BLOCK_SIZE:
            return srcsize - size * BCDEC_BC4_BLOCK_SIZE
        for j in range(height):
            i = 0
            while i < width:
                i += 4
                bcdec_bc4(psrc, pdst, <int> width)
                psrc += BCDEC_BC4_BLOCK_SIZE
                pdst += 4
            pdst += width * 3

    elif bcn == 5:
        # BC5_UNORM
        if srcsize < size * BCDEC_BC5_BLOCK_SIZE:
            return srcsize - size * BCDEC_BC5_BLOCK_SIZE
        pitch = <int> (width * 2)
        for j in range(height):
            i = 0
            while i < width:
                i += 4
                bcdec_bc5(psrc, pdst, pitch)
                psrc += BCDEC_BC5_BLOCK_SIZE
                pdst += 8
            pdst += pitch * 3

    elif bcn == 6:
        # BC6H_UF16
        if srcsize < size * BCDEC_BC6H_BLOCK_SIZE:
            return srcsize - size * BCDEC_BC6H_BLOCK_SIZE
        pitch = <int> (width * 3)
        for j in range(height):
            i = 0
            while i < width:
                i += 4
                bcdec_bc6h_half(
                    <float*> psrc, <float*> pdst, pitch, 0
                )
                psrc += BCDEC_BC6H_BLOCK_SIZE
                pdst += 24
            pdst += pitch * 6

    elif bcn == -6:
        # BC6H_SF16
        if srcsize < size * BCDEC_BC6H_BLOCK_SIZE:
            return srcsize - size * BCDEC_BC6H_BLOCK_SIZE
        pitch = <int> (width * 3)
        for j in range(height):
            i = 0
            while i < width:
                i += 4
                bcdec_bc6h_half(
                    <float*> psrc, <float*> pdst, pitch, 1
                )
                psrc += BCDEC_BC6H_BLOCK_SIZE
                pdst += 24
            pdst += pitch * 6

    elif bcn == 7:
        # BC7_UNORM
        if srcsize < size * BCDEC_BC7_BLOCK_SIZE:
            return srcsize - size * BCDEC_BC7_BLOCK_SIZE
        pitch = <int> (width * 4)
        for j in range(height):
            i = 0
            while i < width:
                i += 4
                bcdec_bc7(psrc, pdst, pitch)
                psrc += BCDEC_BC7_BLOCK_SIZE
                pdst += 16
            pdst += pitch * 3

    else:
        return 0

    return 1


# DDS #########################################################################

class DDS:
    """DDS codec constants."""

    available = True


class DdsError(RuntimeError):
    """DDS codec exceptions."""


dds_version = bcn_version


def dds_check(const uint8_t[::1] data):
    """Return whether data is DDS encoded."""
    cdef:
        bytes sig = bytes(data[:4])

    return sig == b'DDS '


def dds_encode(data, out=None):
    """Return DDS encoded data (not implemented)."""
    raise NotImplementedError('dds_encode')


def dds_decode(data, mipmap=0, out=None):
    """Return decoded DDS data.

    Only BCn-compressed formats are supported. Mipmaps cannot be accessed.

    """
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        ssize_t srcsize = src.nbytes
        ssize_t width, height, textures, depth, cubes, offset
        unsigned int fourcc, mipmaps
        DDS_HEADER_t* dds_header = NULL
        DDS_HEADER_DXT10_t* dx10_header = NULL
        char *psrc = NULL
        char *pdst = NULL
        ssize_t ret

    if data is out:
        raise ValueError('cannot decode in-place')

    if not 128 < srcsize <= 2147483647:
        raise ValueError(f'input size {srcsize} out of bounds')

    sig = bytes(src[:4])
    if sig != b'DDS ':
        raise DdsError(f'not a DDS data stream {sig!r}')

    offset = 4 + sizeof(DDS_HEADER_t)
    srcsize -= offset
    if srcsize < 0:
        raise DdsError('incomplete DDS data stream')

    dds_header = <DDS_HEADER_t*> &src[4]

    if not dds_header.flags & DDS_HEADER_FLAGS_TEXTURE:
        raise DdsError(f'not a DDS texture {dds_header.flags!r}')

    fourcc = dds_header.ddspf.fourCC
    width = dds_header.width
    height = dds_header.height
    cubes = 1

    depth = 1
    if dds_header.flags & DDS_HEADER_FLAGS_VOLUME:
        depth = dds_header.depth

    mipmaps = 1
    if dds_header.flags & DDS_HEADER_FLAGS_MIPMAP:
        mipmaps = dds_header.mipMapCount

    if mipmap >= mipmaps:
        raise ValueError(f'{mipmap=} out of range')

    textures = 1
    if dds_header.caps & DDS_SURFACE_FLAGS_CUBEMAP:
        if dds_header.caps2 & DDSCAPS2_CUBEMAP_ALLFACES:
            textures = 6
        else:
            textures = 0
            if dds_header.caps2 & DDSCAPS2_CUBEMAP_POSITIVEX:
                textures += 1
            if dds_header.caps2 & DDSCAPS2_CUBEMAP_NEGATIVEX:
                textures += 1
            if dds_header.caps2 & DDSCAPS2_CUBEMAP_POSITIVEY:
                textures += 1
            if dds_header.caps2 & DDSCAPS2_CUBEMAP_NEGATIVEY:
                textures += 1
            if dds_header.caps2 & DDSCAPS2_CUBEMAP_POSITIVEZ:
                textures += 1
            if dds_header.caps2 & DDSCAPS2_CUBEMAP_NEGATIVEZ:
                textures += 1
            if textures == 0:
                textures = 1

    if not (
        0 < width <= 65536 and 0 < height <= 65536 and 0 < depth <= 65536
    ):
        raise DdsError(f'{depth=}, {height=}, or {width=} out of range')

    if fourcc == DDS_FOURCC_DX10:
        offset += sizeof(DDS_HEADER_DXT10_t)
        srcsize -= sizeof(DDS_HEADER_DXT10_t)
        if srcsize < 0:
            raise DdsError('incomplete a DDS data stream')
        dx10_header = <DDS_HEADER_DXT10_t*> &src[4 + sizeof(DDS_HEADER_t)]
        fourcc = dx10_header.dxgiFormat

        if dx10_header.miscFlag & DDS_RESOURCE_MISC_TEXTURECUBE:
            textures = 6
            cubes = dx10_header.arraySize

    if fourcc == 1429291842:  # BC1U
        fourcc = DDS_FOURCC_DXT1
    elif fourcc == 1429357378:  # BC2U
        fourcc = DDS_FOURCC_DXT3
    elif fourcc == 1429422914:  # BC3U
        fourcc = DDS_FOURCC_DXT5
    elif fourcc == 1429488450:  # BC4U
        fourcc = DXGI_FORMAT_BC4_UNORM
    elif fourcc == 826889281:  # ATI1
        fourcc = DXGI_FORMAT_BC4_UNORM
    elif fourcc == 1429553986:  # BC5U
        fourcc = DXGI_FORMAT_BC5_UNORM
    elif fourcc == 843666497:  # ATI2
        fourcc = DXGI_FORMAT_BC5_UNORM
    elif fourcc == 82:  # DXGI_FORMAT_BC5_TYPELESS
        fourcc = DXGI_FORMAT_BC5_UNORM
    elif fourcc == 97:  # DXGI_FORMAT_BC7_TYPELESS
        fourcc = DXGI_FORMAT_BC7_UNORM
    elif fourcc == 99:  # DXGI_FORMAT_BC7_UNORM_SRGB
        fourcc = DXGI_FORMAT_BC7_UNORM
    # elif fourcc == 1395934018:  # BC4S
    #     fourcc = DXGI_FORMAT_BC4_SNORM
    # elif fourcc == 1395999554:  # BC5S
    #     fourcc = DXGI_FORMAT_BC5_SNORM

    if (
        fourcc == DDS_FOURCC_DXT1
        or fourcc == DDS_FOURCC_DXT3
        or fourcc == DDS_FOURCC_DXT5
        or fourcc == DXGI_FORMAT_BC7_UNORM
    ):
        shape = height, width, 4
        dtype = numpy.uint8
    elif fourcc == DXGI_FORMAT_BC4_UNORM:
        shape = height, width
        dtype = numpy.uint8
    elif fourcc == DXGI_FORMAT_BC5_UNORM:
        shape = height, width, 2  # R, G
        dtype = numpy.uint8
    elif (
        fourcc == DXGI_FORMAT_BC6H_SF16
        or fourcc == DXGI_FORMAT_BC6H_UF16
    ):
        shape = height, width, 3
        dtype = numpy.float16
    else:
        fourcc_str = int(fourcc).to_bytes(4, 'little').decode()
        raise DdsError(f'fourcc {fourcc_str!r} ({fourcc}) not supported')

    if depth > 1:
        shape = (depth,) + shape

    # TODO: support mipmap: calculate offset and reduce shape

    if textures > 1:
        shape = (textures,) + shape
    if cubes > 1:
        shape = (cubes, ) + shape

    out = _create_array(out, shape, dtype)
    dst = out
    pdst = <char*> dst.data
    psrc = <char*> &src[offset]

    with nogil:
        height = cubes * textures * depth * height
        if fourcc == DDS_FOURCC_DXT1:
            ret = _bcn_decode(pdst, psrc, srcsize, width, height, 1)
        elif fourcc == DDS_FOURCC_DXT3:
            ret = _bcn_decode(pdst, psrc, srcsize, width, height, 2)
        elif fourcc == DDS_FOURCC_DXT5:
            ret = _bcn_decode(pdst, psrc, srcsize, width, height, 3)
        elif fourcc == DXGI_FORMAT_BC4_UNORM:
            ret = _bcn_decode(pdst, psrc, srcsize, width, height, 4)
        elif fourcc == DXGI_FORMAT_BC5_UNORM:
            ret = _bcn_decode(pdst, psrc, srcsize, width, height, 5)
        elif fourcc == DXGI_FORMAT_BC6H_UF16:
            ret = _bcn_decode(pdst, psrc, srcsize, width, height, 6)
        elif fourcc == DXGI_FORMAT_BC6H_SF16:
            ret = _bcn_decode(pdst, psrc, srcsize, width, height, -6)
        elif fourcc == DXGI_FORMAT_BC7_UNORM:
            ret = _bcn_decode(pdst, psrc, srcsize, width, height, 7)

    if ret <= 0:
        raise DdsError(f'_bcn_decode returned {ret!r}')

    return out
