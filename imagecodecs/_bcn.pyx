# imagecodecs/_bcn.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: freethreading_compatible = True

# Copyright (c) 2023-2026, Christoph Gohlke
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

"""BCn (Block Compression) and DDS codecs for the imagecodecs package.

BCn refers to a family of texture compression algorithms, while DDS
(DirectDraw Surface) is a container file format using BCn compression.

"""

include '_shared.pxi'

from bcdec cimport *


class BCN:
    """BCN codec constants."""

    available = True

    class FORMAT(enum.IntEnum):
        """BCN compression format."""

        BC1 = 1  # DXT1
        BC2 = 2  # DXT3
        BC3 = 3  # DXT5
        BC4 = 4  # BC4_UNORM
        BC5 = 5  # BC5_UNORM
        BC6HU = 6  # BC6H_UF16
        BC6HS = -6  # BC6H_SF16
        BC7 = 7  # BC7_UNORM


class BcnError(RuntimeError):
    """BCN codec exceptions."""


def bcn_version():
    """Return bcdec library version string."""
    return f'bcdec {BCDEC_VERSION_MAJOR}.{BCDEC_VERSION_MINOR}'


def bcn_check(const uint8_t[::1] data not None, /):
    """Return whether data is BCN encoded or None if unknown."""
    return None if data.shape[0] > 1 else False


def bcn_encode(
    data,
    /,
    *,
    out=None,
):
    """Return BCN encoded data (not implemented)."""
    cdef:
        numpy.ndarray _src = _array_input(data, out)

    raise NotImplementedError('bcn_encode')


def bcn_decode(
    data,
    /,
    format,
    *,
    shape=None,
    out=None,
    bint _info=False,
):
    """Return decoded BCN data."""
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        ssize_t srcsize = src.shape[0]
        ssize_t width, height, images, ret
        const char* psrc = NULL
        char* pdst = NULL
        int bcn, samples

    if data is out:
        raise ValueError('cannot transform in-place')

    if not 8 <= srcsize <= INT32_MAX:
        raise ValueError(f'input size {srcsize} out of bounds')

    if shape is None and hasattr(out, 'shape'):
        shape = out.shape
    elif shape is None:
        #  TODO: calculate output size from data size
        raise TypeError(
            "bcn_decode missing required argument 'shape' or 'out'"
        )
    shape = tuple(int(i) for i in shape)

    bcn = format
    samples, dtype = _bcn_format(bcn)

    ndim = len(shape) - (1 if samples > 1 else 0)
    if (
        ndim < 1
        or (samples > 1 and shape[ndim] != samples)
        or any(i < 0 for i in shape)
    ):
        raise ValueError(f'invalid {shape=!r} for BC{bcn}')
    imwidth = shape[ndim - 1]
    imheight = shape[ndim - 2] if ndim > 1 else 1
    if imwidth * _bcn_pixelsize(bcn) > INT32_MAX:
        raise ValueError(f'invalid {shape=!r} for BC{bcn}')
    nimages = 1
    for i in shape[: max(ndim - 2, 0)]:
        nimages *= i
    size = _bcn_size(imwidth, imheight, nimages, bcn)

    if _info:
        return {
            '_codec': 'bcn',
            'shape': shape,
            'dtype': dtype,
            'format': _enum_or_int(bcn, BCN.FORMAT),
        }

    if srcsize < size:
        raise BcnError(f'input is {size - srcsize} bytes too short')

    width = imwidth
    height = imheight
    images = nimages

    out = _create_array(out, shape, dtype)
    dst = out
    pdst = <char*> dst.data
    psrc = <const char*> &src[0]

    with nogil:
        ret = _bcn_decode(pdst, psrc, srcsize, width, height, images, bcn)

    if ret <= 0:
        raise BcnError(f'_bcn_decode returned {ret!r}')

    return out


def bcn_info(
    data,
    /,
    format,
    *,
    shape=None,
):
    """Return information about BCN encoded array."""
    return bcn_decode(data, format, shape=shape, _info=True)


cdef tuple _bcn_format(int bcn):
    # Return number of samples and dtype of decoded BCN format.
    if bcn == 1 or bcn == 2 or bcn == 3 or bcn == 7:
        return 4, numpy.uint8
    if bcn == 4:
        return 1, numpy.uint8
    if bcn == 5:
        return 2, numpy.uint8
    if bcn == 6 or bcn == -6:
        return 3, numpy.float16
    raise BcnError(f'BC{bcn} not supported')


cdef object _bcn_size(width, height, images, int bcn):
    # Return size of BCN encoded images using Python integers.
    return (
        images * ((height + 3) // 4) * ((width + 3) // 4)
        * _bcn_blocksize(bcn)
    )


cdef inline ssize_t _bcn_blocksize(int bcn) noexcept nogil:
    # Return size of encoded 4x4 block or 0 if format is not supported.
    if bcn == 1:
        return BCDEC_BC1_BLOCK_SIZE
    if bcn == 2:
        return BCDEC_BC2_BLOCK_SIZE
    if bcn == 3:
        return BCDEC_BC3_BLOCK_SIZE
    if bcn == 4:
        return BCDEC_BC4_BLOCK_SIZE
    if bcn == 5:
        return BCDEC_BC5_BLOCK_SIZE
    if bcn == 6 or bcn == -6:
        return BCDEC_BC6H_BLOCK_SIZE
    if bcn == 7:
        return BCDEC_BC7_BLOCK_SIZE
    return 0


cdef inline ssize_t _bcn_pixelsize(int bcn) noexcept nogil:
    # Return size of decoded pixel in bytes or 0 if format is not supported.
    if bcn == 1 or bcn == 2 or bcn == 3 or bcn == 7:
        return 4
    if bcn == 4:
        return 1
    if bcn == 5:
        return 2
    if bcn == 6 or bcn == -6:
        return 6
    return 0


cdef inline void _bcn_decode_block(
    const char* src,
    char* dst,
    int pitch,
    int bcn,
) noexcept nogil:
    # Decode 4x4 BCN block. Pitch is in bytes.
    if bcn == 1:
        bcdec_bc1(src, dst, pitch)
    elif bcn == 2:
        bcdec_bc2(src, dst, pitch)
    elif bcn == 3:
        bcdec_bc3(src, dst, pitch)
    elif bcn == 4:
        bcdec_bc4(src, dst, pitch)
    elif bcn == 5:
        bcdec_bc5(src, dst, pitch)
    elif bcn == 6:
        bcdec_bc6h_half(src, dst, pitch // 2, 0)
    elif bcn == -6:
        bcdec_bc6h_half(src, dst, pitch // 2, 1)
    elif bcn == 7:
        bcdec_bc7(src, dst, pitch)


cdef ssize_t _bcn_decode(
    char* pdst,
    const char* psrc,
    ssize_t srcsize,
    ssize_t width,
    ssize_t height,
    ssize_t images,
    int bcn,
) noexcept nogil:
    # Decode contiguous BCN encoded images, clipping partial edge blocks.
    # Return 1 on success, negative number of missing bytes, or 0 if
    # format is not supported.
    cdef:
        uint16_t[48] block  # 4x4 pixels of up to 6 bytes, 2-byte aligned
        char* pblock = <char*> block
        char* pdstblock
        ssize_t blocksize = _bcn_blocksize(bcn)
        ssize_t pixelsize = _bcn_pixelsize(bcn)
        ssize_t rowsize = width * pixelsize
        ssize_t size, n, x, y, j, w, h

    if blocksize == 0 or pixelsize == 0:
        return 0

    size = images * ((height + 3) // 4) * ((width + 3) // 4) * blocksize
    if srcsize < size:
        return srcsize - size

    for n in range(images):
        for y in range(0, height, 4):
            h = min(height - y, 4)
            for x in range(0, width, 4):
                w = min(width - x, 4)
                pdstblock = pdst + (n * height + y) * rowsize + x * pixelsize
                if w == 4 and h == 4:
                    _bcn_decode_block(psrc, pdstblock, <int> rowsize, bcn)
                else:
                    _bcn_decode_block(
                        psrc, pblock, <int> (4 * pixelsize), bcn
                    )
                    for j in range(h):
                        memcpy(
                            pdstblock + j * rowsize,
                            pblock + j * 4 * pixelsize,
                            w * pixelsize,
                        )
                psrc += blocksize

    return 1


# DDS #########################################################################

class DDS:
    """DDS codec constants."""

    available = True

    class FOURCC(enum.IntEnum):

        DDS = DDS_FOURCC_DDS
        DXT1 = DDS_FOURCC_DXT1
        DXT3 = DDS_FOURCC_DXT3
        DXT5 = DDS_FOURCC_DXT5
        DX10 = DDS_FOURCC_DX10
        BC4_UNORM = DXGI_FORMAT_BC4_UNORM
        BC5_UNORM = DXGI_FORMAT_BC5_UNORM
        BC6H_UF16 = DXGI_FORMAT_BC6H_UF16
        BC6H_SF16 = DXGI_FORMAT_BC6H_SF16
        BC7_UNORM = DXGI_FORMAT_BC7_UNORM


class DdsError(RuntimeError):
    """DDS codec exceptions."""


dds_version = bcn_version


def dds_check(const uint8_t[::1] data not None, /):
    """Return whether data is DDS encoded or None if unknown."""
    return _startswith(data, b'DDS ')


def dds_encode(
    data,
    /,
    *,
    out=None,
):
    """Return DDS encoded data (not implemented)."""
    cdef:
        numpy.ndarray _src = _array_input(data, out)

    raise NotImplementedError('dds_encode')


def dds_decode(
    data,
    /,
    *,
    mipmap=0,
    out=None,
    bint _info=False,
):
    """Return decoded DDS data.

    Supported:

    - BCn block-compressed formats specified via legacy FourCC
      (DXT1, DXT3, DXT5, BC1U-BC5U, ATI1, ATI2).
    - BCn block-compressed formats specified via DX10 header
      (BC1-BC5 UNORM/TYPELESS/SRGB, BC6H UF16/SF16, BC7 UNORM/SRGB).
    - Volume textures, cubemaps, and texture arrays.
    - First mipmap level only.

    Not supported:

    - Uncompressed formats (RGB, luminance, alpha, or DXGI non-BCn).
    - DXT2 and DXT4 (premultiplied alpha).
    - Signed BC4 and BC5.
    - Other FourCC formats.

    """
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        ssize_t srcsize = src.shape[0]
        ssize_t width, height, depth, textures, offset, i
        ssize_t nfaces, imagesize, facestride, dstfacesize
        unsigned int fourcc, mipmaps, arraysize, caps2, cubeface
        int bcn, samples
        DDS_HEADER_t dds_header
        DDS_HEADER_DXT10_t dx10_header
        const char* psrc = NULL
        char* pdst = NULL
        ssize_t ret = 0

    if data is out:
        raise ValueError('cannot transform in-place')

    if not 128 < srcsize <= INT32_MAX:
        raise ValueError(f'input size {srcsize} out of bounds')

    if not _startswith(src, b'DDS '):
        raise DdsError(f'not a DDS data stream {bytes(src[:4])!r}')

    offset = 4 + sizeof(DDS_HEADER_t)
    # copy header to avoid unaligned access
    memcpy(&dds_header, &src[4], sizeof(DDS_HEADER_t))

    # per specification, do not rely on DDSD_CAPS, DDSD_PIXELFORMAT,
    # and DDSD_MIPMAPCOUNT flags
    if not dds_header.flags & DDS_HEADER_FLAGS_TEXTURE:
        raise DdsError(f'not a DDS texture {dds_header.flags!r}')

    fourcc = dds_header.ddspf.fourCC
    width = dds_header.width
    height = dds_header.height
    caps2 = dds_header.caps2
    arraysize = 1

    depth = 1
    if dds_header.flags & DDS_HEADER_FLAGS_VOLUME:
        depth = dds_header.depth

    mipmaps = max(dds_header.mipMapCount, <unsigned int> 1)

    if not 0 <= mipmap < mipmaps:
        raise ValueError(f'{mipmap=} out of range')
    if mipmap != 0:
        raise NotImplementedError(f'{mipmap=} not supported')

    textures = 1
    if caps2 & DDSCAPS2_CUBEMAP:
        textures = 0
        for cubeface in (
            DDSCAPS2_CUBEMAP_POSITIVEX,
            DDSCAPS2_CUBEMAP_NEGATIVEX,
            DDSCAPS2_CUBEMAP_POSITIVEY,
            DDSCAPS2_CUBEMAP_NEGATIVEY,
            DDSCAPS2_CUBEMAP_POSITIVEZ,
            DDSCAPS2_CUBEMAP_NEGATIVEZ,
        ):
            # face constants include the DDSCAPS2_CUBEMAP bit
            if (caps2 & cubeface) == cubeface:
                textures += 1
        if textures == 0:
            textures = 1

    if not (
        0 < width <= 65536 and 0 < height <= 65536 and 0 < depth <= 65536
    ):
        raise DdsError(f'{depth=}, {height=}, or {width=} out of range')

    if fourcc == DDS_FOURCC_DX10:
        if srcsize < offset + <ssize_t> sizeof(DDS_HEADER_DXT10_t):
            raise DdsError('incomplete DDS data stream')
        memcpy(&dx10_header, &src[offset], sizeof(DDS_HEADER_DXT10_t))
        offset += sizeof(DDS_HEADER_DXT10_t)
        fourcc = dx10_header.dxgiFormat
        arraysize = dx10_header.arraySize
        if arraysize == 0:
            raise DdsError(f'invalid {arraysize=}')
        if dx10_header.miscFlag & DDS_RESOURCE_MISC_TEXTURECUBE:
            textures = 6
        else:
            textures = 1

    if fourcc in (1429291842, 70, 71, 72):
        # BC1U, DXGI_FORMAT_BC1_TYPELESS, _UNORM, _UNORM_SRGB
        fourcc = DDS_FOURCC_DXT1
    elif fourcc in (1429357378, 73, 74, 75):
        # BC2U, DXGI_FORMAT_BC2_TYPELESS, _UNORM, _UNORM_SRGB
        fourcc = DDS_FOURCC_DXT3
    elif fourcc in (1429422914, 76, 77, 78):
        # BC3U, DXGI_FORMAT_BC3_TYPELESS, _UNORM, _UNORM_SRGB
        fourcc = DDS_FOURCC_DXT5
    elif fourcc in (1429488450, 826889281, 79):
        # BC4U, ATI1, DXGI_FORMAT_BC4_TYPELESS
        fourcc = DXGI_FORMAT_BC4_UNORM
    elif fourcc in (1429553986, 843666497, 82):
        # BC5U, ATI2, DXGI_FORMAT_BC5_TYPELESS
        fourcc = DXGI_FORMAT_BC5_UNORM
    elif fourcc == 94:
        # DXGI_FORMAT_BC6H_TYPELESS
        fourcc = DXGI_FORMAT_BC6H_UF16
    elif fourcc in (97, 99):
        # DXGI_FORMAT_BC7_TYPELESS, _UNORM_SRGB
        fourcc = DXGI_FORMAT_BC7_UNORM
    # elif fourcc == 1395934018:  # BC4S
    #     fourcc = DXGI_FORMAT_BC4_SNORM
    # elif fourcc == 1395999554:  # BC5S
    #     fourcc = DXGI_FORMAT_BC5_SNORM

    if fourcc == DDS_FOURCC_DXT1:
        bcn = 1
    elif fourcc == DDS_FOURCC_DXT3:
        bcn = 2
    elif fourcc == DDS_FOURCC_DXT5:
        bcn = 3
    elif fourcc == DXGI_FORMAT_BC4_UNORM:
        bcn = 4
    elif fourcc == DXGI_FORMAT_BC5_UNORM:
        bcn = 5
    elif fourcc == DXGI_FORMAT_BC6H_UF16:
        bcn = 6
    elif fourcc == DXGI_FORMAT_BC6H_SF16:
        bcn = -6
    elif fourcc == DXGI_FORMAT_BC7_UNORM:
        bcn = 7
    else:
        fourcc_str = int(fourcc).to_bytes(4, 'little').decode('latin-1')
        raise DdsError(f'fourcc {fourcc_str!r} ({fourcc}) not supported')

    samples, dtype = _bcn_format(bcn)
    if samples == 1:
        shape = height, width
    else:
        shape = height, width, samples
    if depth > 1:
        shape = (depth, *shape)

    # TODO: support mipmap: calculate offset and reduce shape

    if textures > 1:
        shape = (textures, *shape)
    if arraysize > 1:
        shape = (arraysize, *shape)

    if _info:
        return {
            '_codec': 'dds',
            'shape': shape,
            'dtype': dtype,
            'fourcc': _enum_or_int(fourcc, DDS.FOURCC),
        }

    # each face or array element is followed by its mipmap chain
    size = _bcn_size(width, height, depth, bcn)
    stride = size
    if mipmaps > 1 and arraysize * textures > 1:
        if mipmaps > int(max(width, height, depth)).bit_length():
            raise DdsError(f'invalid mipMapCount {mipmaps}')
        stride = 0
        for i in range(<ssize_t> mipmaps):
            stride += _bcn_size(
                max(width >> i, 1),
                max(height >> i, 1),
                max(depth >> i, 1),
                bcn
            )
    required = offset + (int(arraysize) * textures - 1) * stride + size
    if srcsize < required:
        raise DdsError(
            f'incomplete DDS data stream, {required - srcsize} bytes missing'
        )

    nfaces = arraysize * textures
    imagesize = size
    facestride = stride
    dstfacesize = depth * height * width * _bcn_pixelsize(bcn)

    out = _create_array(out, shape, dtype)
    dst = out
    pdst = <char*> dst.data
    psrc = <const char*> &src[offset]

    with nogil:
        for i in range(nfaces):
            ret = _bcn_decode(
                pdst + i * dstfacesize,
                psrc + i * facestride,
                imagesize,
                width,
                height,
                depth,
                bcn,
            )
            if ret <= 0:
                break

    if ret <= 0:
        raise DdsError(f'_bcn_decode returned {ret!r}')

    return out


def dds_info(
    data,
    /,
    *,
    mipmap=0,
):
    """Return information about DDS image."""
    return dds_decode(data, mipmap=mipmap, _info=True)
