# imagecodecs/_shared.pxd

# Shared function definitions for imagecodecs extensions.

cimport numpy
from libc.stdint cimport uint8_t, uint64_t


cdef const uint8_t[::1] _readable_input(data)

cdef const uint8_t[::1] _writable_input(data)

cdef const uint8_t[::1] _inplace_input(data)

cdef tuple _parse_output(out, ssize_t outsize=*, bint outgiven=*, outtype=*)

cdef _create_output(out, ssize_t size, const char* string=*)

cdef _return_output(out, ssize_t size, ssize_t used, bint outgiven)

cdef numpy.ndarray _create_array(
    out, tuple shape, dtype, tuple strides=*, bint zero=*, bint contig=*
)

cdef tuple _squeeze_shape(tuple shape, ssize_t ndim)

cdef _default_value(value, default, smallest, largest)

cdef _enum_value(value, enum_class, default=*)

cdef uint8_t _default_threads(numthreads)


# imagelayout_t

ctypedef uint64_t imagecaps_t

ctypedef struct imagelayout_t:
    ssize_t frames  # number of frames (product of batch dims)
    ssize_t depth  # number of slices in volumetric axis
    ssize_t height  # image height (rows)
    ssize_t width  # image width (columns)
    ssize_t samples  # total samples per pixel (color + alpha + extra)
    int sampleformat  # IC_SF_* value
    int bitspersample  # effective bits per sample
    int itemsize  # dtype size in bytes
    int photometric  # IC_PHOTO_* value
    bint planar  # True if samples axis precedes spatial axes
    uint8_t extraindex  # 0-based sample index of first extra, 0 if none
    uint8_t extratype  # IC_EXTRA_* value
    int extracount  # number of extra samples beyond photometric

cdef int _photo_samples(int photometric) noexcept nogil

cdef imagecaps_t _photo_cap(int photometric) noexcept nogil

cdef int _image_layout(
    imagecaps_t caps,
    int ndim,
    const numpy.npy_intp* shape,
    numpy.dtype dtype,
    object photometric,
    object bitspersample,
    object planar,
    object frames,
    object volumetric,
    object extrasample,
    imagelayout_t* layout,
) except -1


cdef:
    # photometric interpretation (values match TIFF PHOTOMETRIC_*)
    const int IC_PHOTO_UNSPECIFIED = -1
    const int IC_PHOTO_GRAY = 1  # PHOTOMETRIC_MINISBLACK
    const int IC_PHOTO_RGB = 2  # PHOTOMETRIC_RGB
    const int IC_PHOTO_PALETTE = 3  # PHOTOMETRIC_PALETTE
    const int IC_PHOTO_CMYK = 5  # PHOTOMETRIC_SEPARATED
    const int IC_PHOTO_YCBCR = 6  # PHOTOMETRIC_YCBCR
    const int IC_PHOTO_CIELAB = 8  # PHOTOMETRIC_CIELAB
    const int IC_PHOTO_ICCLAB = 9  # PHOTOMETRIC_ICCLAB

    # sample format (values match TIFF SAMPLEFORMAT_*)
    const int IC_SF_UINT = 1  # SAMPLEFORMAT_UINT
    const int IC_SF_SINT = 2  # SAMPLEFORMAT_INT
    const int IC_SF_FLOAT = 3  # SAMPLEFORMAT_IEEEFP
    const int IC_SF_COMPLEX = 6  # SAMPLEFORMAT_COMPLEXIEEE
    const int IC_SF_BOOL = 7

    # extrasample type (values match TIFF EXTRASAMPLE_*)
    const int IC_EXTRA_UNSPECIFIED = 0  # EXTRASAMPLE_UNSPECIFIED
    const int IC_EXTRA_ASSOCALPHA = 1  # EXTRASAMPLE_ASSOCALPHA
    const int IC_EXTRA_UNASSALPHA = 2  # EXTRASAMPLE_UNASSALPHA

    # capability flag bits (imagecaps_t bitmask)
    const imagecaps_t IC_GRAY = <imagecaps_t>(1 << 0)
    const imagecaps_t IC_RGB = <imagecaps_t>(1 << 1)
    const imagecaps_t IC_PALETTE = <imagecaps_t>(1 << 2)
    const imagecaps_t IC_CMYK = <imagecaps_t>(1 << 3)
    const imagecaps_t IC_YCBCR = <imagecaps_t>(1 << 4)
    const imagecaps_t IC_CIELAB = <imagecaps_t>(1 << 5)
    const imagecaps_t IC_ICCLAB = <imagecaps_t>(1 << 6)
    const imagecaps_t IC_FRAMES = <imagecaps_t>(1 << 8)
    const imagecaps_t IC_PLANAR = <imagecaps_t>(1 << 9)
    const imagecaps_t IC_DEPTH = <imagecaps_t>(1 << 10)
    const imagecaps_t IC_ALPHA = <imagecaps_t>(1 << 11)
    const imagecaps_t IC_EXTRA = <imagecaps_t>(1 << 12)
    const imagecaps_t IC_UINT = <imagecaps_t>(1 << 16)
    const imagecaps_t IC_SINT = <imagecaps_t>(1 << 17)
    const imagecaps_t IC_FLOAT = <imagecaps_t>(1 << 18)
    const imagecaps_t IC_COMPLEX = <imagecaps_t>(1 << 19)
    const imagecaps_t IC_BOOL = <imagecaps_t>(1 << 20)
    const imagecaps_t IC_SZ1 = <imagecaps_t>(1 << 21)
    const imagecaps_t IC_SZ2 = <imagecaps_t>(1 << 22)
    const imagecaps_t IC_SZ4 = <imagecaps_t>(1 << 23)
    const imagecaps_t IC_SZ8 = <imagecaps_t>(1 << 24)
    const imagecaps_t IC_SZ16 = <imagecaps_t>(1 << 25)
    const imagecaps_t IC_BPS = <imagecaps_t>(1 << 26)
