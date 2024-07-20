# imagecodecs/nc4var.pxd
# cython: language_level = 3

# Cython declarations for netcdf4-c's `nc4var.c`.
# https://github.com/Unidata/netcdf-c/blob/main/libsrc4/nc4var.c

from libc.stdint cimport uint32_t

cdef extern from 'nc4var.h' nogil:

    int NC_NAT
    int NC_BYTE
    int NC_CHAR
    int NC_SHORT
    int NC_INT
    int NC_LONG
    int NC_FLOAT
    int NC_DOUBLE
    int NC_UBYTE
    int NC_USHORT
    int NC_UINT
    int NC_INT64
    int NC_UINT64
    int NC_STRING

    int NC_FILL_BYTE
    int NC_FILL_CHAR
    int NC_FILL_SHORT
    int NC_FILL_INT
    int NC_FILL_FLOAT
    int NC_FILL_DOUBLE
    int NC_FILL_UBYTE
    int NC_FILL_USHORT
    int NC_FILL_UINT
    int NC_FILL_INT64
    int NC_FILL_UINT64
    int NC_FILL_STRING

    int NC_NOQUANTIZE
    int NC_QUANTIZE_BITGROOM
    int NC_QUANTIZE_GRANULARBR
    int NC_QUANTIZE_BITROUND

    int NC_NOERR
    int NC_EBADTYPE

    int NC_VERSION_MAJOR
    int NC_VERSION_MINOR
    int NC_VERSION_PATCH

    ctypedef int nc_type

    int nc4_convert_type(
        const void *src,
        void *dest,
        const nc_type src_type,
        const nc_type dest_type,
        const size_t len,
        int *range_error,
        const void *fill_value,
        int strict_nc3,
        int quantize_mode,
        int nsd
    )
