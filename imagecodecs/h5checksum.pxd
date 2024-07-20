# imagecodecs/h5checksum.pxd
# cython: language_level = 3

# Cython declarations for HDF5's `h5checksum.c`.
# https://github.com/HDFGroup/hdf5/blob/develop/src/H5checksum.c

from libc.stdint cimport uint32_t

cdef extern from 'h5checksum.h' nogil:

    int H5_VERS_MAJOR
    int H5_VERS_MINOR
    int H5_VERS_RELEASE

    uint32_t H5_checksum_fletcher32(
        const void *data, size_t len
    )

    uint32_t H5_checksum_crc(
        const void *data, size_t len
    )

    uint32_t H5_checksum_lookup3(
        const void *data,
        size_t len,
        uint32_t initval
    )

    uint32_t H5_checksum_metadata(
        const void *data,
        size_t len,
        uint32_t initval
    )

    uint32_t H5_hash_string(
        const char *str
    )
