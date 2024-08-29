# imagecodecs/sperr.pxd
# cython: language_level = 3

# Cython declarations for the `SPERR 0.8.2` library.
# https://github.com/NCAR/SPERR

cdef extern from 'SPERR_C_API.h' nogil:

    int SPERR_VERSION_MAJOR
    int SPERR_VERSION_MINOR
    int SPERR_VERSION_PATCH

    int sperr_comp_2d(
        const void* src,
        int is_float,
        size_t dimx,
        size_t dimy,
        int mode,
        double quality,
        int out_inc_header,
        void** dst,
        size_t* dst_len
    )

    int sperr_decomp_2d(
        const void* src,
        size_t src_len,
        int output_float,
        size_t dimx,
        size_t dimy,
        void** dst
    )

    void sperr_parse_header(
        const void* src,
        size_t* dimx,
        size_t* dimy,
        size_t* dimz,
        int* is_float
    )

    int sperr_comp_3d(
        const void* src,
        int is_float,
        size_t dimx,
        size_t dimy,
        size_t dimz,
        size_t chunk_x,
        size_t chunk_y,
        size_t chunk_z,
        int mode,
        double quality,
        size_t nthreads,
        void** dst,
        size_t* dst_len
    )

    int sperr_decomp_3d(
        const void* src,
        size_t src_len,
        int output_float,
        size_t nthreads,
        size_t* dimx,
        size_t* dimy,
        size_t* dimz,
        void** dst
    )

    int sperr_trunc_3d(
        const void* src,
        size_t src_len,
        unsigned int pct,
        void** dst,
        size_t* dst_len
    )
