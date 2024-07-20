# imagecodecs/pg_lzcompress.pxd
# cython: language_level = 3

# Cython declarations for the `PostgreSQL 13.2 pg_lzcompress` library.
# https://github.com/postgres/postgres/

from libc.stdint cimport int16_t, int32_t, int

cdef extern from 'pg_lzcompress.h' nogil:

    char* PG_LZCOMPRESS_VERSION

    ctypedef int16_t int16
    ctypedef int32_t int32
    ctypedef bint bool_t

    int32 PGLZ_MAX_OUTPUT(int32)

    ctypedef struct PGLZ_Strategy:
        int32 min_input_size
        int32 max_input_size
        int32 min_comp_rate
        int32 first_success_by
        int32 match_size_good
        int32 match_size_drop

    const PGLZ_Strategy *const PGLZ_strategy_default

    const PGLZ_Strategy *const PGLZ_strategy_always

    int32 pglz_compress(
        const char *source,
        int32 slen,
        char *dest,
        const PGLZ_Strategy *strategy
    )

    int32 pglz_decompress(
        const char *source,
        int32 slen,
        char *dest,
        int32 rawsize,
        bool_t check_complete
    )

    int32 pglz_maximum_compressed_size(
        int32 rawsize,
        int32 total_compressed_size
    )
