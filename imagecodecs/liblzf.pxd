# imagecodecs/liblzf.pxd
# cython: language_level = 3

# Cython declarations for the `liblzf 3.6` library.
# http://oldhome.schmorp.de/marc/liblzf.html

cdef extern from 'lzf.h' nogil:

    int LZF_VERSION

    unsigned int lzf_compress(
        const void* const in_data,
        unsigned int in_len,
        void* out_data,
        unsigned int out_len
    )

    unsigned int lzf_decompress(
        const void* const in_data,
        unsigned int in_len,
        void* out_data,
        unsigned int out_len
    )
