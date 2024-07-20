# imagecodecs/zopfli.pxd
# cython: language_level = 3

# Cython declarations for the `Zopfli 1.0.3` library.
# https://github.com/google/zopfli

cdef extern from 'zopfli.h' nogil:

    ctypedef struct ZopfliOptions:
        int verbose
        int verbose_more
        int numiterations
        int blocksplitting
        int blocksplittinglast
        int blocksplittingmax

    ctypedef enum ZopfliFormat:
        ZOPFLI_FORMAT_GZIP
        ZOPFLI_FORMAT_ZLIB
        ZOPFLI_FORMAT_DEFLATE

    void ZopfliInitOptions(
        ZopfliOptions* options
    )

    void ZopfliCompress(
        const ZopfliOptions* options,
        ZopfliFormat output_type,
        const unsigned char* src,
        size_t insize,
        unsigned char** out,
        size_t* outsize
    )


# zlib_container.h and gzip_container.h missing in Arch Linux
#
# cdef extern from 'zopfli/zlib_container.h' nogil:
#
#     void ZopfliZlibCompress(
#         const ZopfliOptions* options,
#         const unsigned char* src,
#         size_t insize,
#         unsigned char** out,
#         size_t* outsize
#     )
#
#
# cdef extern from 'zopfli/gzip_container.h' nogil:
#
#     void ZopfliGzipCompress(
#         const ZopfliOptions* options,
#         const unsigned char* src,
#         size_t insize,
#         unsigned char** out,
#         size_t* outsize
#     )


# zopflipng_lib.h missing in Debian
#
# cdef extern from 'zopflipng_lib.h' nogil:
#
#     ctypedef enum ZopfliPNGFilterStrategy:
#         kStrategyZero
#         kStrategyOne
#         kStrategyTwo
#         kStrategyThree
#         kStrategyFour
#         kStrategyMinSum
#         kStrategyEntropy
#         kStrategyPredefined
#         kStrategyBruteForce
#         kNumFilterStrategies
#
#     ctypedef struct CZopfliPNGOptions:
#         int lossy_transparent
#         int lossy_8bit
#         ZopfliPNGFilterStrategy* filter_strategies
#         int num_filter_strategies
#         int auto_filter_strategy
#         char** keepchunks
#         int num_keepchunks
#         int use_zopfli
#         int num_iterations
#         int num_iterations_large
#         int block_split_strategy
#
#     void CZopfliPNGSetDefaults(
#         CZopfliPNGOptions* png_options
#     )
#
#     int CZopfliPNGOptimize(
#         const unsigned char* origpng,
#         const size_t origpng_size,
#         const CZopfliPNGOptions* png_options,
#         int verbose,
#         unsigned char** resultpng,
#         size_t* resultpng_size
#     )
