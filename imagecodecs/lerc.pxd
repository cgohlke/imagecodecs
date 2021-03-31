# imagecodecs/lerc.pxd
# cython: language_level = 3

# Cython declarations for the `LERC 2.2.1` library.
# https://github.com/Esri/lerc

cdef extern from 'Lerc_c_api.h':

    ctypedef unsigned int lerc_status

    lerc_status lerc_computeCompressedSize(
        const void* pData,
        unsigned int dataType,
        int nDim,
        int nCols,
        int nRows,
        int nBands,
        const unsigned char* pValidBytes,
        double maxZErr,
        unsigned int* numBytes
    ) nogil

    lerc_status lerc_encode_c 'lerc_encode'(
        const void* pData,
        unsigned int dataType,
        int nDim,
        int nCols,
        int nRows,
        int nBands,
        const unsigned char* pValidBytes,
        double maxZErr,
        unsigned char* pOutBuffer,
        unsigned int outBufferSize,
        unsigned int* nBytesWritten
    ) nogil

    lerc_status lerc_computeCompressedSizeForVersion(
        const void* pData,
        int version,
        unsigned int dataType,
        int nDim,
        int nCols,
        int nRows,
        int nBands,
        const unsigned char* pValidBytes,
        double maxZErr,
        unsigned int* numBytes
    ) nogil

    lerc_status lerc_encodeForVersion(
        const void* pData,
        int version,
        unsigned int dataType,
        int nDim,
        int nCols,
        int nRows,
        int nBands,
        const unsigned char* pValidBytes,
        double maxZErr,
        unsigned char* pOutBuffer,
        unsigned int outBufferSize,
        unsigned int* nBytesWritten
    ) nogil

    lerc_status lerc_getBlobInfo(
        const unsigned char* pLercBlob,
        unsigned int blobSize,
        unsigned int* infoArray,
        double* dataRangeArray,
        int infoArraySize,
        int dataRangeArraySize
    ) nogil

    lerc_status lerc_decode_c 'lerc_decode'(
        const unsigned char* pLercBlob,
        unsigned int blobSize,
        unsigned char* pValidBytes,
        int nDim,
        int nCols,
        int nRows,
        int nBands,
        unsigned int dataType,
        void* pData
    ) nogil

    lerc_status lerc_decodeToDouble(
        const unsigned char* pLercBlob,
        unsigned int blobSize,
        unsigned char* pValidBytes,
        int nDim,
        int nCols,
        int nRows,
        int nBands,
        double* pData
    ) nogil


# cdef extern from 'Lerc_types.h':  # C++ header

ctypedef enum ErrCode:
    Ok
    Failed
    WrongParam
    BufferTooSmall
    NaN

ctypedef enum DataType:
    dt_char
    dt_uchar
    dt_short
    dt_ushort
    dt_int
    dt_uint
    dt_float
    dt_double

ctypedef enum InfoArrOrder:
    version
    dataType
    nDim
    nCols
    nRows
    nBands
    nValidPixels
    blobSize

ctypedef enum DataRangeArrOrder:
    zMin
    zMax
    maxZErrUsed
