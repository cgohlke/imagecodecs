# imagecodecs/lerc.pxd
# cython: language_level = 3

# Cython declarations for the `LERC 4.0.0` library.
# https://github.com/Esri/lerc

cdef extern from 'Lerc_c_api.h' nogil:

    int LERC_VERSION_MAJOR
    int LERC_VERSION_MINOR
    int LERC_VERSION_PATCH
    int LERC_VERSION_NUMBER

    bint LERC_AT_LEAST_VERSION(int, int, int)

    ctypedef unsigned int lerc_status

    lerc_status lerc_computeCompressedSize(
        const void* pData,
        unsigned int dataType,
        int nDepth,
        int nCols,
        int nRows,
        int nBands,
        int nMasks,
        const unsigned char* pValidBytes,
        double maxZErr,
        unsigned int* numBytes
    )

    lerc_status lerc_encode_c 'lerc_encode'(
        const void* pData,
        unsigned int dataType,
        int nDepth,
        int nCols,
        int nRows,
        int nBands,
        int nMasks,
        const unsigned char* pValidBytes,
        double maxZErr,
        unsigned char* pOutBuffer,
        unsigned int outBufferSize,
        unsigned int* nBytesWritten
    )

    lerc_status lerc_computeCompressedSizeForVersion(
        const void* pData,
        int codecVersion,
        unsigned int dataType,
        int nDepth,
        int nCols,
        int nRows,
        int nBands,
        int nMasks,
        const unsigned char* pValidBytes,
        double maxZErr,
        unsigned int* numBytes
    )

    lerc_status lerc_encodeForVersion(
        const void* pData,
        int codecVersion,
        unsigned int dataType,
        int nDepth,
        int nCols,
        int nRows,
        int nBands,
        int nMasks,
        const unsigned char* pValidBytes,
        double maxZErr,
        unsigned char* pOutBuffer,
        unsigned int outBufferSize,
        unsigned int* nBytesWritten
    )

    lerc_status lerc_getBlobInfo(
        const unsigned char* pLercBlob,
        unsigned int blobSize,
        unsigned int* infoArray,
        double* dataRangeArray,
        int infoArraySize,
        int dataRangeArraySize
    )

    lerc_status lerc_getDataRanges(
        const unsigned char* pLercBlob,
        unsigned int blobSize,
        int nDepth,
        int nBands,
        double* pMins,
        double* pMaxs
    )

    lerc_status lerc_decode_c 'lerc_decode'(
        const unsigned char* pLercBlob,
        unsigned int blobSize,
        int nMasks,
        unsigned char* pValidBytes,
        int nDepth,
        int nCols,
        int nRows,
        int nBands,
        unsigned int dataType,
        void* pData
    )

    lerc_status lerc_decodeToDouble(
        const unsigned char* pLercBlob,
        unsigned int blobSize,
        int nMasks,
        unsigned char* pValidBytes,
        int nDepth,
        int nCols,
        int nRows,
        int nBands,
        double* pData
    )

    lerc_status lerc_computeCompressedSize_4D(
        const void* pData,
        unsigned int dataType,
        int nDepth,
        int nCols,
        int nRows,
        int nBands,
        int nMasks,
        const unsigned char* pValidBytes,
        double maxZErr,
        unsigned int* numBytes,
        const unsigned char* pUsesNoData,
        const double* noDataValues
    )

    lerc_status lerc_encode_4D(
        const void* pData,
        unsigned int dataType,
        int nDepth,
        int nCols,
        int nRows,
        int nBands,
        int nMasks,
        const unsigned char* pValidBytes,
        double maxZErr,
        unsigned char* pOutBuffer,
        unsigned int outBufferSize,
        unsigned int* nBytesWritten,
        const unsigned char* pUsesNoData,
        const double* noDataValues
    )

    lerc_status lerc_decode_4D(
        const unsigned char* pLercBlob,
        unsigned int blobSize,
        int nMasks,
        unsigned char* pValidBytes,
        int nDepth,
        int nCols,
        int nRows,
        int nBands,
        unsigned int dataType,
        void* pData,
        unsigned char* pUsesNoData,
        double* noDataValues
    )

    lerc_status lerc_decodeToDouble_4D(
        const unsigned char* pLercBlob,
        unsigned int blobSize,
        int nMasks,
        unsigned char* pValidBytes,
        int nDepth,
        int nCols,
        int nRows,
        int nBands,
        double* pData,
        unsigned char* pUsesNoData,
        double* noDataValues
    )


# cdef extern from 'Lerc_types.h' nogil:  # C++ header

ctypedef enum ErrCode:
    Ok
    Failed
    WrongParam
    BufferTooSmall
    NaN
    HasNoData

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
    nMasks
    nDepth
    nUsesNoDataValue

ctypedef enum DataRangeArrOrder:
    zMin
    zMax
    maxZErrUsed
