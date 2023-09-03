/* nc4var.h */

#ifndef NC4VAR_H
#define NC4VAR_H

#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>

#define NC_NAT 0
#define NC_BYTE 1
#define NC_CHAR 2
#define NC_SHORT 3
#define NC_INT 4
#define NC_LONG NC_INT

#define NC_FLOAT 5
#define NC_DOUBLE 6
#define NC_UBYTE 7
#define NC_USHORT 8
#define NC_UINT 9
#define NC_INT64 10
#define NC_UINT64 11
#define NC_STRING 12

#define X_SCHAR_MIN (-128)
#define X_SCHAR_MAX 127
#define X_UCHAR_MAX 255U
#define X_SHORT_MIN (-32768)
#define X_SHRT_MIN X_SHORT_MIN
#define X_SHORT_MAX 32767
#define X_SHRT_MAX X_SHORT_MAX
#define X_USHORT_MAX 65535U
#define X_USHRT_MAX X_USHORT_MAX
#define X_INT_MIN (-2147483647 - 1)
#define X_INT_MAX 2147483647
#define X_LONG_MIN X_INT_MIN
#define X_LONG_MAX X_INT_MAX
#define X_UINT_MAX 4294967295U
#define X_INT64_MIN (-9223372036854775807LL - 1LL)
#define X_INT64_MAX 9223372036854775807LL
#define X_UINT64_MAX 18446744073709551615ULL
#ifdef _WIN32
#define X_FLOAT_MAX 3.402823466e+38f
#else
#define X_FLOAT_MAX 3.40282347e+38f
#endif
#define X_FLOAT_MIN (-X_FLOAT_MAX)
#define X_DOUBLE_MAX 1.7976931348623157e+308
#define X_DOUBLE_MIN (-X_DOUBLE_MAX)

#define NC_FILL_BYTE ((signed char)-127)
#define NC_FILL_CHAR ((char)0)
#define NC_FILL_SHORT ((short)-32767)
#define NC_FILL_INT (-2147483647)
#define NC_FILL_FLOAT (9.9692099683868690e+36f)
#define NC_FILL_DOUBLE (9.9692099683868690e+36)
#define NC_FILL_UBYTE (255)
#define NC_FILL_USHORT (65535)
#define NC_FILL_UINT (4294967295U)
#define NC_FILL_INT64 ((long long)-9223372036854775806LL)
#define NC_FILL_UINT64 ((unsigned long long)18446744073709551614ULL)
#define NC_FILL_STRING ((char *)"")

#define NC_NOQUANTIZE 0
#define NC_QUANTIZE_BITGROOM 1
#define NC_QUANTIZE_GRANULARBR 2
#define NC_QUANTIZE_BITROUND 3

#define NC_NOERR 0
#define NC_EBADTYPE (-45)

#define NC_VERSION_MAJOR 4
#define NC_VERSION_MINOR 9
#define NC_VERSION_PATCH 2

typedef int nc_type;

int
nc4_convert_type(
    const void *src,
    void *dest,
    const nc_type src_type,
    const nc_type dest_type,
    const size_t len,
    int *range_error,
    const void *fill_value,
    int strict_nc3,
    int quantize_mode,
    int nsd);

#endif
