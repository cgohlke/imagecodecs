/* Copyright 2003-2018, University Corporation for Atmospheric
 * Research. See COPYRIGHT file for copying and redistribution
 * conditions.*/
/**
 * @file
 * @internal This file is part of netcdf-4, a netCDF-like interface
 * for HDF5, or a HDF5 backend for netCDF, depending on your point of
 * view. This file handles the NetCDF-4 variable functions.
 *
 * @author Ed Hartnett, Dennis Heimbigner, Ward Fisher
 */

/* Lint, header, remove LOG only keep nc4_convert_type function */

#include "nc4var.h"
#include <math.h>

/* Define log_e for 10 and 2. Prefer constants defined in math.h,
 * however, GCC environments can have hard time defining M_LN10/M_LN2
 * despite finding math.h */
#ifndef M_LN10
#define M_LN10 2.30258509299404568402 /**< log_e 10 */
#endif /* M_LN10 */
#ifndef M_LN2
#define M_LN2 0.69314718055994530942 /**< log_e 2 */
#endif /* M_LN2 */

/** Used in quantize code. Number of explicit bits in significand for
 * floats. Bits 0-22 of SP significands are explicit. Bit 23 is
 * implicitly 1. Currently redundant with NC_QUANTIZE_MAX_FLOAT_NSB
 * and with limits.h/climit (FLT_MANT_DIG-1) */
#define BIT_XPL_NBR_SGN_FLT (23)

/** Used in quantize code. Number of explicit bits in significand for
 * doubles. Bits 0-51 of DP significands are explicit. Bit 52 is
 * implicitly 1. Currently redundant with NC_QUANTIZE_MAX_DOUBLE_NSB
 * and with limits.h/climit (DBL_MANT_DIG-1) */
#define BIT_XPL_NBR_SGN_DBL (52)

/** Pointer union for floating point and bitmask types. */
typedef union { /* ptr_unn */
    float *fp;
    double *dp;
    unsigned int *ui32p;
    unsigned long long *ui64p;
    void *vp;
} ptr_unn;

/**
 * @internal Copy data from one buffer to another, performing
 * appropriate data conversion.
 *
 * This function will copy data from one buffer to another, in
 * accordance with the types. Range errors will be noted, and the fill
 * value used (or the default fill value if none is supplied) for
 * values that overflow the type.
 *
 * This function applies quantization to float and double data, if
 * desired. The code to do this is derived from the corresponding
 * filter in the CCR project (e.g.,
 * https://github.com/ccr/ccr/blob/master/hdf5_plugins/BITGROOM/src/H5Zbitgroom.c).
 *
 * @param src Pointer to source of data.
 * @param dest Pointer that gets data.
 * @param src_type Type ID of source data.
 * @param dest_type Type ID of destination data.
 * @param len Number of elements of data to copy.
 * @param range_error Pointer that gets 1 if there was a range error.
 * @param fill_value The fill value.
 * @param strict_nc3 Non-zero if strict model in effect.
 * @param quantize_mode May be ::NC_NOQUANTIZE, ::NC_QUANTIZE_BITGROOM,
 * ::NC_QUANTIZE_GRANULARBR, or ::NC_QUANTIZE_BITROUND.
 * @param nsd Number of significant digits for quantize. Ignored
 * unless quantize_mode is ::NC_QUANTIZE_BITGROOM,
 * ::NC_QUANTIZE_GRANULARBR, or ::NC_QUANTIZE_BITROUND
 *
 * @returns ::NC_NOERR No error.
 * @returns ::NC_EBADTYPE Type not found.
 * @author Ed Hartnett, Dennis Heimbigner
 */
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
    int nsd)
{
    /* These vars are used with quantize feature. */
    const double bit_per_dgt = M_LN10 / M_LN2; /* 3.32 [frc] Bits per decimal digit of precision  = log2(10) */
    const double dgt_per_bit = M_LN2 / M_LN10; /* 0.301 [frc] Decimal digits per bit of precision = log10(2) */
    double mnt; /* [frc] Mantissa, 0.5 <= mnt < 1.0 */
    double mnt_fabs; /* [frc] fabs(mantissa) */
    double mnt_log10_fabs; /* [frc] log10(fabs(mantissa))) */
    double val; /* [frc] Copy of input value to avoid indirection */
    double mss_val_cmp_dbl; /* Missing value for comparison to double precision values */
    float mss_val_cmp_flt; /* Missing value for comparison to single precision values */
    int bit_xpl_nbr_zro; /* [nbr] Number of explicit bits to zero */
    int dgt_nbr; /* [nbr] Number of digits before decimal point */
    int qnt_pwr; /* [nbr] Power of two in quantization mask: qnt_msk = 2^qnt_pwr */
    int xpn_bs2; /* [nbr] Binary exponent xpn_bs2 in val = sign(val) * 2^xpn_bs2 * mnt, 0.5 < mnt <= 1.0 */
    size_t idx;
    unsigned int *u32_ptr;
    unsigned int msk_f32_u32_zro;
    unsigned int msk_f32_u32_one;
    unsigned int msk_f32_u32_hshv;
    unsigned long long int *u64_ptr;
    unsigned long long int msk_f64_u64_zro;
    unsigned long long int msk_f64_u64_one;
    unsigned long long int msk_f64_u64_hshv;
    unsigned short prc_bnr_xpl_rqr; /* [nbr] Explicitly represented binary digits required to retain */
    ptr_unn op1; /* I/O [frc] Values to quantize */

    char *cp, *cp1;
    float *fp, *fp1;
    double *dp, *dp1;
    int *ip, *ip1;
    short *sp, *sp1;
    signed char *bp, *bp1;
    unsigned char *ubp, *ubp1;
    unsigned short *usp, *usp1;
    unsigned int *uip, *uip1;
    long long *lip, *lip1;
    unsigned long long *ulip, *ulip1;
    size_t count = 0;

    *range_error = 0;

    /* If quantize is in use, set up some values. Quantize can only be
     * used when the destination type is NC_FLOAT or NC_DOUBLE. */
    if (quantize_mode != NC_NOQUANTIZE) {
        if ((dest_type != NC_FLOAT) && (dest_type != NC_DOUBLE))
            return NC_EBADTYPE;

        /* Parameters shared by all quantization codecs */
        if (dest_type == NC_FLOAT) {
            /* Determine the fill value. */
            if (fill_value)
                mss_val_cmp_flt = *(float *)fill_value;
            else
                mss_val_cmp_flt = NC_FILL_FLOAT;

        } else {
            /* Determine the fill value. */
            if (fill_value)
                mss_val_cmp_dbl = *(double *)fill_value;
            else
                mss_val_cmp_dbl = NC_FILL_DOUBLE;
        }

        /* Set parameters used by BitGroom and BitRound here, outside value loop.
           Equivalent parameters used by GranularBR are set inside value loop,
           since keep bits and thus masks can change for every value. */
        if (quantize_mode == NC_QUANTIZE_BITGROOM || quantize_mode == NC_QUANTIZE_BITROUND) {
            if (quantize_mode == NC_QUANTIZE_BITGROOM) {
                /* BitGroom interprets nsd as number of significant decimal digits
                 * Must convert that to number of significant bits to preserve
                 * How many bits to preserve? Being conservative, we round up the
                 * exact binary digits of precision. Add one because the first bit
                 * is implicit not explicit but corner cases prevent our taking
                 * advantage of this. */
                prc_bnr_xpl_rqr = (unsigned short)ceil(nsd * bit_per_dgt) + 1;

            } else if (quantize_mode == NC_QUANTIZE_BITROUND) {
                /* BitRound interprets nsd as number of significant binary digits (bits) */
                prc_bnr_xpl_rqr = (unsigned short)nsd;
            }

            if (dest_type == NC_FLOAT) {
                bit_xpl_nbr_zro = BIT_XPL_NBR_SGN_FLT - prc_bnr_xpl_rqr;

                /* Create mask */
                msk_f32_u32_zro = 0u; /* Zero all bits */
                msk_f32_u32_zro = ~msk_f32_u32_zro; /* Turn all bits to ones */

                /* BitShave mask for AND: Left shift zeros into bits to be
                 * rounded, leave ones in untouched bits. */
                msk_f32_u32_zro <<= bit_xpl_nbr_zro;

                /* BitSet mask for OR: Put ones into bits to be set, zeros in
                 * untouched bits. */
                msk_f32_u32_one = ~msk_f32_u32_zro;

                /* BitRound mask for ADD: Set one bit: the MSB of LSBs */
                msk_f32_u32_hshv = msk_f32_u32_one & (msk_f32_u32_zro >> 1);

            } else {
                bit_xpl_nbr_zro = BIT_XPL_NBR_SGN_DBL - prc_bnr_xpl_rqr;
                /* Create mask. */
                msk_f64_u64_zro = 0ul; /* Zero all bits. */
                msk_f64_u64_zro = ~msk_f64_u64_zro; /* Turn all bits to ones. */

                /* BitShave mask for AND: Left shift zeros into bits to be
                 * rounded, leave ones in untouched bits. */
                msk_f64_u64_zro <<= bit_xpl_nbr_zro;

                /* BitSet mask for OR: Put ones into bits to be set, zeros in
                 * untouched bits. */
                msk_f64_u64_one = ~msk_f64_u64_zro;

                /* BitRound mask for ADD: Set one bit: the MSB of LSBs */
                msk_f64_u64_hshv = msk_f64_u64_one & (msk_f64_u64_zro >> 1);
            }
        }

    } /* endif quantize */

    /* OK, this is ugly. If you can think of anything better, I'm open
       to suggestions!

       Note that we don't use a default fill value for type
       NC_BYTE. This is because Lord Voldemort cast a nofilleramous spell
       at Harry Potter, but it bounced off his scar and hit the netcdf-4
       code.
    */
    switch (src_type) {
        case NC_CHAR:
            switch (dest_type) {
                case NC_CHAR:
                    for (cp = (char *)src, cp1 = dest; count < len; count++) *cp1++ = *cp++;
                    break;
                default:;
            }
            break;

        case NC_BYTE:
            switch (dest_type) {
                case NC_BYTE:
                    for (bp = (signed char *)src, bp1 = dest; count < len; count++) *bp1++ = *bp++;
                    break;
                case NC_UBYTE:
                    for (bp = (signed char *)src, ubp = dest; count < len; count++) {
                        if (*bp < 0)
                            (*range_error)++;
                        *ubp++ = (unsigned char)*bp++;
                    }
                    break;
                case NC_SHORT:
                    for (bp = (signed char *)src, sp = dest; count < len; count++) *sp++ = *bp++;
                    break;
                case NC_USHORT:
                    for (bp = (signed char *)src, usp = dest; count < len; count++) {
                        if (*bp < 0)
                            (*range_error)++;
                        *usp++ = (unsigned short)*bp++;
                    }
                    break;
                case NC_INT:
                    for (bp = (signed char *)src, ip = dest; count < len; count++) *ip++ = *bp++;
                    break;
                case NC_UINT:
                    for (bp = (signed char *)src, uip = dest; count < len; count++) {
                        if (*bp < 0)
                            (*range_error)++;
                        *uip++ = (unsigned int)*bp++;
                    }
                    break;
                case NC_INT64:
                    for (bp = (signed char *)src, lip = dest; count < len; count++) *lip++ = *bp++;
                    break;
                case NC_UINT64:
                    for (bp = (signed char *)src, ulip = dest; count < len; count++) {
                        if (*bp < 0)
                            (*range_error)++;
                        *ulip++ = (unsigned long long)*bp++;
                    }
                    break;
                case NC_FLOAT:
                    for (bp = (signed char *)src, fp = dest; count < len; count++) *fp++ = *bp++;
                    break;
                case NC_DOUBLE:
                    for (bp = (signed char *)src, dp = dest; count < len; count++) *dp++ = *bp++;
                    break;
                default:
                    return NC_EBADTYPE;
            }
            break;

        case NC_UBYTE:
            switch (dest_type) {
                case NC_BYTE:
                    for (ubp = (unsigned char *)src, bp = dest; count < len; count++) {
                        if (!strict_nc3 && *ubp > X_SCHAR_MAX)
                            (*range_error)++;
                        *bp++ = (signed char)*ubp++;
                    }
                    break;
                case NC_SHORT:
                    for (ubp = (unsigned char *)src, sp = dest; count < len; count++) *sp++ = *ubp++;
                    break;
                case NC_UBYTE:
                    for (ubp = (unsigned char *)src, ubp1 = dest; count < len; count++) *ubp1++ = *ubp++;
                    break;
                case NC_USHORT:
                    for (ubp = (unsigned char *)src, usp = dest; count < len; count++) *usp++ = *ubp++;
                    break;
                case NC_INT:
                    for (ubp = (unsigned char *)src, ip = dest; count < len; count++) *ip++ = *ubp++;
                    break;
                case NC_UINT:
                    for (ubp = (unsigned char *)src, uip = dest; count < len; count++) *uip++ = *ubp++;
                    break;
                case NC_INT64:
                    for (ubp = (unsigned char *)src, lip = dest; count < len; count++) *lip++ = *ubp++;
                    break;
                case NC_UINT64:
                    for (ubp = (unsigned char *)src, ulip = dest; count < len; count++) *ulip++ = *ubp++;
                    break;
                case NC_FLOAT:
                    for (ubp = (unsigned char *)src, fp = dest; count < len; count++) *fp++ = *ubp++;
                    break;
                case NC_DOUBLE:
                    for (ubp = (unsigned char *)src, dp = dest; count < len; count++) *dp++ = *ubp++;
                    break;
                default:
                    return NC_EBADTYPE;
            }
            break;

        case NC_SHORT:
            switch (dest_type) {
                case NC_UBYTE:
                    for (sp = (short *)src, ubp = dest; count < len; count++) {
                        if (*sp > X_UCHAR_MAX || *sp < 0)
                            (*range_error)++;
                        *ubp++ = (unsigned char)*sp++;
                    }
                    break;
                case NC_BYTE:
                    for (sp = (short *)src, bp = dest; count < len; count++) {
                        if (*sp > X_SCHAR_MAX || *sp < X_SCHAR_MIN)
                            (*range_error)++;
                        *bp++ = (signed char)*sp++;
                    }
                    break;
                case NC_SHORT:
                    for (sp = (short *)src, sp1 = dest; count < len; count++) *sp1++ = *sp++;
                    break;
                case NC_USHORT:
                    for (sp = (short *)src, usp = dest; count < len; count++) {
                        if (*sp < 0)
                            (*range_error)++;
                        *usp++ = (unsigned short)*sp++;
                    }
                    break;
                case NC_INT:
                    for (sp = (short *)src, ip = dest; count < len; count++) *ip++ = *sp++;
                    break;
                case NC_UINT:
                    for (sp = (short *)src, uip = dest; count < len; count++) {
                        if (*sp < 0)
                            (*range_error)++;
                        *uip++ = (unsigned int)*sp++;
                    }
                    break;
                case NC_INT64:
                    for (sp = (short *)src, lip = dest; count < len; count++) *lip++ = *sp++;
                    break;
                case NC_UINT64:
                    for (sp = (short *)src, ulip = dest; count < len; count++) {
                        if (*sp < 0)
                            (*range_error)++;
                        *ulip++ = (unsigned long long)*sp++;
                    }
                    break;
                case NC_FLOAT:
                    for (sp = (short *)src, fp = dest; count < len; count++) *fp++ = *sp++;
                    break;
                case NC_DOUBLE:
                    for (sp = (short *)src, dp = dest; count < len; count++) *dp++ = *sp++;
                    break;
                default:
                    return NC_EBADTYPE;
            }
            break;

        case NC_USHORT:
            switch (dest_type) {
                case NC_UBYTE:
                    for (usp = (unsigned short *)src, ubp = dest; count < len; count++) {
                        if (*usp > X_UCHAR_MAX)
                            (*range_error)++;
                        *ubp++ = (unsigned char)*usp++;
                    }
                    break;
                case NC_BYTE:
                    for (usp = (unsigned short *)src, bp = dest; count < len; count++) {
                        if (*usp > X_SCHAR_MAX)
                            (*range_error)++;
                        *bp++ = (signed char)*usp++;
                    }
                    break;
                case NC_SHORT:
                    for (usp = (unsigned short *)src, sp = dest; count < len; count++) {
                        if (*usp > X_SHORT_MAX)
                            (*range_error)++;
                        *sp++ = (signed short)*usp++;
                    }
                    break;
                case NC_USHORT:
                    for (usp = (unsigned short *)src, usp1 = dest; count < len; count++) *usp1++ = *usp++;
                    break;
                case NC_INT:
                    for (usp = (unsigned short *)src, ip = dest; count < len; count++) *ip++ = *usp++;
                    break;
                case NC_UINT:
                    for (usp = (unsigned short *)src, uip = dest; count < len; count++) *uip++ = *usp++;
                    break;
                case NC_INT64:
                    for (usp = (unsigned short *)src, lip = dest; count < len; count++) *lip++ = *usp++;
                    break;
                case NC_UINT64:
                    for (usp = (unsigned short *)src, ulip = dest; count < len; count++) *ulip++ = *usp++;
                    break;
                case NC_FLOAT:
                    for (usp = (unsigned short *)src, fp = dest; count < len; count++) *fp++ = *usp++;
                    break;
                case NC_DOUBLE:
                    for (usp = (unsigned short *)src, dp = dest; count < len; count++) *dp++ = *usp++;
                    break;
                default:
                    return NC_EBADTYPE;
            }
            break;

        case NC_INT:
            switch (dest_type) {
                case NC_UBYTE:
                    for (ip = (int *)src, ubp = dest; count < len; count++) {
                        if (*ip > X_UCHAR_MAX || *ip < 0)
                            (*range_error)++;
                        *ubp++ = (unsigned char)*ip++;
                    }
                    break;
                case NC_BYTE:
                    for (ip = (int *)src, bp = dest; count < len; count++) {
                        if (*ip > X_SCHAR_MAX || *ip < X_SCHAR_MIN)
                            (*range_error)++;
                        *bp++ = (signed char)*ip++;
                    }
                    break;
                case NC_SHORT:
                    for (ip = (int *)src, sp = dest; count < len; count++) {
                        if (*ip > X_SHORT_MAX || *ip < X_SHORT_MIN)
                            (*range_error)++;
                        *sp++ = (short)*ip++;
                    }
                    break;
                case NC_USHORT:
                    for (ip = (int *)src, usp = dest; count < len; count++) {
                        if (*ip > X_USHORT_MAX || *ip < 0)
                            (*range_error)++;
                        *usp++ = (unsigned short)*ip++;
                    }
                    break;
                case NC_INT: /* src is int */
                    for (ip = (int *)src, ip1 = dest; count < len; count++) {
                        if (*ip > X_INT_MAX || *ip < X_INT_MIN)
                            (*range_error)++;
                        *ip1++ = *ip++;
                    }
                    break;
                case NC_UINT:
                    for (ip = (int *)src, uip = dest; count < len; count++) {
                        if (*ip > X_UINT_MAX || *ip < 0)
                            (*range_error)++;
                        *uip++ = (unsigned int)*ip++;
                    }
                    break;
                case NC_INT64:
                    for (ip = (int *)src, lip = dest; count < len; count++) *lip++ = *ip++;
                    break;
                case NC_UINT64:
                    for (ip = (int *)src, ulip = dest; count < len; count++) {
                        if (*ip < 0)
                            (*range_error)++;
                        *ulip++ = (unsigned long long)*ip++;
                    }
                    break;
                case NC_FLOAT:
                    for (ip = (int *)src, fp = dest; count < len; count++) *fp++ = (float)*ip++;
                    break;
                case NC_DOUBLE:
                    for (ip = (int *)src, dp = dest; count < len; count++) *dp++ = (double)*ip++;
                    break;
                default:
                    return NC_EBADTYPE;
            }
            break;

        case NC_UINT:
            switch (dest_type) {
                case NC_UBYTE:
                    for (uip = (unsigned int *)src, ubp = dest; count < len; count++) {
                        if (*uip > X_UCHAR_MAX)
                            (*range_error)++;
                        *ubp++ = (unsigned char)*uip++;
                    }
                    break;
                case NC_BYTE:
                    for (uip = (unsigned int *)src, bp = dest; count < len; count++) {
                        if (*uip > X_SCHAR_MAX)
                            (*range_error)++;
                        *bp++ = (signed char)*uip++;
                    }
                    break;
                case NC_SHORT:
                    for (uip = (unsigned int *)src, sp = dest; count < len; count++) {
                        if (*uip > X_SHORT_MAX)
                            (*range_error)++;
                        *sp++ = (signed short)*uip++;
                    }
                    break;
                case NC_USHORT:
                    for (uip = (unsigned int *)src, usp = dest; count < len; count++) {
                        if (*uip > X_USHORT_MAX)
                            (*range_error)++;
                        *usp++ = (unsigned short)*uip++;
                    }
                    break;
                case NC_INT:
                    for (uip = (unsigned int *)src, ip = dest; count < len; count++) {
                        if (*uip > X_INT_MAX)
                            (*range_error)++;
                        *ip++ = (int)*uip++;
                    }
                    break;
                case NC_UINT:
                    for (uip = (unsigned int *)src, uip1 = dest; count < len; count++) {
                        if (*uip > X_UINT_MAX)
                            (*range_error)++;
                        *uip1++ = *uip++;
                    }
                    break;
                case NC_INT64:
                    for (uip = (unsigned int *)src, lip = dest; count < len; count++) *lip++ = *uip++;
                    break;
                case NC_UINT64:
                    for (uip = (unsigned int *)src, ulip = dest; count < len; count++) *ulip++ = *uip++;
                    break;
                case NC_FLOAT:
                    for (uip = (unsigned int *)src, fp = dest; count < len; count++) *fp++ = (float)*uip++;
                    break;
                case NC_DOUBLE:
                    for (uip = (unsigned int *)src, dp = dest; count < len; count++) *dp++ = *uip++;
                    break;
                default:

                    return NC_EBADTYPE;
            }
            break;

        case NC_INT64:
            switch (dest_type) {
                case NC_UBYTE:
                    for (lip = (long long *)src, ubp = dest; count < len; count++) {
                        if (*lip > X_UCHAR_MAX || *lip < 0)
                            (*range_error)++;
                        *ubp++ = (unsigned char)*lip++;
                    }
                    break;
                case NC_BYTE:
                    for (lip = (long long *)src, bp = dest; count < len; count++) {
                        if (*lip > X_SCHAR_MAX || *lip < X_SCHAR_MIN)
                            (*range_error)++;
                        *bp++ = (signed char)*lip++;
                    }
                    break;
                case NC_SHORT:
                    for (lip = (long long *)src, sp = dest; count < len; count++) {
                        if (*lip > X_SHORT_MAX || *lip < X_SHORT_MIN)
                            (*range_error)++;
                        *sp++ = (short)*lip++;
                    }
                    break;
                case NC_USHORT:
                    for (lip = (long long *)src, usp = dest; count < len; count++) {
                        if (*lip > X_USHORT_MAX || *lip < 0)
                            (*range_error)++;
                        *usp++ = (unsigned short)*lip++;
                    }
                    break;
                case NC_UINT:
                    for (lip = (long long *)src, uip = dest; count < len; count++) {
                        if (*lip > X_UINT_MAX || *lip < 0)
                            (*range_error)++;
                        *uip++ = (unsigned int)*lip++;
                    }
                    break;
                case NC_INT:
                    for (lip = (long long *)src, ip = dest; count < len; count++) {
                        if (*lip > X_INT_MAX || *lip < X_INT_MIN)
                            (*range_error)++;
                        *ip++ = (int)*lip++;
                    }
                    break;
                case NC_INT64:
                    for (lip = (long long *)src, lip1 = dest; count < len; count++) *lip1++ = *lip++;
                    break;
                case NC_UINT64:
                    for (lip = (long long *)src, ulip = dest; count < len; count++) {
                        if (*lip < 0)
                            (*range_error)++;
                        *ulip++ = (unsigned long long)*lip++;
                    }
                    break;
                case NC_FLOAT:
                    for (lip = (long long *)src, fp = dest; count < len; count++) *fp++ = (float)*lip++;
                    break;
                case NC_DOUBLE:
                    for (lip = (long long *)src, dp = dest; count < len; count++) *dp++ = (double)*lip++;
                    break;
                default:
                    return NC_EBADTYPE;
            }
            break;

        case NC_UINT64:
            switch (dest_type) {
                case NC_UBYTE:
                    for (ulip = (unsigned long long *)src, ubp = dest; count < len; count++) {
                        if (*ulip > X_UCHAR_MAX)
                            (*range_error)++;
                        *ubp++ = (unsigned char)*ulip++;
                    }
                    break;
                case NC_BYTE:
                    for (ulip = (unsigned long long *)src, bp = dest; count < len; count++) {
                        if (*ulip > X_SCHAR_MAX)
                            (*range_error)++;
                        *bp++ = (signed char)*ulip++;
                    }
                    break;
                case NC_SHORT:
                    for (ulip = (unsigned long long *)src, sp = dest; count < len; count++) {
                        if (*ulip > X_SHORT_MAX)
                            (*range_error)++;
                        *sp++ = (short)*ulip++;
                    }
                    break;
                case NC_USHORT:
                    for (ulip = (unsigned long long *)src, usp = dest; count < len; count++) {
                        if (*ulip > X_USHORT_MAX)
                            (*range_error)++;
                        *usp++ = (unsigned short)*ulip++;
                    }
                    break;
                case NC_UINT:
                    for (ulip = (unsigned long long *)src, uip = dest; count < len; count++) {
                        if (*ulip > X_UINT_MAX)
                            (*range_error)++;
                        *uip++ = (unsigned int)*ulip++;
                    }
                    break;
                case NC_INT:
                    for (ulip = (unsigned long long *)src, ip = dest; count < len; count++) {
                        if (*ulip > X_INT_MAX)
                            (*range_error)++;
                        *ip++ = (int)*ulip++;
                    }
                    break;
                case NC_INT64:
                    for (ulip = (unsigned long long *)src, lip = dest; count < len; count++) {
                        if (*ulip > X_INT64_MAX)
                            (*range_error)++;
                        *lip++ = (long long)*ulip++;
                    }
                    break;
                case NC_UINT64:
                    for (ulip = (unsigned long long *)src, ulip1 = dest; count < len; count++) *ulip1++ = *ulip++;
                    break;
                case NC_FLOAT:
                    for (ulip = (unsigned long long *)src, fp = dest; count < len; count++) *fp++ = (float)*ulip++;
                    break;
                case NC_DOUBLE:
                    for (ulip = (unsigned long long *)src, dp = dest; count < len; count++) *dp++ = (double)*ulip++;
                    break;
                default:
                    return NC_EBADTYPE;
            }
            break;

        case NC_FLOAT:
            switch (dest_type) {
                case NC_UBYTE:
                    for (fp = (float *)src, ubp = dest; count < len; count++) {
                        if (*fp > X_UCHAR_MAX || *fp < 0)
                            (*range_error)++;
                        *ubp++ = (unsigned char)*fp++;
                    }
                    break;
                case NC_BYTE:
                    for (fp = (float *)src, bp = dest; count < len; count++) {
                        if (*fp > (double)X_SCHAR_MAX || *fp < (double)X_SCHAR_MIN)
                            (*range_error)++;
                        *bp++ = (signed char)*fp++;
                    }
                    break;
                case NC_SHORT:
                    for (fp = (float *)src, sp = dest; count < len; count++) {
                        if (*fp > (double)X_SHORT_MAX || *fp < (double)X_SHORT_MIN)
                            (*range_error)++;
                        *sp++ = (short)*fp++;
                    }
                    break;
                case NC_USHORT:
                    for (fp = (float *)src, usp = dest; count < len; count++) {
                        if (*fp > X_USHORT_MAX || *fp < 0)
                            (*range_error)++;
                        *usp++ = (unsigned short)*fp++;
                    }
                    break;
                case NC_UINT:
                    for (fp = (float *)src, uip = dest; count < len; count++) {
                        if (*fp > (float)X_UINT_MAX || *fp < 0)
                            (*range_error)++;
                        *uip++ = (unsigned int)*fp++;
                    }
                    break;
                case NC_INT:
                    for (fp = (float *)src, ip = dest; count < len; count++) {
                        if (*fp > (double)X_INT_MAX || *fp < (double)X_INT_MIN)
                            (*range_error)++;
                        *ip++ = (int)*fp++;
                    }
                    break;
                case NC_INT64:
                    for (fp = (float *)src, lip = dest; count < len; count++) {
                        if (*fp > (float)X_INT64_MAX || *fp < X_INT64_MIN)
                            (*range_error)++;
                        *lip++ = (long long)*fp++;
                    }
                    break;
                case NC_UINT64:
                    for (fp = (float *)src, ulip = dest; count < len; count++) {
                        if (*fp > (float)X_UINT64_MAX || *fp < 0)
                            (*range_error)++;
                        *ulip++ = (unsigned long long)*fp++;
                    }
                    break;
                case NC_FLOAT:
                    for (fp = (float *)src, fp1 = dest; count < len; count++)
                        *fp1++ = *fp++;
                    break;
                case NC_DOUBLE:
                    for (fp = (float *)src, dp = dest; count < len; count++)
                        *dp++ = *fp++;
                    break;
                default:
                    return NC_EBADTYPE;
            }
            break;

        case NC_DOUBLE:
            switch (dest_type) {
                case NC_UBYTE:
                    for (dp = (double *)src, ubp = dest; count < len; count++) {
                        if (*dp > X_UCHAR_MAX || *dp < 0)
                            (*range_error)++;
                        *ubp++ = (unsigned char)*dp++;
                    }
                    break;
                case NC_BYTE:
                    for (dp = (double *)src, bp = dest; count < len; count++) {
                        if (*dp > X_SCHAR_MAX || *dp < X_SCHAR_MIN)
                            (*range_error)++;
                        *bp++ = (signed char)*dp++;
                    }
                    break;
                case NC_SHORT:
                    for (dp = (double *)src, sp = dest; count < len; count++) {
                        if (*dp > X_SHORT_MAX || *dp < X_SHORT_MIN)
                            (*range_error)++;
                        *sp++ = (short)*dp++;
                    }
                    break;
                case NC_USHORT:
                    for (dp = (double *)src, usp = dest; count < len; count++) {
                        if (*dp > X_USHORT_MAX || *dp < 0)
                            (*range_error)++;
                        *usp++ = (unsigned short)*dp++;
                    }
                    break;
                case NC_UINT:
                    for (dp = (double *)src, uip = dest; count < len; count++) {
                        if (*dp > X_UINT_MAX || *dp < 0)
                            (*range_error)++;
                        *uip++ = (unsigned int)*dp++;
                    }
                    break;
                case NC_INT:
                    for (dp = (double *)src, ip = dest; count < len; count++) {
                        if (*dp > X_INT_MAX || *dp < X_INT_MIN)
                            (*range_error)++;
                        *ip++ = (int)*dp++;
                    }
                    break;
                case NC_INT64:
                    for (dp = (double *)src, lip = dest; count < len; count++) {
                        if (*dp > (double)X_INT64_MAX || *dp < X_INT64_MIN)
                            (*range_error)++;
                        *lip++ = (long long)*dp++;
                    }
                    break;
                case NC_UINT64:
                    for (dp = (double *)src, ulip = dest; count < len; count++) {
                        if (*dp > (double)X_UINT64_MAX || *dp < 0)
                            (*range_error)++;
                        *ulip++ = (unsigned long long)*dp++;
                    }
                    break;
                case NC_FLOAT:
                    for (dp = (double *)src, fp = dest; count < len; count++) {
                        if (isgreater(*dp, X_FLOAT_MAX) || isless(*dp, X_FLOAT_MIN))
                            (*range_error)++;
                        *fp++ = (float)*dp++;
                    }
                    break;
                case NC_DOUBLE:
                    for (dp = (double *)src, dp1 = dest; count < len; count++)
                        *dp1++ = *dp++;
                    break;
                default:
                    return NC_EBADTYPE;
            }
            break;

        default:
            return NC_EBADTYPE;
    }

    /* If quantize is in use, determine masks, copy the data, do the
     * quantization. */
    if (quantize_mode == NC_QUANTIZE_BITGROOM) {
        if (dest_type == NC_FLOAT) {
            /* BitGroom: alternately shave and set LSBs */
            op1.fp = (float *)dest;
            u32_ptr = op1.ui32p;
            for (idx = 0L; idx < len; idx += 2L)
                if (op1.fp[idx] != mss_val_cmp_flt)
                    u32_ptr[idx] &= msk_f32_u32_zro;
            for (idx = 1L; idx < len; idx += 2L)
                if (op1.fp[idx] != mss_val_cmp_flt && u32_ptr[idx] != 0U) /* Never quantize upwards floating point values of zero */
                    u32_ptr[idx] |= msk_f32_u32_one;
        } else {
            /* BitGroom: alternately shave and set LSBs. */
            op1.dp = (double *)dest;
            u64_ptr = op1.ui64p;
            for (idx = 0L; idx < len; idx += 2L)
                if (op1.dp[idx] != mss_val_cmp_dbl)
                    u64_ptr[idx] &= msk_f64_u64_zro;
            for (idx = 1L; idx < len; idx += 2L)
                if (op1.dp[idx] != mss_val_cmp_dbl && u64_ptr[idx] != 0ULL) /* Never quantize upwards floating point values of zero */
                    u64_ptr[idx] |= msk_f64_u64_one;
        }
    } /* endif BitGroom */

    if (quantize_mode == NC_QUANTIZE_BITROUND) {
        if (dest_type == NC_FLOAT) {
            /* BitRound: Quantize to user-specified NSB with IEEE-rounding */
            op1.fp = (float *)dest;
            u32_ptr = op1.ui32p;
            for (idx = 0L; idx < len; idx++) {
                if (op1.fp[idx] != mss_val_cmp_flt) {
                    u32_ptr[idx] += msk_f32_u32_hshv; /* Add 1 to the MSB of LSBs, carry 1 to mantissa or even exponent */
                    u32_ptr[idx] &= msk_f32_u32_zro; /* Shave it */
                }
            }
        } else {
            /* BitRound: Quantize to user-specified NSB with IEEE-rounding */
            op1.dp = (double *)dest;
            u64_ptr = op1.ui64p;
            for (idx = 0L; idx < len; idx++) {
                if (op1.dp[idx] != mss_val_cmp_dbl) {
                    u64_ptr[idx] += msk_f64_u64_hshv; /* Add 1 to the MSB of LSBs, carry 1 to mantissa or even exponent */
                    u64_ptr[idx] &= msk_f64_u64_zro; /* Shave it */
                }
            }
        }
    } /* endif BitRound */

    if (quantize_mode == NC_QUANTIZE_GRANULARBR) {
        if (dest_type == NC_FLOAT) {
            /* Granular BitRound */
            op1.fp = (float *)dest;
            u32_ptr = op1.ui32p;
            for (idx = 0L; idx < len; idx++) {
                if ((val = op1.fp[idx]) != mss_val_cmp_flt && u32_ptr[idx] != 0U) {
                    mnt = frexp(val, &xpn_bs2); /* DGG19 p. 4102 (8) */
                    mnt_fabs = fabs(mnt);
                    mnt_log10_fabs = log10(mnt_fabs);
                    /* 20211003 Continuous determination of dgt_nbr improves CR by ~10% */
                    dgt_nbr = (int)floor(xpn_bs2 * dgt_per_bit + mnt_log10_fabs) + 1; /* DGG19 p. 4102 (8.67) */
                    qnt_pwr = (int)floor(bit_per_dgt * (dgt_nbr - nsd)); /* DGG19 p. 4101 (7) */
                    prc_bnr_xpl_rqr = mnt_fabs == 0.0 ? 0 : abs((int)floor(xpn_bs2 - bit_per_dgt * mnt_log10_fabs) - qnt_pwr); /* Protect against mnt = -0.0 */
                    prc_bnr_xpl_rqr--; /* 20211003 Reduce formula result by 1 bit: Passes all tests, improves CR by ~10% */

                    bit_xpl_nbr_zro = BIT_XPL_NBR_SGN_FLT - prc_bnr_xpl_rqr;
                    msk_f32_u32_zro = 0u; /* Zero all bits */
                    msk_f32_u32_zro = ~msk_f32_u32_zro; /* Turn all bits to ones */
                    /* Bit Shave mask for AND: Left shift zeros into bits to be rounded, leave ones in untouched bits */
                    msk_f32_u32_zro <<= bit_xpl_nbr_zro;
                    /* Bit Set   mask for OR:  Put ones into bits to be set, zeros in untouched bits */
                    msk_f32_u32_one = ~msk_f32_u32_zro;
                    msk_f32_u32_hshv = msk_f32_u32_one & (msk_f32_u32_zro >> 1); /* Set one bit: the MSB of LSBs */
                    u32_ptr[idx] += msk_f32_u32_hshv; /* Add 1 to the MSB of LSBs, carry 1 to mantissa or even exponent */
                    u32_ptr[idx] &= msk_f32_u32_zro; /* Shave it */

                } /* !mss_val_cmp_flt */
            }
        } else {
            /* Granular BitRound */
            op1.dp = (double *)dest;
            u64_ptr = op1.ui64p;
            for (idx = 0L; idx < len; idx++) {
                if ((val = op1.dp[idx]) != mss_val_cmp_dbl && u64_ptr[idx] != 0ULL) {
                    mnt = frexp(val, &xpn_bs2); /* DGG19 p. 4102 (8) */
                    mnt_fabs = fabs(mnt);
                    mnt_log10_fabs = log10(mnt_fabs);
                    /* 20211003 Continuous determination of dgt_nbr improves CR by ~10% */
                    dgt_nbr = (int)floor(xpn_bs2 * dgt_per_bit + mnt_log10_fabs) + 1; /* DGG19 p. 4102 (8.67) */
                    qnt_pwr = (int)floor(bit_per_dgt * (dgt_nbr - nsd)); /* DGG19 p. 4101 (7) */
                    prc_bnr_xpl_rqr = mnt_fabs == 0.0 ? 0 : abs((int)floor(xpn_bs2 - bit_per_dgt * mnt_log10_fabs) - qnt_pwr); /* Protect against mnt = -0.0 */
                    prc_bnr_xpl_rqr--; /* 20211003 Reduce formula result by 1 bit: Passes all tests, improves CR by ~10% */

                    bit_xpl_nbr_zro = BIT_XPL_NBR_SGN_DBL - prc_bnr_xpl_rqr;
                    msk_f64_u64_zro = 0ull; /* Zero all bits */
                    msk_f64_u64_zro = ~msk_f64_u64_zro; /* Turn all bits to ones */
                    /* Bit Shave mask for AND: Left shift zeros into bits to be rounded, leave ones in untouched bits */
                    msk_f64_u64_zro <<= bit_xpl_nbr_zro;
                    /* Bit Set   mask for OR:  Put ones into bits to be set, zeros in untouched bits */
                    msk_f64_u64_one = ~msk_f64_u64_zro;
                    msk_f64_u64_hshv = msk_f64_u64_one & (msk_f64_u64_zro >> 1); /* Set one bit: the MSB of LSBs */
                    u64_ptr[idx] += msk_f64_u64_hshv; /* Add 1 to the MSB of LSBs, carry 1 to mantissa or even exponent */
                    u64_ptr[idx] &= msk_f64_u64_zro; /* Shave it */

                } /* !mss_val_cmp_dbl */
            }
        }
    } /* endif GranularBR */

    return NC_NOERR;
}
