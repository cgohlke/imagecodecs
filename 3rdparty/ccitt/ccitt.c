/* ccitt.c */

/* Initially generated with assistance from Claude (Anthropic),
   then reviewed and adapted by Christoph Gohlke. */

/*
SPDX-License-Identifier: 0BSD

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
*/

#include "ccitt.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>


/* Return nrows * rowlen, or CCITT_VALUE_ERROR on overflow. */
static inline ssize_t
ccitt_safe_product(
    const ssize_t nrows,
    const ssize_t rowlen)
{
    ssize_t result = nrows * rowlen;
    if (nrows != 0 && result / nrows != rowlen) {
        return CCITT_VALUE_ERROR;
    }
    return result;
}


/*************************** CCITT Huffman Tables ****************************/

/* ITU-T T.4 Tables 1/T.4, 2/T.4, 3/T.4.
   Modified Huffman run-length codes for bilevel image compression.

   Each entry: {bit pattern (MSB-aligned in 16 bits), bit length, run length}.
   Terminating codes encode runs 0-63.
   Make-up codes encode runs 64, 128, ..., 2560.
*/

typedef struct {
    uint16_t code;   /* Huffman code, left-aligned in 16 bits */
    uint8_t bits;    /* number of significant bits */
    uint16_t runlen; /* run length */
} ccitt_code_t;


/* Table 1/T.4 - White terminating codes (run lengths 0-63) */

static const ccitt_code_t ccitt_white_terminating[64] = {
    /* RL  code          bits */
    { 0x3500, 8, 0 },   /*  0: 00110101 */
    { 0x1C00, 6, 1 },   /*  1: 000111   */
    { 0x7000, 4, 2 },   /*  2: 0111     */
    { 0x8000, 4, 3 },   /*  3: 1000     */
    { 0xB000, 4, 4 },   /*  4: 1011     */
    { 0xC000, 4, 5 },   /*  5: 1100     */
    { 0xE000, 4, 6 },   /*  6: 1110     */
    { 0xF000, 4, 7 },   /*  7: 1111     */
    { 0x9800, 5, 8 },   /*  8: 10011    */
    { 0xA000, 5, 9 },   /*  9: 10100    */
    { 0x3800, 5, 10 },  /* 10: 00111    */
    { 0x4000, 5, 11 },  /* 11: 01000    */
    { 0x2000, 6, 12 },  /* 12: 001000   */
    { 0x0C00, 6, 13 },  /* 13: 000011   */
    { 0xD000, 6, 14 },  /* 14: 110100   */
    { 0xD400, 6, 15 },  /* 15: 110101   */
    { 0xA800, 6, 16 },  /* 16: 101010   */
    { 0xAC00, 6, 17 },  /* 17: 101011   */
    { 0x4E00, 7, 18 },  /* 18: 0100111  */
    { 0x1800, 7, 19 },  /* 19: 0001100  */
    { 0x1000, 7, 20 },  /* 20: 0001000  */
    { 0x2E00, 7, 21 },  /* 21: 0010111  */
    { 0x0600, 7, 22 },  /* 22: 0000011  */
    { 0x0800, 7, 23 },  /* 23: 0000100  */
    { 0x5000, 7, 24 },  /* 24: 0101000  */
    { 0x5600, 7, 25 },  /* 25: 0101011  */
    { 0x2600, 7, 26 },  /* 26: 0010011  */
    { 0x4800, 7, 27 },  /* 27: 0100100  */
    { 0x3000, 7, 28 },  /* 28: 0011000  */
    { 0x0200, 8, 29 },  /* 29: 00000010 */
    { 0x0300, 8, 30 },  /* 30: 00000011 */
    { 0x1A00, 8, 31 },  /* 31: 00011010 */
    { 0x1B00, 8, 32 },  /* 32: 00011011 */
    { 0x1200, 8, 33 },  /* 33: 00010010 */
    { 0x1300, 8, 34 },  /* 34: 00010011 */
    { 0x1400, 8, 35 },  /* 35: 00010100 */
    { 0x1500, 8, 36 },  /* 36: 00010101 */
    { 0x1600, 8, 37 },  /* 37: 00010110 */
    { 0x1700, 8, 38 },  /* 38: 00010111 */
    { 0x2800, 8, 39 },  /* 39: 00101000 */
    { 0x2900, 8, 40 },  /* 40: 00101001 */
    { 0x2A00, 8, 41 },  /* 41: 00101010 */
    { 0x2B00, 8, 42 },  /* 42: 00101011 */
    { 0x2C00, 8, 43 },  /* 43: 00101100 */
    { 0x2D00, 8, 44 },  /* 44: 00101101 */
    { 0x0400, 8, 45 },  /* 45: 00000100 */
    { 0x0500, 8, 46 },  /* 46: 00000101 */
    { 0x0A00, 8, 47 },  /* 47: 00001010 */
    { 0x0B00, 8, 48 },  /* 48: 00001011 */
    { 0x5200, 8, 49 },  /* 49: 01010010 */
    { 0x5300, 8, 50 },  /* 50: 01010011 */
    { 0x5400, 8, 51 },  /* 51: 01010100 */
    { 0x5500, 8, 52 },  /* 52: 01010101 */
    { 0x2400, 8, 53 },  /* 53: 00100100 */
    { 0x2500, 8, 54 },  /* 54: 00100101 */
    { 0x5800, 8, 55 },  /* 55: 01011000 */
    { 0x5900, 8, 56 },  /* 56: 01011001 */
    { 0x5A00, 8, 57 },  /* 57: 01011010 */
    { 0x5B00, 8, 58 },  /* 58: 01011011 */
    { 0x4A00, 8, 59 },  /* 59: 01001010 */
    { 0x4B00, 8, 60 },  /* 60: 01001011 */
    { 0x3200, 8, 61 },  /* 61: 00110010 */
    { 0x3300, 8, 62 },  /* 62: 00110011 */
    { 0x3400, 8, 63 },  /* 63: 00110100 */
};


/* Table 1/T.4 - White make-up codes (run lengths 64-2560) */

static const ccitt_code_t ccitt_white_makeup[27] = {
    /* RL    code          bits */
    { 0xD800,  5,   64 },  /*   64: 11011    */
    { 0x9000,  5,  128 },  /*  128: 10010    */
    { 0x5C00,  6,  192 },  /*  192: 010111   */
    { 0x6E00,  7,  256 },  /*  256: 0110111  */
    { 0x3600,  8,  320 },  /*  320: 00110110 */
    { 0x3700,  8,  384 },  /*  384: 00110111 */
    { 0x6400,  8,  448 },  /*  448: 01100100 */
    { 0x6500,  8,  512 },  /*  512: 01100101 */
    { 0x6800,  8,  576 },  /*  576: 01101000 */
    { 0x6700,  8,  640 },  /*  640: 01100111 */
    { 0x6600,  9,  704 },  /*  704: 011001100  */
    { 0x6680,  9,  768 },  /*  768: 011001101  */
    { 0x6900,  9,  832 },  /*  832: 011010010  */
    { 0x6980,  9,  896 },  /*  896: 011010011  */
    { 0x6A00,  9,  960 },  /*  960: 011010100  */
    { 0x6A80,  9, 1024 },  /* 1024: 011010101  */
    { 0x6B00,  9, 1088 },  /* 1088: 011010110  */
    { 0x6B80,  9, 1152 },  /* 1152: 011010111  */
    { 0x6C00,  9, 1216 },  /* 1216: 011011000  */
    { 0x6C80,  9, 1280 },  /* 1280: 011011001  */
    { 0x6D00,  9, 1344 },  /* 1344: 011011010  */
    { 0x6D80,  9, 1408 },  /* 1408: 011011011  */
    { 0x4C00,  9, 1472 },  /* 1472: 010011000  */
    { 0x4C80,  9, 1536 },  /* 1536: 010011001  */
    { 0x4D00,  9, 1600 },  /* 1600: 010011010  */
    { 0x6000,  6, 1664 },  /* 1664: 011000   */
    { 0x4D80,  9, 1728 },  /* 1728: 010011011  */
};


/* Table 2/T.4 - Black terminating codes (run lengths 0-63) */

static const ccitt_code_t ccitt_black_terminating[64] = {
    /* RL  code            bits */
    { 0x0DC0, 10,  0 },  /*  0: 0000110111 */
    { 0x4000,  3,  1 },  /*  1: 010        */
    { 0xC000,  2,  2 },  /*  2: 11         */
    { 0x8000,  2,  3 },  /*  3: 10         */
    { 0x6000,  3,  4 },  /*  4: 011        */
    { 0x3000,  4,  5 },  /*  5: 0011       */
    { 0x2000,  4,  6 },  /*  6: 0010       */
    { 0x1800,  5,  7 },  /*  7: 00011      */
    { 0x1400,  6,  8 },  /*  8: 000101     */
    { 0x1000,  6,  9 },  /*  9: 000100     */
    { 0x0800,  7, 10 },  /* 10: 0000100    */
    { 0x0A00,  7, 11 },  /* 11: 0000101    */
    { 0x0E00,  7, 12 },  /* 12: 0000111    */
    { 0x0400,  8, 13 },  /* 13: 00000100   */
    { 0x0700,  8, 14 },  /* 14: 00000111   */
    { 0x0C00,  9, 15 },  /* 15: 000011000  */
    { 0x05C0, 10, 16 },  /* 16: 0000010111 */
    { 0x0600, 10, 17 },  /* 17: 0000011000 */
    { 0x0200, 10, 18 },  /* 18: 0000001000 */
    { 0x0CE0, 11, 19 },  /* 19: 00001100111  */
    { 0x0D00, 11, 20 },  /* 20: 00001101000  */
    { 0x0D80, 11, 21 },  /* 21: 00001101100  */
    { 0x06E0, 11, 22 },  /* 22: 00000110111  */
    { 0x0500, 11, 23 },  /* 23: 00000101000  */
    { 0x02E0, 11, 24 },  /* 24: 00000010111  */
    { 0x0300, 11, 25 },  /* 25: 00000011000  */
    { 0x0CA0, 12, 26 },  /* 26: 000011001010 */
    { 0x0CB0, 12, 27 },  /* 27: 000011001011 */
    { 0x0CC0, 12, 28 },  /* 28: 000011001100 */
    { 0x0CD0, 12, 29 },  /* 29: 000011001101 */
    { 0x0680, 12, 30 },  /* 30: 000001101000 */
    { 0x0690, 12, 31 },  /* 31: 000001101001 */
    { 0x06A0, 12, 32 },  /* 32: 000001101010 */
    { 0x06B0, 12, 33 },  /* 33: 000001101011 */
    { 0x0D20, 12, 34 },  /* 34: 000011010010 */
    { 0x0D30, 12, 35 },  /* 35: 000011010011 */
    { 0x0D40, 12, 36 },  /* 36: 000011010100 */
    { 0x0D50, 12, 37 },  /* 37: 000011010101 */
    { 0x0D60, 12, 38 },  /* 38: 000011010110 */
    { 0x0D70, 12, 39 },  /* 39: 000011010111 */
    { 0x06C0, 12, 40 },  /* 40: 000001101100 */
    { 0x06D0, 12, 41 },  /* 41: 000001101101 */
    { 0x0DA0, 12, 42 },  /* 42: 000011011010 */
    { 0x0DB0, 12, 43 },  /* 43: 000011011011 */
    { 0x0540, 12, 44 },  /* 44: 000001010100 */
    { 0x0550, 12, 45 },  /* 45: 000001010101 */
    { 0x0560, 12, 46 },  /* 46: 000001010110 */
    { 0x0570, 12, 47 },  /* 47: 000001010111 */
    { 0x0640, 12, 48 },  /* 48: 000001100100 */
    { 0x0650, 12, 49 },  /* 49: 000001100101 */
    { 0x0520, 12, 50 },  /* 50: 000001010010 */
    { 0x0530, 12, 51 },  /* 51: 000001010011 */
    { 0x0240, 12, 52 },  /* 52: 000000100100 */
    { 0x0370, 12, 53 },  /* 53: 000000110111 */
    { 0x0380, 12, 54 },  /* 54: 000000111000 */
    { 0x0270, 12, 55 },  /* 55: 000000100111 */
    { 0x0280, 12, 56 },  /* 56: 000000101000 */
    { 0x0580, 12, 57 },  /* 57: 000001011000 */
    { 0x0590, 12, 58 },  /* 58: 000001011001 */
    { 0x02B0, 12, 59 },  /* 59: 000000101011 */
    { 0x02C0, 12, 60 },  /* 60: 000000101100 */
    { 0x05A0, 12, 61 },  /* 61: 000001011010 */
    { 0x0660, 12, 62 },  /* 62: 000001100110 */
    { 0x0670, 12, 63 },  /* 63: 000001100111 */
};


/* Table 2/T.4 - Black make-up codes (run lengths 64-2560) */

static const ccitt_code_t ccitt_black_makeup[27] = {
    /* RL    code            bits */
    { 0x03C0, 10,   64 },  /*   64: 0000001111 */
    { 0x0C80, 12,  128 },  /*  128: 000011001000 */
    { 0x0C90, 12,  192 },  /*  192: 000011001001 */
    { 0x05B0, 12,  256 },  /*  256: 000001011011 */
    { 0x0330, 12,  320 },  /*  320: 000000110011 */
    { 0x0340, 12,  384 },  /*  384: 000000110100 */
    { 0x0350, 12,  448 },  /*  448: 000000110101 */
    { 0x0360, 13,  512 },  /*  512: 0000001101100 */
    { 0x0368, 13,  576 },  /*  576: 0000001101101 */
    { 0x0250, 13,  640 },  /*  640: 0000001001010 */
    { 0x0258, 13,  704 },  /*  704: 0000001001011 */
    { 0x0260, 13,  768 },  /*  768: 0000001001100 */
    { 0x0268, 13,  832 },  /*  832: 0000001001101 */
    { 0x0390, 13,  896 },  /*  896: 0000001110010 */
    { 0x0398, 13,  960 },  /*  960: 0000001110011 */
    { 0x03A0, 13, 1024 },  /* 1024: 0000001110100 */
    { 0x03A8, 13, 1088 },  /* 1088: 0000001110101 */
    { 0x03B0, 13, 1152 },  /* 1152: 0000001110110 */
    { 0x03B8, 13, 1216 },  /* 1216: 0000001110111 */
    { 0x0290, 13, 1280 },  /* 1280: 0000001010010 */
    { 0x0298, 13, 1344 },  /* 1344: 0000001010011 */
    { 0x02A0, 13, 1408 },  /* 1408: 0000001010100 */
    { 0x02A8, 13, 1472 },  /* 1472: 0000001010101 */
    { 0x02D0, 13, 1536 },  /* 1536: 0000001011010 */
    { 0x02D8, 13, 1600 },  /* 1600: 0000001011011 */
    { 0x0320, 13, 1664 },  /* 1664: 0000001100100 */
    { 0x0328, 13, 1728 },  /* 1728: 0000001100101 */
};


/* Table 3/T.4 - Extended make-up codes, shared by white and black.
   (run lengths 1792-2560, plus EOL) */

static const ccitt_code_t ccitt_extended_makeup[8] = {
    /* RL    code        bits */
    { 0x0100, 11, 1792 },  /* 1792: 00000001000 */
    { 0x0180, 11, 1856 },  /* 1856: 00000001100 */
    { 0x01A0, 11, 1920 },  /* 1920: 00000001101 */
    { 0x0120, 12, 1984 },  /* 1984: 000000010010 */
    { 0x0130, 12, 2048 },  /* 2048: 000000010011 */
    { 0x0140, 12, 2112 },  /* 2112: 000000010100 */
    { 0x0150, 12, 2176 },  /* 2176: 000000010101 */
    { 0x0160, 12, 2240 },  /* 2240: 000000010110 */
};

/* Additional extended make-up codes (2304-2560) */
static const ccitt_code_t ccitt_extended_makeup2[5] = {
    { 0x0170, 12, 2304 },  /* 2304: 000000010111 */
    { 0x01C0, 12, 2368 },  /* 2368: 000000011100 */
    { 0x01D0, 12, 2432 },  /* 2432: 000000011101 */
    { 0x01E0, 12, 2496 },  /* 2496: 000000011110 */
    { 0x01F0, 12, 2560 },  /* 2560: 000000011111 */
};

/* End-of-line code: 000000000001 (12 bits) */
#define CCITT_EOL_CODE   0x0010
#define CCITT_EOL_BITS   12

/* End-of-facsimile-block: two consecutive EOLs */
#define CCITT_EOFB_BITS  24


/*
 * Lookup tables for fast Huffman decoding.
 *
 * Instead of bit-by-bit tree walking, peek at the next N bits of input and
 * look up the result in a flat table.  White codes are at most 12 bits,
 * black codes at most 13 bits.  We use 13-bit tables for both (8192 entries
 * each) to keep the implementation uniform.
 *
 * Each entry encodes:
 *   bits 15..8 : code length (0 = invalid / not a match)
 *   bits  7..0 : encoded value, interpreted as:
 *                0..63    = terminating run of that length
 *                64..103  = make-up run: (value - 64) * 64 + 64
 *                           i.e., 64 -> 64, 65 -> 128, ..., 103 -> 2560
 *                105      = EOL
 *                0 in both fields = invalid code
 *
 * For make-up codes the actual run length is ((entry & 0xFF) - 64) * 64 + 64,
 * which must be added to the run length from the immediately following
 * terminating code (or further make-up codes).
 */

#define CCITT_LUT_BITS   13
#define CCITT_LUT_SIZE   (1 << CCITT_LUT_BITS)  /* 8192 */

/* Encoded lookup value helpers */
#define CCITT_LUT_ENTRY(codelen, val) \
    (uint16_t)(((codelen) << 8) | (val))
#define CCITT_LUT_CODELEN(e)  ((e) >> 8)
#define CCITT_LUT_VALUE(e)    ((e) & 0xFF)
#define CCITT_LUT_INVALID     0

/* Special value in lookup table for EOL */
#define CCITT_LUT_EOL_VALUE   105

/* Convert make-up lookup value back to run length */
#define CCITT_LUT_IS_MAKEUP(v)   ((v) >= 64 && (v) <= 103)
#define CCITT_LUT_MAKEUP_RUNLEN(v) (((ssize_t)(v) - 64) * 64 + 64)

/* Lookup tables (populated by ccitt_lut_init) */
static uint16_t ccitt_lut_white[CCITT_LUT_SIZE];
static uint16_t ccitt_lut_black[CCITT_LUT_SIZE];
static bool ccitt_lut_initialized = false;


/* Build lookup table from code table entries. */
static void
ccitt_lut_add(
    uint16_t* lut,
    const uint16_t code,
    const uint8_t bits,
    const uint8_t value)
{
    /* code is left-aligned in 16 bits; shift to get the LUT_BITS-wide index */
    int shift = 16 - CCITT_LUT_BITS;
    int prefix = code >> shift;
    int fill_count = 1 << (CCITT_LUT_BITS - bits);
    int i;
    uint16_t entry = CCITT_LUT_ENTRY(bits, value);

    assert(prefix >= 0 && prefix + fill_count <= CCITT_LUT_SIZE);

    for (i = 0; i < fill_count; i++) {
        lut[prefix + i] = entry;
    }
}


/* Initialize both white and black lookup tables.
 *
 * Must be called before any decode function.  Subsequent calls are no-ops.
 * Not thread-safe; callers must ensure serialization
 * (for example, call once at module initialization under a lock). */
void
ccitt_lut_init(
    void)
{
    int i;
    uint8_t makeup_val;

    memset(ccitt_lut_white, 0, sizeof(ccitt_lut_white));
    memset(ccitt_lut_black, 0, sizeof(ccitt_lut_black));

    /* White terminating codes 0-63 -> value = run length directly */
    for (i = 0; i < 64; i++) {
        ccitt_lut_add(
            ccitt_lut_white,
            ccitt_white_terminating[i].code,
            ccitt_white_terminating[i].bits,
            (uint8_t)i
        );
    }

    /* White make-up codes -> value = 64 + index (decode via macro) */
    for (i = 0; i < 27; i++) {
        makeup_val = (uint8_t)(64 + i);  /* 64..90 */
        ccitt_lut_add(
            ccitt_lut_white,
            ccitt_white_makeup[i].code,
            ccitt_white_makeup[i].bits,
            makeup_val
        );
    }

    /* Black terminating codes 0-63 -> value = run length directly */
    for (i = 0; i < 64; i++) {
        ccitt_lut_add(
            ccitt_lut_black,
            ccitt_black_terminating[i].code,
            ccitt_black_terminating[i].bits,
            (uint8_t)i
        );
    }

    /* Black make-up codes -> value = 64 + index */
    for (i = 0; i < 27; i++) {
        makeup_val = (uint8_t)(64 + i);  /* 64..90 */
        ccitt_lut_add(
            ccitt_lut_black,
            ccitt_black_makeup[i].code,
            ccitt_black_makeup[i].bits,
            makeup_val
        );
    }

    /* Extended make-up codes (shared by white and black) */
    /* value = 64 + 27 + index = 91..98 */
    for (i = 0; i < 8; i++) {
        makeup_val = (uint8_t)(91 + i);
        ccitt_lut_add(
            ccitt_lut_white,
            ccitt_extended_makeup[i].code,
            ccitt_extended_makeup[i].bits,
            makeup_val
        );
        ccitt_lut_add(
            ccitt_lut_black,
            ccitt_extended_makeup[i].code,
            ccitt_extended_makeup[i].bits,
            makeup_val
        );
    }

    /* Additional extended make-up codes: value = 99..103 */
    for (i = 0; i < 5; i++) {
        makeup_val = (uint8_t)(99 + i);
        ccitt_lut_add(
            ccitt_lut_white,
            ccitt_extended_makeup2[i].code,
            ccitt_extended_makeup2[i].bits,
            makeup_val
        );
        ccitt_lut_add(
            ccitt_lut_black,
            ccitt_extended_makeup2[i].code,
            ccitt_extended_makeup2[i].bits,
            makeup_val
        );
    }

    /* EOL code in both tables: value = CCITT_LUT_EOL_VALUE */
    ccitt_lut_add(
        ccitt_lut_white,
        CCITT_EOL_CODE,
        CCITT_EOL_BITS,
        CCITT_LUT_EOL_VALUE
    );
    ccitt_lut_add(
        ccitt_lut_black,
        CCITT_EOL_CODE,
        CCITT_EOL_BITS,
        CCITT_LUT_EOL_VALUE
    );

    ccitt_lut_initialized = true;
}


/*************************** CCITT Bitstream Reader **************************/

/* MSB-first bitstream reader for CCITT compressed data.
   Maintains a 32-bit accumulator for efficient multi-bit reads. */

typedef struct {
    const uint8_t *data;  /* source data pointer */
    ssize_t nbytes;       /* remaining bytes in source */
    uint32_t accumulator; /* bit accumulator, MSB-aligned */
    int bits;             /* number of valid bits in accumulator (0-32) */
} ccitt_bitstream_t;


/* Initialize bitstream reader. */
static void
ccitt_bitstream_init(
    ccitt_bitstream_t *bs,
    const uint8_t *data,
    const ssize_t nbytes)
{
    bs->data = data;
    bs->nbytes = nbytes;
    bs->accumulator = 0;
    bs->bits = 0;
}


/* Fill accumulator with bytes from source until at least 'need' bits
   are available or source is exhausted. */
static void
ccitt_bitstream_fill(
    ccitt_bitstream_t *bs,
    const int need)
{
    while (bs->bits < need && bs->nbytes > 0) {
        bs->accumulator |=
            (uint32_t)(*(bs->data)) << (24 - bs->bits);
        bs->data++;
        bs->nbytes--;
        bs->bits += 8;
    }
}


/* Peek at the top 'n' bits of the accumulator without consuming them.
   Caller must ensure enough bits are available. */
static inline uint32_t
ccitt_bitstream_peek(
    const ccitt_bitstream_t *bs,
    const int n)
{
    return bs->accumulator >> (32 - n);
}


/* Skip (consume) 'n' bits from the accumulator.
   Caller must ensure enough bits are available. */
static inline void
ccitt_bitstream_skip(
    ccitt_bitstream_t *bs,
    const int n)
{
    bs->accumulator <<= n;
    bs->bits -= n;
}


/* Peek at top CCITT_LUT_BITS bits for LUT lookup.
   Fills the accumulator if necessary.
   Returns the LUT index (top 13 bits). */
static inline uint32_t
ccitt_bitstream_peek_lut(
    ccitt_bitstream_t *bs)
{
    if (bs->bits < CCITT_LUT_BITS) {
        ccitt_bitstream_fill(bs, CCITT_LUT_BITS);
    }
    return bs->accumulator >> (32 - CCITT_LUT_BITS);
}


/**************************** CCITT 1D Line Decoder **************************/

/* Decode one line of 1D (Modified Huffman) coded data.
 *
 * Reads alternating white and black Huffman-coded runs starting with white.
 * Writes byte-per-pixel output (0=white, 1=black) to dst.
 * Makeup codes accumulate; a terminating code (0-63) finalizes the run.
 *
 * Always returns 0.  On invalid or truncated input the remaining pixels
 * are filled with white (robust/lenient decoding). */
static int
ccitt_decode_1d_line(
    ccitt_bitstream_t *bs,
    uint8_t *dst,
    const ssize_t rowlen)
{
    ssize_t pos = 0;
    int color = 0;  /* 0=white, 1=black */
    ssize_t run = 0;

    while (pos < rowlen) {
        uint32_t index;
        uint16_t entry;
        int codelen;
        int value;

        index = ccitt_bitstream_peek_lut(bs);
        entry = color ? ccitt_lut_black[index] : ccitt_lut_white[index];

        if (entry == CCITT_LUT_INVALID) {
            /* Non-conforming encoder: the row may be missing a terminating
             * code, or a final makeup run ends exactly at the row boundary
             * without a terminating code.  The fill bits that precede the
             * next EOL (all zeros) produce an invalid 13-bit LUT index.
             *
             * Commit the pending run if it fills the remaining pixels.
             * In either case break without consuming any bits so that the
             * caller can resynchronize via EOL detection. */
            if (run > 0 && pos + run == rowlen) {
                memset(dst + pos, (uint8_t)color, (size_t)run);
                pos = rowlen;
                run = 0;
            }
            break;
        }

        codelen = CCITT_LUT_CODELEN(entry);
        value = CCITT_LUT_VALUE(entry);

        if (codelen > bs->bits) {
            /* source data truncated */
            break;
        }

        if (value == CCITT_LUT_EOL_VALUE) {
            /* EOL encountered before line complete.  Don't consume the
             * EOL bits so the caller's EOL synchronization can find
             * and consume the EOL properly. */
            break;
        }

        ccitt_bitstream_skip(bs, codelen);

        if (CCITT_LUT_IS_MAKEUP(value)) {
            /* makeup code: accumulate run length, continue same color */
            run += CCITT_LUT_MAKEUP_RUNLEN(value);
            continue;
        }

        /* Terminating code: value is run length 0-63 */
        run += (ssize_t)value;

        if (pos + run > rowlen) {
            run = rowlen - pos;
        }

        /* Fill pixels with color value */
        memset(dst + pos, color, (size_t)run);
        pos += run;

        run = 0;
        color ^= 1;  /* alternate white <-> black */
    }

    /* Fill remainder with white if line is incomplete */
    if (pos < rowlen) {
        memset(dst + pos, 0, (size_t)(rowlen - pos));
    }

    return 0;
}


/****************************** CCITT EOL Detection **************************/

/* Find and consume an End-Of-Line (EOL) code in the bitstream.
 *
 * An EOL is 000000000001 (twelve bits: eleven 0s followed by a 1).
 * When byte_align is true, optional fill 0-bits up to the next byte
 * boundary are skipped before the EOL pattern (T.4 fill mode, TIFF
 * t4options bit 2).
 *
 * Returns:
 *   1: EOL found and consumed
 *   0: EOL not found (bitstream unchanged)
 *  -1: input exhausted                                                 */
static int
ccitt_find_eol(
    ccitt_bitstream_t *bs,
    const int byte_align)
{
    ccitt_bitstream_t saved = *bs;  /* snapshot for rollback */

    if (byte_align) {
        /* Skip fill bits (zeros) that precede the EOL.
         *
         * In T.4 fill mode (t4options bit 2) the encoder may insert zero-bits
         * to align the following EOL on a byte boundary.  The standard allows
         * at most 7 fill bits; non-conforming encoders may use more.
         *
         * We re-fill the accumulator before each check so that we can handle
         * arbitrarily many fill bits.  The loop exits when:
         *   - the next bit is not zero (no fill or already at EOL's own bits)
         *   - the 12-bit window already contains the EOL pattern 000000000001
         *   - the bitstream is exhausted
         */
        for (;;) {
            ccitt_bitstream_fill(bs, CCITT_EOL_BITS + 1);
            if (bs->bits < CCITT_EOL_BITS)
                break;  /* insufficient data */
            if (ccitt_bitstream_peek(bs, 1) != 0)
                break;  /* next bit is 1 – fill ended, check for EOL */
            if (ccitt_bitstream_peek(bs, CCITT_EOL_BITS) == 0x001)
                break;  /* EOL pattern already visible – stop skipping */
            ccitt_bitstream_skip(bs, 1);  /* consume one fill zero */
        }
    }
    else {
        ccitt_bitstream_fill(bs, CCITT_EOL_BITS);
        if (bs->bits < CCITT_EOL_BITS) {
            *bs = saved;
            return -1;
        }
    }

    /* Insufficient bits to hold an EOL
      (for example, stream ended mid-fill). */
    if (bs->bits < CCITT_EOL_BITS) {
        *bs = saved;
        return -1;
    }

    /* Check for the 12-bit EOL pattern: 000000000001 */
    if (ccitt_bitstream_peek(bs, CCITT_EOL_BITS) == 0x001) {
        ccitt_bitstream_skip(bs, CCITT_EOL_BITS);
        return 1;
    }

    /* No EOL found — restore bitstream position */
    *bs = saved;
    return 0;
}


/* Scan forward bit-by-bit to find and consume an EOL code.
 *
 * Unlike ccitt_find_eol(), this scans through arbitrary data (not just
 * fill zeros), making it suitable for initial synchronization on streams
 * that contain a preamble before the first EOL.
 *
 * To avoid consuming too much data on streams without EOL codes, the
 * search is limited to maxbits bits.  On failure the bitstream position
 * is restored.
 *
 * Returns:
 *   1: EOL found and consumed
 *   0: EOL not found within maxbits (bitstream unchanged) */
static int
ccitt_scan_to_eol(
    ccitt_bitstream_t *bs,
    const ssize_t maxbits)
{
    ccitt_bitstream_t saved = *bs;
    ssize_t scanned = 0;

    while (scanned < maxbits) {
        ccitt_bitstream_fill(bs, CCITT_EOL_BITS);
        if (bs->bits < CCITT_EOL_BITS) {
            break;  /* exhausted */
        }
        if (ccitt_bitstream_peek(bs, CCITT_EOL_BITS) == 0x001) {
            ccitt_bitstream_skip(bs, CCITT_EOL_BITS);
            return 1;
        }
        ccitt_bitstream_skip(bs, 1);
        scanned++;
    }

    *bs = saved;
    return 0;
}


/**************************** CCITT Run Decoder ******************************/

/* Decode a single Huffman-coded run length (makeup + terminating).
 *
 * Uses the given lookup table (white or black).
 * Returns the run length (>= 0), or -1 on error. */
static ssize_t
ccitt_decode_run(
    ccitt_bitstream_t *bs,
    const uint16_t *lut)
{
    ssize_t run = 0;

    for (;;) {
        uint32_t index = ccitt_bitstream_peek_lut(bs);
        uint16_t entry = lut[index];
        int codelen;
        int value;

        if (entry == CCITT_LUT_INVALID) {
            return -1;
        }

        codelen = CCITT_LUT_CODELEN(entry);
        value = CCITT_LUT_VALUE(entry);

        if (codelen > bs->bits) {
            return -1;
        }

        ccitt_bitstream_skip(bs, codelen);

        if (value == CCITT_LUT_EOL_VALUE) {
            return -1;  /* unexpected EOL in run */
        }

        if (CCITT_LUT_IS_MAKEUP(value)) {
            run += CCITT_LUT_MAKEUP_RUNLEN(value);
            continue;
        }

        /* Terminating code: value = 0..63 */
        run += (ssize_t)value;
        return run;
    }
}


/**************************** CCITT 2D Line Decoder **************************/

/* Find b1 and b2 on the reference line relative to position a0.
 *
 * b1: first changing element on the reference line strictly to the right of
 *     a0 whose color is opposite to a0_color.
 * b2: next changing element on the reference line to the right of b1.
 *
 * A changing element is a position where the pixel color differs from the
 * previous pixel (position 0 is implicitly preceded by white).
 *
 * b1 and/or b2 are set to rowlen when not found. */
static void
ccitt_find_b1b2(
    const uint8_t *ref,
    const ssize_t rowlen,
    const ssize_t a0,
    const int a0_color,
    ssize_t *b1,
    ssize_t *b2)
{
    ssize_t pos;
    int ref_at_a0;

    pos = (a0 < 0) ? 0 : a0 + 1;

    if (pos >= rowlen) {
        *b1 = rowlen;
        *b2 = rowlen;
        return;
    }

    /* Color of the reference line at position a0 (imaginary white
     * if a0 < 0 or past end) */
    ref_at_a0 = (a0 >= 0 && a0 < rowlen) ? (int)ref[a0] : 0;

    if (ref_at_a0 == a0_color) {
        /* Reference has same color as a0 at this position.
         * Advance to the first pixel with opposite color. */
        while (pos < rowlen && (int)ref[pos] == a0_color) {
            pos++;
        }
    }
    else {
        /* Reference already has opposite color at a0's position.
         * This opposite-color run started before a0, so its start is not
         * to the right of a0.  Skip past this opposite-color run, then
         * past the following same-color run, to find the next transition
         * to opposite color. */
        while (pos < rowlen && (int)ref[pos] != a0_color) {
            pos++;
        }
        while (pos < rowlen && (int)ref[pos] == a0_color) {
            pos++;
        }
    }

    *b1 = pos;

    /* b2: end of the run starting at b1 */
    if (pos < rowlen) {
        int b1_color = (int)ref[pos];
        pos++;
        while (pos < rowlen && (int)ref[pos] == b1_color) {
            pos++;
        }
    }
    *b2 = pos;
}


/* Decode one line of 2D-coded data using the reference line.
 *
 * Implements ITU-T T.4 Table 4: Pass, Horizontal, and Vertical (V0, VR1-3,
 * VL1-3) modes.  Used by both Group 3 2D and Group 4 decoders.
 * Writes byte-per-pixel output (0=white, 1=black) to dst.
 *
 * Returns 0 on success, negative on error. */
static int
ccitt_decode_2d_line(
    ccitt_bitstream_t *bs,
    uint8_t *dst,
    const uint8_t *ref,
    const ssize_t rowlen)
{
    ssize_t a0 = -1;    /* imaginary white element before position 0 */
    int a0_color = 0;    /* white */

    while (a0 < rowlen) {
        ssize_t b1, b2;
        ssize_t start;
        uint32_t bits;
        int skip;
        int mode;           /* 0=vertical, 1=horizontal, 2=pass */
        ssize_t voffset = 0;

        ccitt_find_b1b2(ref, rowlen, a0, a0_color, &b1, &b2);

        /* Read up to 7 bits for 2D mode code (Table 4/T.4) */
        ccitt_bitstream_fill(bs, 7);
        if (bs->bits < 1) {
            break;  /* input exhausted */
        }

        bits = ccitt_bitstream_peek(bs, 7);

        /* Decode the mode code */
        if (bits >= 0x40) {
            /* 1xxxxxx -> V(0) */
            skip = 1; mode = 0; voffset = 0;
        }
        else if (bits >= 0x30) {
            /* 011xxxx -> VR(1) */
            skip = 3; mode = 0; voffset = 1;
        }
        else if (bits >= 0x20) {
            /* 010xxxx -> VL(1) */
            skip = 3; mode = 0; voffset = -1;
        }
        else if (bits >= 0x10) {
            /* 001xxxx -> Horizontal */
            skip = 3; mode = 1;
        }
        else if (bits >= 0x08) {
            /* 0001xxx -> Pass */
            skip = 4; mode = 2;
        }
        else if (bits >= 0x06) {
            /* 000011x -> VR(2) */
            skip = 6; mode = 0; voffset = 2;
        }
        else if (bits >= 0x04) {
            /* 000010x -> VL(2) */
            skip = 6; mode = 0; voffset = -2;
        }
        else if (bits == 0x03) {
            /* 0000011 -> VR(3) */
            skip = 7; mode = 0; voffset = 3;
        }
        else if (bits == 0x02) {
            /* 0000010 -> VL(3) */
            skip = 7; mode = 0; voffset = -3;
        }
        else {
            /* 0000001 or 0000000: extension or EOL prefix */
            return CCITT_VALUE_ERROR;
        }

        if (skip > bs->bits) {
            break;  /* truncated */
        }
        ccitt_bitstream_skip(bs, skip);

        start = (a0 < 0) ? 0 : a0;

        if (mode == 0) {
            /* Vertical mode: a1 = b1 + voffset */
            ssize_t a1 = b1 + voffset;

            if (a1 < start) {
                a1 = start;
            }
            if (a1 > rowlen) {
                a1 = rowlen;
            }
            if (a1 > start) {
                memset(
                    dst + start,
                    (uint8_t)a0_color,
                    (size_t)(a1 - start)
                );
            }
            a0 = a1;
            a0_color ^= 1;
        }
        else if (mode == 1) {
            /* Horizontal mode: two 1D-coded runs */
            ssize_t run1, run2;
            const uint16_t *lut1, *lut2;

            /* First run uses a0's color table */
            lut1 = a0_color ? ccitt_lut_black : ccitt_lut_white;
            run1 = ccitt_decode_run(bs, lut1);
            if (run1 < 0) {
                return CCITT_VALUE_ERROR;
            }

            /* Second run uses opposite color table */
            lut2 = a0_color ? ccitt_lut_white : ccitt_lut_black;
            run2 = ccitt_decode_run(bs, lut2);
            if (run2 < 0) {
                return CCITT_VALUE_ERROR;
            }

            /* Fill first run with a0's color */
            if (start + run1 > rowlen) {
                run1 = rowlen - start;
            }
            if (run1 > 0) {
                memset(
                    dst + start,
                    (uint8_t)a0_color,
                    (size_t)run1
                );
            }
            start += run1;

            /* Fill second run with opposite color */
            if (start + run2 > rowlen) {
                run2 = rowlen - start;
            }
            if (run2 > 0) {
                memset(
                    dst + start,
                    (uint8_t)(a0_color ^ 1),
                    (size_t)run2
                );
            }

            a0 = start + run2;
            /* a0_color unchanged after two alternating runs */
        }
        else {
            /* Pass mode: advance a0 to b2, color unchanged */
            ssize_t end = (b2 > rowlen) ? rowlen : b2;

            if (end > start) {
                memset(
                    dst + start,
                    (uint8_t)a0_color,
                    (size_t)(end - start)
                );
            }
            a0 = end;
        }
    }

    /* Fill any remaining pixels with white */
    if (a0 >= 0 && a0 < rowlen) {
        memset(dst + a0, 0, (size_t)(rowlen - a0));
    }
    else if (a0 < 0) {
        memset(dst, 0, (size_t)rowlen);
    }

    return 0;
}


/********************************* CCITTRLE **********************************/

/* Section 10: Modified Huffman Compression. TIFF Revision 6.0 Final. 1992

TIFF compression scheme 2, a method for compressing bilevel data based on the
CCITT Group 3 1D facsimile compression scheme.

*/

/* Return length of uncompressed CCITTRLE. */
ssize_t ccitt_rle_decode_size(
    const uint8_t* src,
    const ssize_t srcsize,
    const ssize_t rowlen)
{
    ccitt_bitstream_t bs;
    ssize_t nrows = 0;
    int eol_found;

    if ((src == NULL) || (srcsize < 0) || (rowlen <= 0)) {
        return CCITT_VALUE_ERROR;
    }

    if (!ccitt_lut_initialized) {
        ccitt_lut_init();
    }

    ccitt_bitstream_init(&bs, src, srcsize);

    /* Count rows by decoding run-length data until the stream is exhausted
     * or an unrecognisable code is encountered.
     *
     * CCITTRLE rows are self-delimiting (exactly rowlen pixels per row) and
     * carry no mandatory row-separator.  Some encoders do emit an EOL code
     * between rows; we consume it if present but do not stop on its absence.
     */
    for (;;) {
        ssize_t run;
        int color = 0;
        ssize_t pos = 0;

        /* Decode one line's worth of runs to advance past it */
        while (pos < rowlen) {
            const uint16_t *lut =
                color ? ccitt_lut_black : ccitt_lut_white;
            run = ccitt_decode_run(&bs, lut);
            if (run < 0) {
                goto ccitt_rle_size_done;
            }
            pos += run;
            color ^= 1;
        }
        nrows++;

        /* TIFF compression 2 pads each row to a byte boundary */
        {
            int pad = bs.bits % 8;
            if (pad > 0) {
                ccitt_bitstream_skip(&bs, pad);
            }
        }

        /* Consume optional EOL between rows; do not stop on its absence */
        eol_found = ccitt_find_eol(&bs, 0);
        (void)eol_found;

        /* Stop when the stream is empty */
        ccitt_bitstream_fill(&bs, 1);
        if (bs.bits == 0) {
            break;
        }
    }

ccitt_rle_size_done:
    if (nrows < 1) {
        return CCITT_VALUE_ERROR;
    }
    return ccitt_safe_product(nrows, rowlen);
}


/* Decode CCITTRLE. */
ssize_t ccitt_rle_decode(
    const uint8_t* src,
    const ssize_t srcsize,
    uint8_t* dst,
    const ssize_t dstsize,
    const ssize_t rowlen)
{
    ccitt_bitstream_t bs;
    ssize_t nrows;
    ssize_t row;

    if (
        (src == NULL) || (srcsize < 0) || (dst == NULL) || (dstsize < 0)
        || (rowlen <= 0))
    {
        return CCITT_VALUE_ERROR;
    }

    if (!ccitt_lut_initialized) {
        ccitt_lut_init();
    }

    nrows = dstsize / rowlen;
    if (nrows < 1) {
        return CCITT_VALUE_ERROR;
    }

    ccitt_bitstream_init(&bs, src, srcsize);

    for (row = 0; row < nrows; row++) {
        uint8_t *line = dst + row * rowlen;
        int ret;

        ret = ccitt_decode_1d_line(&bs, line, rowlen);
        if (ret < 0) {
            return (ssize_t)ret;
        }

        /* TIFF compression 2 pads each row to a byte boundary */
        {
            int pad = bs.bits % 8;
            if (pad > 0) {
                ccitt_bitstream_skip(&bs, pad);
            }
        }

        /* Try to find EOL for the next row.
         * Missing EOL at end of data is tolerated. */
        ccitt_find_eol(&bs, 0);
    }

    return ccitt_safe_product(nrows, rowlen);
}


/************************* CCITT T.4 Group 3 (FAX3) **************************/

/* CCITT Bilevel Encodings. TIFF Revision 6.0, Section 10. 1992
   T.4: Standardization of Group 3 Facsimile Apparatus for Document
   Transmission. ITU-T Recommendation T.4. 1999

   TIFF compression scheme 3: CCITT Group 3 facsimile encoding.
   t4options bit 0: 0 = 1D encoding, 1 = 2D encoding (mixed 1D/2D)
   t4options bit 2: 0 = no fill bits before EOL, 1 = fill bits before EOL
   (bit 1: uncompressed mode, not implemented)
*/

/* Return length of decoded CCITTFAX3 data. */
ssize_t ccitt_fax3_decode_size(
    const uint8_t* src,
    const ssize_t srcsize,
    const ssize_t rowlen,
    const int t4options)
{
    ccitt_bitstream_t bs;
    const int twod = (t4options & 1);
    const int byte_align = (t4options & 4) ? 1 : 0;
    ssize_t nrows = 0;
    uint8_t *line = NULL;
    uint8_t *refline = NULL;

    if ((src == NULL) || (srcsize < 0) || (rowlen <= 0)) {
        return CCITT_VALUE_ERROR;
    }

    if (!ccitt_lut_initialized) {
        ccitt_lut_init();
    }

    /* Scratch buffer for decoding lines (needed to advance bitstream) */
    line = (uint8_t *)calloc((size_t)rowlen, sizeof(uint8_t));
    if (line == NULL) {
        return CCITT_MEMORY_ERROR;
    }
    if (twod) {
        refline = (uint8_t *)calloc((size_t)rowlen, sizeof(uint8_t));
        if (refline == NULL) {
            free(line);
            return CCITT_MEMORY_ERROR;
        }
    }

    ccitt_bitstream_init(&bs, src, srcsize);

    /* Skip initial EOL if present.  Some streams contain preamble data
       before the first EOL; fall back to a bit-by-bit scan to skip past it.
    */
    if (ccitt_find_eol(&bs, byte_align) <= 0) {
        ccitt_scan_to_eol(&bs, rowlen * 2);
    }

    for (;;) {
        int ret;

        if (twod) {
            int tag;
            ccitt_bitstream_fill(&bs, 1);
            if (bs.bits < 1) {
                break;
            }
            tag = (int)ccitt_bitstream_peek(&bs, 1);
            ccitt_bitstream_skip(&bs, 1);

            if (tag) {
                ret = ccitt_decode_1d_line(&bs, line, rowlen);
            }
            else {
                ret = ccitt_decode_2d_line(&bs, line, refline, rowlen);
            }
            if (ret < 0) {
                break;
            }
            memcpy(refline, line, (size_t)rowlen);
        }
        else {
            ret = ccitt_decode_1d_line(&bs, line, rowlen);
            if (ret < 0) {
                break;
            }
        }

        nrows++;

        /* Look for EOL separating rows */
        if (ccitt_find_eol(&bs, byte_align) <= 0) {
            break;
        }
    }

    free(line);
    if (refline != NULL) {
        free(refline);
    }

    if (nrows < 1) {
        return CCITT_VALUE_ERROR;
    }
    return ccitt_safe_product(nrows, rowlen);
}


/* Decode CCITTFAX3. */
ssize_t ccitt_fax3_decode(
    const uint8_t* src,
    const ssize_t srcsize,
    uint8_t* dst,
    const ssize_t dstsize,
    const ssize_t rowlen,
    const int t4options)
{
    ccitt_bitstream_t bs;
    const int twod = (t4options & 1);     /* bit 0: 2D encoding */
    const int byte_align = (t4options & 4) ? 1 : 0;  /* bit 2: fill bits */
    ssize_t nrows;
    ssize_t row;
    uint8_t *refline = NULL;

    if (
        (src == NULL) || (srcsize < 0) || (dst == NULL) || (dstsize < 0)
        || (rowlen <= 0))
    {
        return CCITT_VALUE_ERROR;
    }

    if (!ccitt_lut_initialized) {
        ccitt_lut_init();
    }

    nrows = dstsize / rowlen;
    if (nrows < 1) {
        return CCITT_VALUE_ERROR;
    }

    ccitt_bitstream_init(&bs, src, srcsize);

    if (twod) {
        /* Allocate reference line buffer for 2D decoding.
         * The first reference line is all white. */
        refline = (uint8_t *)calloc((size_t)rowlen, sizeof(uint8_t));
        if (refline == NULL) {
            return CCITT_MEMORY_ERROR;
        }
    }

    /* Skip initial EOL if present.  Some streams contain preamble data
     * before the first EOL; fall back to a bit-by-bit scan (similar to
     * libtiff's Fax3Sync) to skip past it. */
    if (ccitt_find_eol(&bs, byte_align) <= 0) {
        ccitt_scan_to_eol(&bs, rowlen * 2);
    }

    for (row = 0; row < nrows; row++) {
        uint8_t *line = dst + row * rowlen;
        int ret;

        if (twod) {
            /* In 2D mode, a tag bit after the EOL indicates the encoding:
             *   1 = 1D-coded line
             *   0 = 2D-coded line */
            int tag;

            ccitt_bitstream_fill(&bs, 1);
            if (bs.bits < 1) {
                break;  /* input exhausted */
            }
            tag = (int)ccitt_bitstream_peek(&bs, 1);
            ccitt_bitstream_skip(&bs, 1);

            if (tag) {
                ret = ccitt_decode_1d_line(&bs, line, rowlen);
            }
            else {
                ret = ccitt_decode_2d_line(&bs, line, refline, rowlen);
            }

            if (ret < 0) {
                free(refline);
                return (ssize_t)ret;
            }

            /* Current line becomes reference for the next line */
            memcpy(refline, line, (size_t)rowlen);
        }
        else {
            /* Pure 1D mode */
            ret = ccitt_decode_1d_line(&bs, line, rowlen);
            if (ret < 0) {
                return (ssize_t)ret;
            }
        }

        /* Try to find EOL for the next row.
         * Missing EOL at end of data is tolerated. */
        ccitt_find_eol(&bs, byte_align);
    }

    if (refline != NULL) {
        free(refline);
    }

    return ccitt_safe_product(nrows, rowlen);
}


/************************* CCITT T.6 Group 4 (FAX4) **************************/

/* CCITT Bilevel Encodings. TIFF Revision 6.0, Section 10. 1992
   T.6: Facsimile Coding Schemes and Coding Control Functions for Group 4
   Facsimile Apparatus. ITU-T Recommendation T.6. 1988

   TIFF compression scheme 4: CCITT Group 4 facsimile encoding.
   Always 2D. Terminated by EOFB (two consecutive EOL codes).
*/

/* Return length of decoded CCITTFAX4 data. */
ssize_t ccitt_fax4_decode_size(
    const uint8_t* src,
    const ssize_t srcsize,
    const ssize_t rowlen)
{
    ccitt_bitstream_t bs;
    ssize_t nrows = 0;
    uint8_t *line = NULL;
    uint8_t *refline = NULL;

    if ((src == NULL) || (srcsize < 0) || (rowlen <= 0)) {
        return CCITT_VALUE_ERROR;
    }

    if (!ccitt_lut_initialized) {
        ccitt_lut_init();
    }

    /* Scratch buffers for decoding */
    line = (uint8_t *)calloc((size_t)rowlen, sizeof(uint8_t));
    if (line == NULL) {
        return CCITT_MEMORY_ERROR;
    }
    refline = (uint8_t *)calloc((size_t)rowlen, sizeof(uint8_t));
    if (refline == NULL) {
        free(line);
        return CCITT_MEMORY_ERROR;
    }

    ccitt_bitstream_init(&bs, src, srcsize);

    for (;;) {
        int ret = ccitt_decode_2d_line(&bs, line, refline, rowlen);
        if (ret < 0) {
            break;  /* EOFB or invalid data */
        }
        nrows++;
        memcpy(refline, line, (size_t)rowlen);
    }

    free(line);
    free(refline);

    if (nrows < 1) {
        return CCITT_VALUE_ERROR;
    }
    return ccitt_safe_product(nrows, rowlen);
}


/* Decode CCITTFAX4. */
ssize_t ccitt_fax4_decode(
    const uint8_t* src,
    const ssize_t srcsize,
    uint8_t* dst,
    const ssize_t dstsize,
    const ssize_t rowlen)
{
    ccitt_bitstream_t bs;
    ssize_t nrows;
    ssize_t row;
    uint8_t *refline;

    if (
        (src == NULL) || (srcsize < 0) || (dst == NULL) || (dstsize < 0)
        || (rowlen <= 0))
    {
        return CCITT_VALUE_ERROR;
    }

    if (!ccitt_lut_initialized) {
        ccitt_lut_init();
    }

    nrows = dstsize / rowlen;
    if (nrows < 1) {
        return CCITT_VALUE_ERROR;
    }

    /* Reference line starts as all white */
    refline = (uint8_t *)calloc((size_t)rowlen, sizeof(uint8_t));
    if (refline == NULL) {
        return CCITT_MEMORY_ERROR;
    }

    ccitt_bitstream_init(&bs, src, srcsize);

    for (row = 0; row < nrows; row++) {
        uint8_t *line = dst + row * rowlen;
        int ret;

        ret = ccitt_decode_2d_line(&bs, line, refline, rowlen);
        if (ret < 0) {
            /* Check for EOFB (two consecutive EOL codes = 24 zero-bits
             * followed by two 1-bits, but the 2D decoder may have already
             * consumed partial codes).  Treat decode error after valid
             * rows as end-of-data. */
            if (row > 0) {
                /* Fill remaining rows with white */
                ssize_t remaining = (nrows - row) * rowlen;
                memset(line, 0, (size_t)remaining);
                break;
            }
            free(refline);
            return (ssize_t)ret;
        }

        /* Current line becomes reference for the next line */
        memcpy(refline, line, (size_t)rowlen);
    }

    free(refline);
    return ccitt_safe_product(nrows, rowlen);
}
