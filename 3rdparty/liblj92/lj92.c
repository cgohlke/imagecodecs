/* lj92.c */

/* This file is a modified version of

https://bitbucket.org/baldand/mlrawviewer/src/master/liblj92/lj92.c

(c) Andrew Baldwin 2014

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

/*
Modifications applied by Christoph Gohlke:

- Lint code.
- Remove SLOW_HUFF and DEBUG code.
- Remove unnecessary code and variables.
- Use stdint types.
- Use const qualifiers.
- Sort struct fields by size.
- Fix compiler warnings.
- For invalid Huffman table index, use last instead of first table.
- Add __builtin_clz function for MSVC.
- https://patch-diff.githubusercontent.com/raw/ilia3101/MLV-App/pull/151.patch
- https://patch-diff.githubusercontent.com/raw/ilia3101/MLV-App/pull/221.patch
*/

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lj92.h"

#define LJ92_MAX_COMPONENTS (16)

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>

static uint32_t __inline __builtin_clz(uint32_t x)
{
    unsigned long r = 0;
    if (_BitScanReverse(&r, x)) {
        return (31 - r);
    } else {
        return 32;
    }
}
#endif

/********************************** Decoder **********************************/

typedef struct _ljp {
    uint8_t *data;
    uint8_t *dataend;
    uint16_t *image;
    uint16_t *rowcache;
    uint16_t *outrow[2];
    const uint16_t *linearize;  // linearization table
    uint16_t *hufflut[LJ92_MAX_COMPONENTS];
    int huffbits[LJ92_MAX_COMPONENTS];
    int datalen;
    int scanstart;
    int ix;
    int x;           // width
    int y;           // height
    int bits;        // bit depth
    int components;  // components(Nf)
    int writelen;    // write rows this long
    int skiplen;     // skip this many values after each row
    int linlen;
    // int sssshist[16];
    int num_huff_idx;
    int cnt;
    uint32_t b;
} ljp;

static int
find(ljp *self)
{
    int ix = self->ix;
    uint8_t *data = self->data;
    while (ix < (self->datalen - 1) && data[ix] != 0xFF) {
        ix += 1;
    }
    ix += 2;
    if (ix >= self->datalen) {
        return -1;
    }
    self->ix = ix;
    return data[ix - 1];
}

// swap endian
#define BEH(ptr) ((((int)(*&ptr)) << 8) | (*(&ptr + 1)))

static int
parseHuff(ljp *self)
{
    uint8_t *huffhead = &self->data[self->ix];
    uint8_t *bits = &huffhead[2];
    bits[0] = 0;  // table starts from 1
    int hufflen = BEH(huffhead[0]);
    if ((self->ix + hufflen) >= self->datalen) {
        return LJ92_ERROR_CORRUPT;
    }
    // calculate Huffman direct lut
    // how many bits in the table - find highest entry
    uint8_t *huffvals = &self->data[self->ix + 19];
    int maxbits = 16;
    while (maxbits > 0) {
        if (bits[maxbits]) {
            break;
        }
        maxbits--;
    }
    self->huffbits[self->num_huff_idx] = maxbits;
    // fill the lut
    uint16_t *hufflut =
        (uint16_t *)calloc(((size_t)1 << maxbits), sizeof(uint16_t));
    if (hufflut == NULL) {
        return LJ92_ERROR_NO_MEMORY;
    }
    self->hufflut[self->num_huff_idx] = hufflut;
    int i = 0;
    int hv = 0;
    int rv = 0;
    int vl = 0;  // i
    int hcode;
    int bitsused = 1;
    while (i < 1 << maxbits) {
        if (bitsused > maxbits) {
            break;  // should never get here
        }
        if (vl >= bits[bitsused]) {
            bitsused++;
            vl = 0;
            continue;
        }
        if (rv == 1 << (maxbits - bitsused)) {
            rv = 0;
            vl++;
            hv++;
            continue;
        }
        hcode = huffvals[hv];
        hufflut[i] = (uint16_t)(hcode << 8 | bitsused);
        i++;
        rv++;
    }
    self->num_huff_idx++;

    return LJ92_ERROR_NONE;
}

static int
parseSof3(ljp *self)
{
    if (self->ix + 8 >= self->datalen) {
        return LJ92_ERROR_CORRUPT;
    }
    self->y = BEH(self->data[self->ix + 3]);
    self->x = BEH(self->data[self->ix + 5]);
    self->bits = self->data[self->ix + 2];
    self->components = self->data[self->ix + 7];
    self->ix += BEH(self->data[self->ix]);
    if ((self->components < 1) || (self->components >= 6)) {
        return LJ92_ERROR_CORRUPT;
    }
    return LJ92_ERROR_NONE;
}

static int
parseBlock(ljp *self)
{
    self->ix += BEH(self->data[self->ix]);
    if (self->ix >= self->datalen) {
        return LJ92_ERROR_CORRUPT;
    }
    return LJ92_ERROR_NONE;
}

inline static int
nextdiff(ljp *self, int component_idx, int *errcode)
{
    if (component_idx > self->num_huff_idx) {
        if (errcode) {
            (*errcode) = LJ92_ERROR_CORRUPT;
        }
        return 0;
    }

    uint32_t b = self->b;
    int cnt = self->cnt;
    int huffbits = self->huffbits[component_idx];
    int ix = self->ix;
    int next;
    while (cnt < huffbits) {
        next = *(uint16_t *)&self->data[ix];
        int one = next & 0xFF;
        int two = next >> 8;
        b = (b << 16) | (one << 8) | two;
        cnt += 16;
        ix += 2;
        if (one == 0xFF) {
            b >>= 8;
            cnt -= 8;
        } else if (two == 0xFF) {
            ix++;
        }
    }
    int index = b >> (cnt - huffbits);
    uint16_t ssssused = self->hufflut[component_idx][index];
    int usedbits = ssssused & 0xFF;
    int t = ssssused >> 8;
    // self->sssshist[t]++;
    cnt -= usedbits;
    int keepbitsmask = (1 << cnt) - 1;
    b &= keepbitsmask;
    int diff;
    if (t == 16) {
        diff = 1 << 15;
    } else {
        while (cnt < t) {
            next = *(uint16_t *)&self->data[ix];
            int one = next & 0xFF;
            int two = next >> 8;
            b = (b << 16) | (one << 8) | two;
            cnt += 16;
            ix += 2;
            if (one == 0xFF) {
                b >>= 8;
                cnt -= 8;
            } else if (two == 0xFF) {
                ix++;
            }
        }
        cnt -= t;
        diff = b >> cnt;
        int vt = 1 << (t - 1);
        if (diff < vt) {
            vt = 1 - (1 << t);
            diff += vt;
        }
    }
    keepbitsmask = (1 << cnt) - 1;
    self->b = b & keepbitsmask;
    self->cnt = cnt;
    self->ix = ix;

    return diff;
}

static int
parsePred6(ljp *self)
{
    // TODO: support self->components
    self->ix = self->scanstart;
    // int compcount = self->data[self->ix + 2];
    self->ix += BEH(self->data[self->ix]);
    self->cnt = 0;
    self->b = 0;
    int write = self->writelen;
    // decode Huffman coded values
    int c = 0;
    int pixels = self->y * self->x;
    uint16_t *out = self->image;
    uint16_t *temprow;
    uint16_t *thisrow = self->outrow[0];
    uint16_t *lastrow = self->outrow[1];

    // first pixel predicted from base value
    int diff;
    int Px = 0;
    int col = 0;
    int row = 0;
    int left = 0;
    int linear;

    if (self->num_huff_idx > self->components) {
        return LJ92_ERROR_CORRUPT;
    }
    int errcode = LJ92_ERROR_NONE;
    // first pixel
    diff = nextdiff(self, self->num_huff_idx - 1, &errcode);
    if (errcode != LJ92_ERROR_NONE) {
        return errcode;
    }
    Px = 1 << (self->bits - 1);
    left = Px + diff;
    left = (uint16_t)(left % 65536);
    if (self->linearize) {
        linear = self->linearize[left];
    } else {
        linear = left;
    }
    thisrow[col++] = (uint16_t)left;
    out[c++] = (uint16_t)linear;
    if (self->ix >= self->datalen) {
        return LJ92_ERROR_CORRUPT;
    }
    --write;
    int rowcount = self->x - 1;
    while (rowcount--) {
        int _errcode = LJ92_ERROR_NONE;
        diff = nextdiff(self, self->num_huff_idx - 1, &_errcode);
        if (_errcode != LJ92_ERROR_NONE) {
            return _errcode;
        }
        Px = left;
        left = Px + diff;
        left = (uint16_t)(left % 65536);
        if (self->linearize) {
            linear = self->linearize[left];
        } else {
            linear = left;
        }

        thisrow[col++] = (uint16_t)left;
        out[c++] = (uint16_t)linear;
        if (self->ix >= self->datalen) {
            return LJ92_ERROR_CORRUPT;
        }
        if (--write == 0) {
            out += self->skiplen;
            write = self->writelen;
        }
    }
    temprow = lastrow;
    lastrow = thisrow;
    thisrow = temprow;
    row++;
    while (c < pixels) {
        col = 0;
        int _errcode = LJ92_ERROR_NONE;
        diff = nextdiff(self, self->num_huff_idx - 1, &_errcode);
        if (_errcode != LJ92_ERROR_NONE) {
            return _errcode;
        }

        Px = lastrow[col];  // use value above for first pixel in row
        left = Px + diff;
        left = (uint16_t)(left % 65536);
        if (self->linearize) {
            if (left > self->linlen) {
                return LJ92_ERROR_CORRUPT;
            }
            linear = self->linearize[left];
        } else {
            linear = left;
        }
        thisrow[col++] = (uint16_t)left;
        out[c++] = (uint16_t)linear;
        if (self->ix >= self->datalen) {
            break;
        }
        rowcount = self->x - 1;
        if (--write == 0) {
            out += self->skiplen;
            write = self->writelen;
        }
        while (rowcount--) {
            int errcode_d2 = LJ92_ERROR_NONE;
            diff = nextdiff(self, self->num_huff_idx - 1, &errcode_d2);
            if (errcode_d2 != LJ92_ERROR_NONE) {
                return errcode_d2;
            }

            Px = lastrow[col] + ((left - lastrow[col - 1]) >> 1);
            left = Px + diff;
            left = (uint16_t)(left % 65536);
            if (self->linearize) {
                if (left > self->linlen) {
                    return LJ92_ERROR_CORRUPT;
                }
                linear = self->linearize[left];
            } else {
                linear = left;
            }
            thisrow[col++] = (uint16_t)left;
            out[c++] = (uint16_t)linear;
            if (--write == 0) {
                out += self->skiplen;
                write = self->writelen;
            }
        }
        temprow = lastrow;
        lastrow = thisrow;
        thisrow = temprow;
        if (self->ix >= self->datalen) {
            break;
        }
    }
    if (c >= pixels) {
        return LJ92_ERROR_NONE;
    }
    return LJ92_ERROR_CORRUPT;
}

static int
parseScan(ljp *self)
{
    // memset(self->sssshist, 0, sizeof(self->sssshist));
    self->ix = self->scanstart;
    int compcount = self->data[self->ix + 2];
    int pred = self->data[self->ix + 3 + 2 * compcount];
    if (pred > 7) {
        return LJ92_ERROR_CORRUPT;
    }

    // disable fast path for multiple components until parsePred6 supports it
    if ((pred == 6) && (self->components == 1)) {
        return parsePred6(self);
    }

    self->ix += BEH(self->data[self->ix]);
    self->cnt = 0;
    self->b = 0;
    uint16_t *out = self->image;
    uint16_t *thisrow = self->outrow[0];
    uint16_t *lastrow = self->outrow[1];

    // first pixel predicted from base value
    int diff;
    int Px = 0;
    int left = 0;
    for (int row = 0; row < self->y; row++) {
        for (int col = 0; col < self->x; col++) {
            int colx = col * self->components;
            for (int c = 0; c < self->components; c++) {
                if ((col == 0) && (row == 0)) {
                    Px = 1 << (self->bits - 1);
                } else if (row == 0) {
                    // Px = left;
                    Px = thisrow[(col - 1) * self->components + c];
                } else if (col == 0) {
                    Px = lastrow[c];  // use value above for first pixel in row
                } else {
                    int prev_colx = (col - 1) * self->components;
                    // previous pixel
                    left = thisrow[prev_colx + c];
                    switch (pred) {
                        case 0:
                            Px = 0;
                            break;  // no prediction... should not be used
                        case 1:
                            Px = left;  // thisrow[prev_colx + c];
                            break;
                        case 2:
                            Px = lastrow[colx + c];
                            break;
                        case 3:
                            Px = lastrow[prev_colx + c];
                            break;
                        case 4:
                            Px = left + lastrow[colx + c] -
                                 lastrow[prev_colx + c];
                            break;
                        case 5:
                            Px = left + ((lastrow[colx + c] -
                                          lastrow[prev_colx + c]) >>
                                         1);
                            break;
                        case 6:
                            Px = lastrow[colx + c] +
                                 ((left - lastrow[prev_colx + c]) >> 1);
                            break;
                        case 7:
                            Px = (left + lastrow[colx + c]) >> 1;
                            break;
                    }
                }

                int huff_idx = c;
                if (c >= self->num_huff_idx) {
                    // invalid Huffman table index, use last table
                    huff_idx = self->num_huff_idx - 1;
                }

                int errcode = LJ92_ERROR_NONE;
                diff = nextdiff(self, huff_idx, &errcode);
                if (errcode != LJ92_ERROR_NONE) {
                    return errcode;
                }
                left = Px + diff;
                left = (uint16_t)(left % 65536);
                int linear;
                if (self->linearize) {
                    if (left > self->linlen) {
                        return LJ92_ERROR_CORRUPT;
                    }
                    linear = self->linearize[left];
                } else {
                    linear = left;
                }

                thisrow[colx + c] = (uint16_t)left;
                out[colx + c] = (uint16_t)linear;
            }
        }  // col

        // swap pointers for input and working row buffer
        uint16_t *temprow = lastrow;
        lastrow = thisrow;
        thisrow = temprow;

        out += self->x * self->components + self->skiplen;
    }  // row

    return LJ92_ERROR_NONE;
}

static int
parseImage(ljp *self)
{
    int ret = LJ92_ERROR_NONE;
    while (1) {
        int nextMarker = find(self);
        if (nextMarker == 0xc4) {
            ret = parseHuff(self);
        } else if (nextMarker == 0xc3) {
            ret = parseSof3(self);
        } else if (nextMarker == 0xfe) {
            // Comment
            ret = parseBlock(self);
        } else if (nextMarker == 0xd9) {
            // End of image
            break;
        } else if (nextMarker == 0xda) {
            self->scanstart = self->ix;
            ret = LJ92_ERROR_NONE;
            break;
        } else if (nextMarker == -1) {
            ret = LJ92_ERROR_CORRUPT;
            break;
        } else {
            ret = parseBlock(self);
        }
        if (ret != LJ92_ERROR_NONE) {
            break;
        }
    }
    return ret;
}

static int
findSoI(ljp *self)
{
    if (find(self) == 0xd8) {
        return parseImage(self);
    }
    return LJ92_ERROR_CORRUPT;
}

static void
free_memory(ljp *self)
{
    for (int i = 0; i < self->num_huff_idx; i++) {
        free(self->hufflut[i]);
        self->hufflut[i] = NULL;
    }
    free(self->rowcache);
    self->rowcache = NULL;
}

int
lj92_open(
    lj92 *lj,
    const uint8_t *data,
    int datalen,
    int *width,
    int *height,
    int *bitdepth,
    int *components)
{
    ljp *self = (ljp *)calloc(1, sizeof(ljp));
    if (self == NULL) {
        return LJ92_ERROR_NO_MEMORY;
    }

    self->data = (uint8_t *)data;
    self->dataend = self->data + datalen;
    self->datalen = datalen;
    self->num_huff_idx = 0;

    int ret = findSoI(self);

    if (ret == LJ92_ERROR_NONE) {
        uint16_t *rowcache = (uint16_t *)calloc(
            self->x * self->components * 2, sizeof(uint16_t));
        if (rowcache == NULL) {
            ret = LJ92_ERROR_NO_MEMORY;
        } else {
            self->rowcache = rowcache;
            self->outrow[0] = rowcache;
            self->outrow[1] = &rowcache[self->x * self->components];
        }
    }

    if (ret != LJ92_ERROR_NONE) {
        *lj = NULL;
        free_memory(self);
        free(self);
    } else {
        *width = self->x;
        *height = self->y;
        *bitdepth = self->bits;
        *components = self->components;
        *lj = self;
    }
    return ret;
}

int
lj92_decode(
    lj92 lj,
    uint16_t *target,
    int writeLength,
    int skipLength,
    const uint16_t *linearize,
    int linearizeLength)
{
    ljp *self = lj;
    if (self == NULL) {
        return LJ92_ERROR_BAD_HANDLE;
    }
    self->image = target;
    self->writelen = writeLength;
    self->skiplen = skipLength;
    self->linearize = linearize;
    self->linlen = linearizeLength;
    return parseScan(self);
}

void
lj92_close(lj92 lj)
{
    ljp *self = lj;
    if (self != NULL) {
        free_memory(self);
    }
    free(self);
}

/********************************** Encoder **********************************/

typedef struct _lje {
    const uint16_t *image;
    const uint16_t *delinearize;
    uint8_t *encoded;
    int width;
    int height;
    int bitdepth;
    int components;
    int readLength;
    int skipLength;
    int delinearizeLength;
    int encodedWritten;
    int encodedLength;
    int hist[18];  // SSSS frequency histogram
    int bits[18];
    int huffval[18];
    int huffsym[18];
    uint16_t huffenc[18];
    uint16_t huffbits[18];
} lje;

int
frequencyScan(lje *self)
{
    // scan through the tile using the standard type 6 prediction
    // need to cache the previous 2 row in target coordinates because of tiling
    const uint16_t *pixel = self->image;
    int pixcount = self->width * self->height;
    int scan = self->readLength;
    uint16_t *rowcache = (uint16_t *)calloc(
        self->width * self->components * 2, sizeof(uint16_t));
    if (rowcache == NULL) {
        return LJ92_ERROR_NO_MEMORY;
    }
    uint16_t *rows[2];
    rows[0] = rowcache;
    rows[1] = &rowcache[self->width * self->components];

    int col = 0;
    int row = 0;
    int Px = 0;
    int32_t diff = 0;
    // int maxval = (1 << self->bitdepth);
    while (pixcount--) {
        uint16_t p = *pixel;
        /*
        if (self->delinearize) {
            if (p >= self->delinearizeLength) {
                free(rowcache);
                return LJ92_ERROR_TOO_WIDE;
            }
            p = self->delinearize[p];
        }
        if (p >= maxval) {
            free(rowcache);
            return LJ92_ERROR_TOO_WIDE;
        }
        */
        rows[1][col] = p;

        if ((row == 0) && (col == 0)) {
            Px = 1 << (self->bitdepth - 1);
        } else if (row == 0) {
            Px = rows[1][col - 1];
        } else if (col == 0) {
            Px = rows[0][col];
        } else {
            Px = rows[0][col] + ((rows[1][col - 1] - rows[0][col - 1]) >> 1);
        }
        diff = rows[1][col] - Px;
        diff = diff % 65536;
        diff = (int16_t)diff;
        int ssss = 32 - __builtin_clz(abs(diff));
        if (diff == 0) {
            ssss = 0;
        }
        self->hist[ssss]++;
        pixel++;
        scan--;
        col++;
        if (scan == 0) {
            pixel += self->skipLength;
            scan = self->readLength;
        }
        if (col == self->width) {
            uint16_t *tmprow = rows[1];
            rows[1] = rows[0];
            rows[0] = tmprow;
            col = 0;
            row++;
        }
    }
    free(rowcache);
    return LJ92_ERROR_NONE;
}

void
createEncodeTable(lje *self)
{
    float freq[18];
    int codesize[18];
    int others[18];

    // frequencies
    float totalpixels = (float)(self->width * self->height);
    for (int i = 0; i < 17; i++) {
        freq[i] = (float)(self->hist[i]) / totalpixels;
        codesize[i] = 0;
        others[i] = -1;
    }
    codesize[17] = 0;
    others[17] = -1;
    freq[17] = 1.0f;

    float v1f, v2f;
    int v1, v2;

    while (1) {
        v1f = 3.0f;
        v1 = -1;
        for (int i = 0; i < 18; i++) {
            if ((freq[i] <= v1f) && (freq[i] > 0.0f)) {
                v1f = freq[i];
                v1 = i;
            }
        }
        v2f = 3.0f;
        v2 = -1;
        for (int i = 0; i < 18; i++) {
            if (i == v1) {
                continue;
            }
            if ((freq[i] < v2f) && (freq[i] > 0.0f)) {
                v2f = freq[i];
                v2 = i;
            }
        }
        if (v2 == -1) {
            break;  // done
        }

        freq[v1] += freq[v2];
        freq[v2] = 0.0f;

        while (1) {
            codesize[v1]++;
            if (others[v1] == -1) {
                break;
            }
            v1 = others[v1];
        }
        others[v1] = v2;
        while (1) {
            codesize[v2]++;
            if (others[v2] == -1) {
                break;
            }
            v2 = others[v2];
        }
    }
    int *bits = self->bits;
    memset(bits, 0, sizeof(self->bits));
    for (int i = 0; i < 18; i++) {
        if (codesize[i] != 0) {
            bits[codesize[i]]++;
        }
    }
    // adjust bits, this step is a must to remove a code with all ones
    // and fix bug of overriding SSSS-0 category with the code with all ones
    int I = 17;
    while (1) {
        if (bits[I] > 0) {
            int J = I - 1;
            do {
                J = J - 1;
            } while (bits[J] <= 0);
            bits[I] = bits[I] - 2;
            bits[I - 1] = bits[I - 1] + 1;
            bits[J + 1] = bits[J + 1] + 2;
            bits[J] = bits[J] - 1;
        } else {
            I = I - 1;
            if (I != 16) {
                continue;
            }
            while (bits[I] == 0) {
                I = I - 1;
            }
            bits[I] = bits[I] - 1;
            break;
        }
    }
    int *huffval = self->huffval;
    int i = 1;
    int k = 0;
    int j;
    memset(huffval, 0, sizeof(self->huffval));
    while (i <= 32) {
        j = 0;
        while (j < 17) {
            if (codesize[j] == i) {
                huffval[k++] = j;
            }
            j++;
        }
        i++;
    }
    int maxbits = 16;
    while (maxbits > 0) {
        if (bits[maxbits]) {
            break;
        }
        maxbits--;
    }
    uint16_t *huffenc = self->huffenc;
    uint16_t *huffbits = self->huffbits;
    int *huffsym = self->huffsym;
    memset(huffenc, 0, sizeof(self->huffenc));
    memset(huffbits, 0, sizeof(self->huffbits));
    memset(self->huffsym, 0, sizeof(self->huffsym));
    i = 0;
    int hv = 0;
    int rv = 0;
    int vl = 0;  // i
    int bitsused = 1;
    int sym = 0;
    while (i < 1 << maxbits) {
        if (bitsused > maxbits) {
            break;  // should never get here!
        }
        if (vl >= bits[bitsused]) {
            bitsused++;
            vl = 0;
            continue;
        }
        if (rv == 1 << (maxbits - bitsused)) {
            rv = 0;
            vl++;
            hv++;
            continue;
        }
        huffbits[sym] = (uint16_t)bitsused;
        huffenc[sym++] = (uint16_t)(i >> (maxbits - bitsused));
        i += (1 << (maxbits - bitsused));
        rv = 1 << (maxbits - bitsused);
    }
    for (i = 0; i < 17; i++) {
        if (huffbits[i] > 0) {
            huffsym[huffval[i]] = i;
        }
        // if (huffbits[i] > 0) {
        //     huffsym[huffval[i]] = i;
        // }
    }
}

void
writeHeader(lje *self)
{
    int w = self->encodedWritten;
    uint8_t *e = self->encoded;
    e[w++] = 0xff;
    // SOI
    e[w++] = 0xd8;
    e[w++] = 0xff;
    // HUFF
    e[w++] = 0xc4;
    int count = 0;
    for (int i = 0; i < 17; i++) {
        count += self->bits[i];
    }
    e[w++] = 0x0;
    e[w++] = (uint8_t)(17 + 2 + count);  // Lf, frame header length
    e[w++] = (uint8_t)0;                 // Table ID
    for (int i = 1; i < 17; i++) {
        e[w++] = (uint8_t)self->bits[i];
    }
    for (int i = 0; i < count; i++) {
        e[w++] = (uint8_t)self->huffval[i];
    }
    e[w++] = 0xff;
    // SOF
    e[w++] = 0xc3;
    e[w++] = 0x0;
    e[w++] = 11;  // Lf, frame header length
    e[w++] = (uint8_t)self->bitdepth;
    e[w++] = (uint8_t)(self->height >> 8);
    e[w++] = (uint8_t)(self->height & 0xFF);
    e[w++] = (uint8_t)(self->width >> 8);
    e[w++] = (uint8_t)(self->width & 0xFF);
    e[w++] = 1;     // components
    e[w++] = 0;     // component ID
    e[w++] = 0x11;  // component X/Y
    e[w++] = 0;     // unused quantisation
    e[w++] = 0xff;
    // SCAN
    e[w++] = 0xda;
    e[w++] = 0x0;
    e[w++] = 8;  // Ls, scan header length
    e[w++] = 1;  // components
    e[w++] = 0;
    e[w++] = 0;
    e[w++] = 6;  // predictor
    e[w++] = 0;
    e[w++] = 0;
    self->encodedWritten = w;
}

void
writePost(lje *self)
{
    int w = self->encodedWritten;
    uint8_t *e = self->encoded;
    e[w++] = 0xff;
    e[w++] = 0xd9;  // EOI
    self->encodedWritten = w;
}

int
writeBody(lje *self)
{
    // scan through the tile using the standard type 6 prediction
    // need to cache the previous 2 row in target coordinates because of tiling
    const uint16_t *pixel = self->image;
    int pixcount = self->width * self->height;
    int scan = self->readLength;
    uint16_t *rowcache = (uint16_t *)calloc(
        self->width * self->components * 2, sizeof(uint16_t));
    if (rowcache == NULL) {
        return LJ92_ERROR_NO_MEMORY;
    }
    uint16_t *rows[2];
    rows[0] = rowcache;
    rows[1] = &rowcache[self->width * self->components];

    int col = 0;
    int row = 0;
    int Px = 0;
    int32_t diff = 0;
    uint8_t *out = self->encoded;
    int w = self->encodedWritten;
    uint8_t next = 0;
    uint8_t nextbits = 8;
    while (pixcount--) {
        uint16_t p = *pixel;
        if (self->delinearize) {
            p = self->delinearize[p];
        }
        rows[1][col] = p;

        if ((row == 0) && (col == 0)) {
            Px = 1 << (self->bitdepth - 1);
        } else if (row == 0) {
            Px = rows[1][col - 1];
        } else if (col == 0) {
            Px = rows[0][col];
        } else {
            Px = rows[0][col] + ((rows[1][col - 1] - rows[0][col - 1]) >> 1);
        }
        diff = rows[1][col] - Px;
        diff = diff % 65536;
        diff = (int16_t)diff;
        int ssss = 32 - __builtin_clz(abs(diff));
        if (diff == 0) {
            ssss = 0;
        }

        // write the Huffman code for the ssss value
        int huffcode = self->huffsym[ssss];
        int huffenc = self->huffenc[huffcode];
        int huffbits = self->huffbits[huffcode];

        int vt = ssss > 0 ? (1 << (ssss - 1)) : 0;
        if (diff < vt) {
            diff += (1 << (ssss)) - 1;
        }

        // write the ssss
        while (huffbits > 0) {
            int usebits = huffbits > nextbits ? nextbits : huffbits;
            // add top usebits from huffval to next usebits of nextbits
            int tophuff = huffenc >> (huffbits - usebits);
            next |= (tophuff << (nextbits - usebits));
            nextbits -= (uint8_t)usebits;
            huffbits -= usebits;
            huffenc &= (1 << huffbits) - 1;
            if (nextbits == 0) {
                if (w >= self->encodedLength - 1) {
                    free(rowcache);
                    return LJ92_ERROR_ENCODER;
                }
                out[w++] = next;
                if (next == 0xff) {
                    out[w++] = 0x0;
                }
                next = 0;
                nextbits = 8;
            }
        }
        // write the rest of the bits for the value
        if (ssss == 16) {
            // diff values (always 32678) for SSSS=16 are encoded with 0 bits
            ssss = 0;
        }
        while (ssss > 0) {
            int usebits = ssss > nextbits ? nextbits : ssss;
            // add top usebits from huffval to next usebits of nextbits
            int tophuff = diff >> (ssss - usebits);
            next |= (tophuff << (nextbits - usebits));
            nextbits -= (uint8_t)usebits;
            ssss -= usebits;
            diff &= (1 << ssss) - 1;
            if (nextbits == 0) {
                if (w >= self->encodedLength - 1) {
                    free(rowcache);
                    return LJ92_ERROR_ENCODER;
                }
                out[w++] = next;
                if (next == 0xff) {
                    out[w++] = 0x0;
                }
                next = 0;
                nextbits = 8;
            }
        }

        pixel++;
        scan--;
        col++;
        if (scan == 0) {
            pixel += self->skipLength;
            scan = self->readLength;
        }
        if (col == self->width) {
            uint16_t *tmprow = rows[1];
            rows[1] = rows[0];
            rows[0] = tmprow;
            col = 0;
            row++;
        }
    }
    // flush final bits
    if (nextbits < 8) {
        out[w++] = next;
        if (next == 0xff) {
            out[w++] = 0x0;
        }
    }
    free(rowcache);
    self->encodedWritten = w;
    return LJ92_ERROR_NONE;
}

/* Encoder
 * Read tile from an image and encode in one shot
 * Return the encoded data
 */
int
lj92_encode(
    const uint16_t *image,
    int width,
    int height,
    int bitdepth,
    int components,
    int readLength,
    int skipLength,
    const uint16_t *delinearize,
    int delinearizeLength,
    uint8_t **encoded,
    int *encodedLength)
{
    int ret = LJ92_ERROR_NONE;

    lje *self = (lje *)calloc(1, sizeof(lje));
    if (self == NULL) {
        return LJ92_ERROR_NO_MEMORY;
    }
    self->image = image;
    self->width = width;
    self->height = height;
    self->bitdepth = bitdepth;
    self->readLength = readLength;
    self->skipLength = skipLength;
    self->delinearize = delinearize;
    self->delinearizeLength = delinearizeLength;
    self->encodedLength = width * height * components * 3 + 200;
    self->components = components;
    self->encoded = (uint8_t *)malloc(self->encodedLength);
    if (self->encoded == NULL) {
        free(self);
        return LJ92_ERROR_NO_MEMORY;
    }
    // scan through data to gather frequencies of ssss prefixes
    ret = frequencyScan(self);
    if (ret != LJ92_ERROR_NONE) {
        free(self->encoded);
        free(self);
        return ret;
    }
    // create encoded table based on frequencies
    createEncodeTable(self);
    // write JPEG head and scan header
    writeHeader(self);
    // scan through and do the compression
    ret = writeBody(self);
    if (ret != LJ92_ERROR_NONE) {
        free(self->encoded);
        free(self);
        return ret;
    }
    // finish
    writePost(self);
    uint8_t *temp = (uint8_t *)realloc(self->encoded, self->encodedWritten);
    if (temp == NULL) {
        free(self->encoded);
        free(self);
        return LJ92_ERROR_NO_MEMORY;
    }
    self->encoded = temp;
    self->encodedLength = self->encodedWritten;
    *encoded = self->encoded;
    *encodedLength = self->encodedLength;

    free(self);

    return ret;
}
