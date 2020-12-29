/* jpegsof3.cpp */

/* This file is a modified version of

https://github.com/rordenlab/dcm2niix/blob/master/console/jpg_0XC3.cpp

which is distributed under the BSD 3-Clause License:

The Software has been developed for research purposes only and
is not a clinical tool.

Copyright(c) 2014 - 2016 Chris Rorden. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met :
1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright owner nor the name of this project
(dcm2niix) may be used to endorse or promote products derived from this
software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT OWNER ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.IN NO
EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES(INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT(INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*
This file contains the following changes, mostly to the
'decode_JPEG_SOF_0XC3' function:

    Add license.
    Lint C++ code.
    Remove unnecessary headers.
    Remove verbose input parameter and print statements.
    Return error codes instead of printing error messages.
    Add option to query image properties from header.
    Replace input file name parameter with pointer to JPEG byte string.
    Add input pointer to output array.
    Do not allocate output image.
    Use ssize_t instead of long type.
    Rename files and decoder function to reflect API changes.
    Enable 1..16 bit, 1..255 frames (not tested).
    Fix JPEG with multiple DHTs.
    Use UNUSED macro instead of #pragma unused

The changes are released under the BSD 3-Clause License:

Copyright (c) 2018-2021, Christoph Gohlke

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

#include <stddef.h>
#include <stdint.h>

#include "jpegsof3.h"

#define UNUSED(arg) ((void)&(arg))


unsigned char readByte(
    unsigned char* lRawRA,
    ssize_t* lRawPos,
    ssize_t lRawSz)
{
    unsigned char ret = 0x00;
    if (*lRawPos < lRawSz)
        ret = lRawRA[*lRawPos];
    (*lRawPos)++;
    return ret;
}


uint16_t readWord(
    unsigned char* lRawRA,
    ssize_t* lRawPos,
    ssize_t lRawSz)
{
    return ((readByte(lRawRA, lRawPos, lRawSz) << 8) +
             readByte(lRawRA, lRawPos, lRawSz));
}


int readBit(
    unsigned char* lRawRA,
    ssize_t* lRawPos,
    int* lCurrentBitPos)
{
    // Read the next single bit
    int result = (lRawRA[*lRawPos] >> (7 - *lCurrentBitPos)) & 1;
    (*lCurrentBitPos)++;
    if (*lCurrentBitPos == 8) {
        (*lRawPos)++;
        *lCurrentBitPos = 0;
    }
    return result;
}


int bitMask(int bits)
{
    return ((2 << (bits - 1)) - 1);
}


int readBits(
    unsigned char* lRawRA,
    ssize_t* lRawPos,
    int* lCurrentBitPos,
    int lNum)
{
    // lNum: bits to read, not to exceed 16
    int result = lRawRA[*lRawPos];
    result = (result << 8) + lRawRA[(*lRawPos) + 1];
    result = (result << 8) + lRawRA[(*lRawPos) + 2];
    // lCurrentBitPos is incremented from 1, so -1
    result = (result >> (24 - *lCurrentBitPos - lNum)) & bitMask(lNum);
    *lCurrentBitPos = *lCurrentBitPos + lNum;
    if (*lCurrentBitPos > 7) {
        *lRawPos = *lRawPos + (*lCurrentBitPos >> 3);  // div 8
        *lCurrentBitPos = *lCurrentBitPos & 7;  // mod 8
    }
    return result;
}


struct HufTables {
    uint8_t SSSSszRA[18];
    uint8_t LookUpRA[256];
    int DHTliRA[32];
    int DHTstartRA[32];
    int HufSz[32];
    int HufCode[32];
    int HufVal[32];
    int MaxHufSi;
    int MaxHufVal;
};


int decodePixelDifference(
    unsigned char* lRawRA,
    ssize_t* lRawPos,
    int* lCurrentBitPos,
    const struct HufTables &l)
{
    int lByte = (lRawRA[*lRawPos] << *lCurrentBitPos) +
                (lRawRA[*lRawPos + 1] >> (8 - *lCurrentBitPos));
    lByte = lByte & 255;
    int lHufValSSSS = l.LookUpRA[lByte];
    if (lHufValSSSS < 255) {
        /* TODO: possible array overrun if lHufValSSSS > 17 */
        *lCurrentBitPos = l.SSSSszRA[lHufValSSSS] + *lCurrentBitPos;
        *lRawPos = *lRawPos + (*lCurrentBitPos >> 3);
        *lCurrentBitPos = *lCurrentBitPos & 7;
    }
    else {
       // full SSSS is not in the first 8-bits
        int lInput = lByte;
        int lInputBits = 8;
        (*lRawPos)++; // forward 8 bits = precisely 1 byte
        do {
            lInputBits++;
            lInput = (lInput << 1) + readBit(lRawRA, lRawPos, lCurrentBitPos);
            if (l.DHTliRA[lInputBits] != 0) {
                // if any entires with this length
                for (int lI = l.DHTstartRA[lInputBits];
                     lI <= (l.DHTstartRA[lInputBits]+l.DHTliRA[lInputBits]-1);
                     lI++)
                {
                    if (lInput == l.HufCode[lI]) {
                        lHufValSSSS = l.HufVal[lI];
                    }
                }
            }
            if ((lInputBits >= l.MaxHufSi) && (lHufValSSSS > 254)) {
                // exhausted options CR: added rev13
                lHufValSSSS = l.MaxHufVal;
            }
        } while (!(lHufValSSSS < 255));  // found;
    }
    // The HufVal is referred to as the SSSS in the Codec,
    // so it is called 'lHufValSSSS'
    if (lHufValSSSS == 0) {
        // NO CHANGE
        return 0;
    }
    if (lHufValSSSS == 1) {
        if (readBit(lRawRA, lRawPos, lCurrentBitPos) == 0) {
            return -1;
        }
        else {
            return 1;
        }
    }
    if (lHufValSSSS == 16) {
        // ALL CHANGE 16 bit difference: Codec H.1.2.2
        // "No extra bits are appended after SSSS = 16 is encoded."
        // Osiris fails here
        return 32768;
    }
    // to get here - there is a 2..15 bit difference
    int lDiff = readBits(lRawRA, lRawPos, lCurrentBitPos, lHufValSSSS);
    if (lDiff <= bitMask(lHufValSSSS - 1)) {
        //add
        lDiff = lDiff - bitMask(lHufValSSSS);
    }
    return lDiff;
}


/* Note: this function modifies the input array, lRawRA */

int decode_jpegsof3(
    unsigned char* lRawRA,
    ssize_t lRawSz,
    unsigned char* lImgRA8,
    ssize_t lImgSz,
    int* dimX,
    int* dimY,
    int* bits,
    int* frames)
{
    if (lRawSz < 3) {
        return JPEGSOF3_INVALID_SIGNATURE;
    }
    if ((lRawRA[0] != 0xFF) || (lRawRA[1] != 0xD8) || (lRawRA[2] != 0xFF)) {
        // JPEG signature 0xFFD8FF not found
        // http://en.wikipedia.org/wiki/List_of_file_signatures
        return JPEGSOF3_INVALID_SIGNATURE;
    }
    // next: read header
    ssize_t lRawPos = 2;  // Skip initial 0xFFD8, begin with third byte
    unsigned char btS1, btS2, SOSse, SOSahal, btMarkerType, SOSns = 0x00;
    unsigned char SOSpttrans = 0;
    unsigned char SOSss = 0;
    uint8_t SOFnf = 0;
    uint8_t SOFprecision = 0;
    uint16_t SOFydim = 0;
    uint16_t SOFxdim = 0;
    // ssize_t SOSarrayPos;  // SOFarrayPos
    int lnHufTables = 0;
    int lFrameCount = 1;
    const int kmaxFrames = 4;  // 255
    struct HufTables l[kmaxFrames + 1];

    UNUSED(btS1);
    UNUSED(btS2);
    UNUSED(SOSse);

    do {
        // read each marker in the header
        do {
            btS1 = readByte(lRawRA, &lRawPos, lRawSz);
            if (btS1 != 0xFF) {
                // JPEG header tag must begin with 0xFF
                return JPEGSOF3_INVALID_HEADER_TAG;
            }
            btMarkerType = readByte(lRawRA, &lRawPos, lRawSz);
            if ((btMarkerType == 0x01) ||
                (btMarkerType == 0xFF) ||
                ((btMarkerType >= 0xD0) && (btMarkerType <= 0xD7)))
            {
                // only process segments with length fields
                btMarkerType = 0;
            }
        } while ((lRawPos < lRawSz) && (btMarkerType == 0));
        // read marker length
        uint16_t lSegmentLength = readWord(lRawRA, &lRawPos, lRawSz);
        ssize_t lSegmentEnd = lRawPos + (lSegmentLength - 2);
        if (lSegmentEnd > lRawSz) {
            // Segment larger than image
            return JPEGSOF3_SEGMENT_GT_IMAGE;
        }
        if (((btMarkerType >= 0xC0) && (btMarkerType <= 0xC3)) ||
            ((btMarkerType >= 0xC5) && (btMarkerType <= 0xCB)) ||
            ((btMarkerType >= 0xCD) && (btMarkerType <= 0xCF)))
        {
            // if Start-Of-Frame (SOF) marker
            SOFprecision = readByte(lRawRA, &lRawPos, lRawSz);
            SOFydim = readWord(lRawRA, &lRawPos, lRawSz);
            SOFxdim = readWord(lRawRA, &lRawPos, lRawSz);
            SOFnf = readByte(lRawRA, &lRawPos, lRawSz);
            *dimX = SOFxdim;
            *dimY = SOFydim;
            *frames = SOFnf;
            *bits = (SOFprecision > 8) ? 16 : 8;
            // SOFarrayPos = lRawPos;
            lRawPos = lSegmentEnd;
            if (btMarkerType != 0xC3) {
                // Not a lossless JPEG ITU-T81 image (SoF must be 0XC3)
                return JPEGSOF3_INVALID_ITU_T81;
            }
            if ((SOFprecision < 2) || (SOFprecision > 16) ||
                (SOFnf < 1) || (SOFnf > kmaxFrames))
            {
                // Data must be 2..16 bit, 1..4 frames
                return JPEGSOF3_INVALID_BIT_DEPTH;
            }
            if (lImgRA8 == NULL) {
                return JPEGSOF3_OK;
            }
        }
        else if (btMarkerType == 0xC4) {
            // if SOF marker else if define-Huffman-tables marker (DHT)
            do {
                // we read but ignore DHTtcth.
                uint8_t DHTnLi = readByte(lRawRA, &lRawPos, lRawSz);
                UNUSED(DHTnLi);
                // we need to increment the input file position,
                // but we do not care what the value is
                // #pragma unused(DHTnLi)
                DHTnLi = 0;
                for (int lInc = 1; lInc <= 16; lInc++) {
                    l[lFrameCount].DHTliRA[lInc] = readByte(lRawRA, &lRawPos,
                                                            lRawSz);
                    DHTnLi = DHTnLi + l[lFrameCount].DHTliRA[lInc];
                    if (l[lFrameCount].DHTliRA[lInc] != 0) {
                        l[lFrameCount].MaxHufSi = lInc;
                    }
                }
                if (DHTnLi > 17) {
                    // Huffman table corrupted
                    return JPEGSOF3_TABLE_CORRUPTED;
                }
                int lIncY = 0;  // frequency
                for (int lInc = 0; lInc <= 31; lInc++) {
                    // lInc := 0 to 31 do begin
                    l[lFrameCount].HufVal[lInc] = -1;
                    l[lFrameCount].HufSz[lInc] = -1;
                    l[lFrameCount].HufCode[lInc] = -1;
                }
                for (int lInc = 1; lInc <= 16; lInc++) {
                    // set the huffman size values
                    if (l[lFrameCount].DHTliRA[lInc] > 0) {
                        l[lFrameCount].DHTstartRA[lInc] = lIncY + 1;
                        for (int lIncX = 1;
                             lIncX <= l[lFrameCount].DHTliRA[lInc];
                             lIncX++)
                        {
                            lIncY++;
                            btS1 = readByte(lRawRA, &lRawPos, lRawSz);
                            l[lFrameCount].HufVal[lIncY] = btS1;
                            l[lFrameCount].MaxHufVal = btS1;
                            if (btS1 <= 16) {
                                // unsigned ints ALWAYS > 0,
                                // so no need for(btS1 >= 0)
                                l[lFrameCount].HufSz[lIncY] = lInc;
                            }
                            else {
                                // Huffman size array corrupted
                                return JPEGSOF3_TABLE_SIZE_CORRUPTED;
                            }
                        }
                    }
                }
                int K = 1;
                int Code = 0;
                int Si = l[lFrameCount].HufSz[K];
                do {
                    while (Si == l[lFrameCount].HufSz[K]) {
                        l[lFrameCount].HufCode[K] = Code;
                        Code = Code + 1;
                        K++;
                    }
                    if (K <= DHTnLi) {
                        while (l[lFrameCount].HufSz[K] > Si) {
                            Code = Code << 1;
                            Si = Si + 1;
                        }
                    }
                } while (K <= DHTnLi);
                lFrameCount++;
            } while ((lSegmentEnd - lRawPos) >= 18);
            lnHufTables = lFrameCount - 1;
            lRawPos = lSegmentEnd;
        }
        else if (btMarkerType == 0xDD) {
            // if DHT marker else if Define restart interval (DRI) marker
            // btMarkerType == 0xDD: unsupported Restart Segments
            return JPEGSOF3_INVALID_RESTART_SEGMENTS;
            // lRestartSegmentSz = ReadWord(lRawRA, &lRawPos, lRawSz);
            // lRawPos = lSegmentEnd;
        }
        else if (btMarkerType == 0xDA) {
            // if DRI marker else if read Start of Scan (SOS) marker
            SOSns = readByte(lRawRA, &lRawPos, lRawSz);
            // if Ns = 1 then NOT interleaved, else interleaved: see B.2.3
            // SOSarrayPos = lRawPos;  // not required...
            if (SOSns > 0) {
                for (int lInc = 1; lInc <= SOSns; lInc++) {
                    // component identifier 1=Y,2=Cb,3=Cr,4=I,5=Q
                    btS1 = readByte(lRawRA, &lRawPos, lRawSz);
                    // dummy value used to increment file position
                    // #pragma unused(btS1)
                    // horizontal and vertical sampling factors
                    btS2 = readByte(lRawRA, &lRawPos, lRawSz);
                    // dummy value used to increment file position
                    // #pragma unused(btS2)
                }
            }
            // predictor selection B.3
            SOSss = readByte(lRawRA, &lRawPos, lRawSz);
            SOSse = readByte(lRawRA, &lRawPos, lRawSz);
            // dummy value used to increment file position
            // #pragma unused(SOSse)
            // lower 4bits= pointtransform
            SOSahal = readByte(lRawRA, &lRawPos, lRawSz);
            SOSpttrans = SOSahal & 16;
            lRawPos = lSegmentEnd;
        }
        else {
            // if SOS marker else skip marker
            lRawPos = lSegmentEnd;
        }
    } while ((lRawPos < lRawSz) && (btMarkerType != 0xDA));

    // NEXT: Huffman decoding
    if (lnHufTables < 1) {
        // Decoding error: no Huffman tables
        // TODO: use external Huffman tables?
        return JPEGSOF3_NO_TABLE;
    }

    // NEXT: unpad data - delete byte that follows $FF
    ssize_t lIncI = lRawPos;  // input position
    ssize_t lIncO = lRawPos;  // output position
    // int lIsRestartSegments = 0;
    // UNUSED(lIsRestartSegments);

    do {
        lRawRA[lIncO] = lRawRA[lIncI];  // crash when using read-only input
        if (lRawRA[lIncI] == 255) {
            if (lRawRA[lIncI + 1] == 0) {
                lIncI = lIncI + 1;
            }
            else if (lRawRA[lIncI + 1] == 0xD9) {
                // end of padding
                lIncO = -666;
            }
            // else {
            //     lIsRestartSegments = lRawRA[lIncI + 1];
            // }
        }
        lIncI++;
        lIncO++;
    } while (lIncO > 0);

    // if (lIsRestartSegments != 0)
    // detects both restart and corruption
    // https://groups.google.com/forum/#!topic/comp.protocols.dicom/JUuz0B_aE5o

    // NEXT: some RGB images use only a single Huffman table for all
    // 3 colour planes. In this case, replicate the correct values
    // NEXT: prepare lookup table
    for (int lFrameCount = 1; lFrameCount <= lnHufTables; lFrameCount++) {
        for (int lInc = 0; lInc <= 17; lInc++) {
            // Impossible value for SSSS,
            // suggests 8-bits can not describe answer
            l[lFrameCount].SSSSszRA[lInc] = 123;
        }
        for (int lInc = 0; lInc <= 255; lInc++) {
            // Impossible value for SSSS,
            // suggests 8-bits can not describe answer
            l[lFrameCount].LookUpRA[lInc] = 255;
        }
    }

    // NEXT: fill lookuptable
    for (int lFrameCount = 1; lFrameCount <= lnHufTables; lFrameCount++) {
        int lIncY = 0;
        for (int lSz = 1; lSz <= 8; lSz++) {
            // set the huffman lookup table for keys with lengths <= 8
            if (l[lFrameCount].DHTliRA[lSz] > 0) {
                for (int lIncX = 1;
                     lIncX <= l[lFrameCount].DHTliRA[lSz];
                     lIncX++)
                {
                    lIncY++;
                    int lHufVal = l[lFrameCount].HufVal[lIncY];  // SSSS
                    l[lFrameCount].SSSSszRA[lHufVal] = lSz;
                    // K= most sig bits for hufman table
                    int k = (l[lFrameCount].HufCode[lIncY] << (8 - lSz)) & 255;
                    if (lSz < 8) {
                        // fill in all possible bits that exceed
                        // the huffman table
                        int lInc = bitMask(8 - lSz);
                        for (int lCurrentBitPos = 0;
                             lCurrentBitPos <= lInc;
                             lCurrentBitPos++)
                        {
                            l[lFrameCount].LookUpRA[
                                k + lCurrentBitPos] = lHufVal;
                        }
                    }
                    else {
                        // SSSS
                        l[lFrameCount].LookUpRA[k] = lHufVal;
                    }
                }
            }
        }
    }

    // NEXT: some RGB images use only a single Huffman table
    // for all 3 colour planes. In this case, replicate the correct values
    if (lnHufTables < SOFnf) {
        // use single Hufman table for each frame
        for (int lFrameCount = lnHufTables+1;
             lFrameCount <= SOFnf;
             lFrameCount++)
        {
            l[lFrameCount] = l[lnHufTables];
        }
    }

    // NEXT: uncompress data: different loops for different predictors
    ssize_t lItems = SOFxdim * SOFydim * SOFnf;
    int lCurrentBitPos = 0;  // read in a new byte
    // depending on SOSss, we see Table H.1
    int lPredA = 0;
    int lPredB = 0;
    int lPredC = 0;
    if (SOSss == 2) {
        // predictor selection 2: above
        lPredA = SOFxdim - 1;
    }
    else if (SOSss == 3) {
        // predictor selection 3: above+left
        lPredA = SOFxdim;
    }
    else if ((SOSss == 4) || (SOSss == 5)) {
        // these use left, above and above+left WEIGHT LEFT
        lPredA = 0;  // Ra left
        lPredB = SOFxdim - 1;  // Rb directly above
        lPredC = SOFxdim;  // Rc UpperLeft:above and to the left
    }
    else if (SOSss == 6) {
        // also use left, above and above+left, WEIGHT ABOVE
        lPredB = 0;
        lPredA = SOFxdim - 1;  // Rb directly above
        lPredC = SOFxdim;  // Rc UpperLeft:above and to the left
    }
    else {
        lPredA = 0;  // Ra: directly to left
    }
    if ((SOFprecision > 8) && (SOFnf == 1)) {
        // 16 bit, 1 frame
        if (lItems * 2 < lImgSz) {
            // output array too small
            return JPEGSOF3_INVALID_OUTPUT;
        }
        ssize_t lPx = -1;  // pixel position
        int lPredicted = 1 << (SOFprecision - 1 - SOSpttrans);
        uint16_t* lImgRA16 = (uint16_t*)lImgRA8;
        for (ssize_t i = 0; i < lItems; i++) {
            // zero array
            lImgRA16[i] = 0;
        }
        for (int lIncX = 1; lIncX <= SOFxdim; lIncX++) {
            // for first row - here we ALWAYS use LEFT as predictor
            lPx++;
            if (lIncX > 1) {
                lPredicted = lImgRA16[lPx - 1];
            }
            lImgRA16[lPx] = lPredicted + decodePixelDifference(
                lRawRA, &lRawPos, &lCurrentBitPos, l[1]);
        }
        for (int lIncY = 2; lIncY <= SOFydim; lIncY++) {
            // for all subsequent rows
            lPx++;
            lPredicted = lImgRA16[lPx - SOFxdim];  // use ABOVE
            lImgRA16[lPx] = lPredicted + decodePixelDifference(
                lRawRA, &lRawPos, &lCurrentBitPos, l[1]);
            if (SOSss == 4) {
                for (int lIncX = 2; lIncX <= SOFxdim; lIncX++) {
                    lPredicted = lImgRA16[lPx - lPredA] +
                                 lImgRA16[lPx - lPredB] -
                                 lImgRA16[lPx - lPredC];
                    lPx++;
                    lImgRA16[lPx] = lPredicted + decodePixelDifference(
                        lRawRA, &lRawPos, &lCurrentBitPos, l[1]);
                }
            }
            else if ((SOSss == 5) || (SOSss == 6)) {
                for (int lIncX = 2; lIncX <= SOFxdim; lIncX++) {
                    lPredicted = lImgRA16[lPx - lPredA] +
                                 ((lImgRA16[lPx - lPredB] -
                                   lImgRA16[lPx - lPredC]) >> 1);
                    lPx++;
                    lImgRA16[lPx] = lPredicted + decodePixelDifference(
                        lRawRA, &lRawPos, &lCurrentBitPos, l[1]);
                }
            }
            else if (SOSss == 7) {
                for (int lIncX = 2; lIncX <= SOFxdim; lIncX++) {
                    lPx++;
                    lPredicted = (lImgRA16[lPx - 1] +
                                  lImgRA16[lPx - SOFxdim]) >> 1;
                    lImgRA16[lPx] = lPredicted + decodePixelDifference(
                        lRawRA, &lRawPos, &lCurrentBitPos, l[1]);
                }
            }
            else {
                // SOSss 1,2,3 read single values
                for (int lIncX = 2; lIncX <= SOFxdim; lIncX++) {
                    lPredicted = lImgRA16[lPx - lPredA];
                    lPx++;
                    lImgRA16[lPx] = lPredicted + decodePixelDifference(
                        lRawRA, &lRawPos, &lCurrentBitPos, l[1]);
                }
            }
        }
    }
    else if ((SOFprecision > 8) && (SOFnf > 1)) {
        // 16 bit, 3 frames
        if (lItems * 2 < lImgSz) {
            // output array too small
            return JPEGSOF3_INVALID_OUTPUT;
        }
        uint16_t* lImgRA16 = (uint16_t*)lImgRA8;
        ssize_t lPx[kmaxFrames + 1];
        int lPredicted[kmaxFrames + 1];  // pixel position
        for (int f = 1; f <= SOFnf; f++) {
            lPx[f] = ((f - 1) * (SOFxdim * SOFydim)) - 1;
            lPredicted[f] = 1 << (SOFprecision - 1 - SOSpttrans);
        }
        for (ssize_t i = 0; i < lItems; i++) {
            // zero array
            lImgRA16[i] = 65535;
        }
        for (int lIncX = 1; lIncX <= SOFxdim; lIncX++) {
            // for first row - here we ALWAYS use LEFT as predictor
            for (int f = 1; f <= SOFnf; f++) {
                lPx[f]++;
                if (lIncX > 1) {
                    lPredicted[f] = lImgRA16[lPx[f] - 1];
                }
                lImgRA16[lPx[f]] = lPredicted[f] + decodePixelDifference(
                    lRawRA, &lRawPos, &lCurrentBitPos, l[f]);
            }
        }
        for (int lIncY = 2; lIncY <= SOFydim; lIncY++) {
            // for all subsequent rows
            for (int f = 1; f <= SOFnf; f++) {
                lPx[f]++;
                lPredicted[f] = lImgRA16[lPx[f] - SOFxdim];  // use ABOVE
                lImgRA16[lPx[f]] = lPredicted[f] + decodePixelDifference(
                    lRawRA, &lRawPos, &lCurrentBitPos, l[f]);
            }
            if (SOSss == 4) {
                for (int lIncX = 2; lIncX <= SOFxdim; lIncX++) {
                    for (int f = 1; f <= SOFnf; f++) {
                        lPredicted[f] = lImgRA16[lPx[f] - lPredA] +
                            lImgRA16[lPx[f] - lPredB] -
                            lImgRA16[lPx[f] - lPredC];
                        lPx[f]++;
                        lImgRA16[lPx[f]] = lPredicted[f] +
                            decodePixelDifference(lRawRA, &lRawPos,
                                &lCurrentBitPos, l[f]);
                    }
                }
            }
            else if ((SOSss == 5) || (SOSss == 6)) {
                for (int lIncX = 2; lIncX <= SOFxdim; lIncX++) {
                    for (int f = 1; f <= SOFnf; f++) {
                        lPredicted[f] = lImgRA16[lPx[f] - lPredA] +
                            ((lImgRA16[lPx[f] - lPredB] -
                                lImgRA16[lPx[f] - lPredC]) >> 1);
                        lPx[f]++;
                        lImgRA16[lPx[f]] = lPredicted[f] +
                            decodePixelDifference(lRawRA, &lRawPos,
                                &lCurrentBitPos, l[f]);
                    }
                }
            }
            else if (SOSss == 7) {
                for (int lIncX = 2; lIncX <= SOFxdim; lIncX++) {
                    for (int f = 1; f <= SOFnf; f++) {
                        lPx[f]++;
                        lPredicted[f] = (lImgRA16[lPx[f] - 1] +
                            lImgRA16[lPx[f] - SOFxdim]) >> 1;
                        lImgRA16[lPx[f]] = lPredicted[f] +
                            decodePixelDifference(lRawRA, &lRawPos,
                                &lCurrentBitPos, l[f]);
                    }
                }
            }
            else {
                // SOSss 1,2,3 read single values
                for (int lIncX = 2; lIncX <= SOFxdim; lIncX++) {
                    for (int f = 1; f <= SOFnf; f++) {
                        lPredicted[f] = lImgRA16[lPx[f] - lPredA];
                        lPx[f]++;
                        lImgRA16[lPx[f]] = lPredicted[f] +
                            decodePixelDifference(lRawRA, &lRawPos,
                                &lCurrentBitPos, l[f]);
                    }
                }
            }
        }
    }
    else if ((SOFprecision <= 8) && (SOFnf > 1)) {
        // 8-bit, 3 frames
        if (lItems < lImgSz) {
            // output array too small
            return JPEGSOF3_INVALID_OUTPUT;
        }
        ssize_t lPx[kmaxFrames + 1];
        int lPredicted[kmaxFrames + 1];  // pixel position
        for (int f = 1; f <= SOFnf; f++) {
            lPx[f] = ((f - 1) * (SOFxdim * SOFydim)) - 1;
            lPredicted[f] = 1 << (SOFprecision - 1 - SOSpttrans);
        }
        for (ssize_t i = 0; i < lItems; i++) {
            // zero array
            lImgRA8[i] = 255;
        }
        for (int lIncX = 1; lIncX <= SOFxdim; lIncX++) {
            // for first row - here we ALWAYS use LEFT as predictor
            for (int f = 1; f <= SOFnf; f++) {
                lPx[f]++;
                if (lIncX > 1) {
                    lPredicted[f] = lImgRA8[lPx[f] - 1];
                }
                lImgRA8[lPx[f]] = lPredicted[f] + decodePixelDifference(
                    lRawRA, &lRawPos, &lCurrentBitPos, l[f]);
            }
        }
        for (int lIncY = 2; lIncY <= SOFydim; lIncY++) {
            // for all subsequent rows
            for (int f = 1; f <= SOFnf; f++) {
                lPx[f]++;
                lPredicted[f] = lImgRA8[lPx[f] - SOFxdim];  // use ABOVE
                lImgRA8[lPx[f]] = lPredicted[f] + decodePixelDifference(
                    lRawRA, &lRawPos, &lCurrentBitPos, l[f]);
            }
            if (SOSss == 4) {
                for (int lIncX = 2; lIncX <= SOFxdim; lIncX++) {
                    for (int f = 1; f <= SOFnf; f++) {
                        lPredicted[f] = lImgRA8[lPx[f] - lPredA] +
                                        lImgRA8[lPx[f] - lPredB] -
                                        lImgRA8[lPx[f] - lPredC];
                        lPx[f]++;
                        lImgRA8[lPx[f]] = lPredicted[f] +
                            decodePixelDifference(lRawRA, &lRawPos,
                                                  &lCurrentBitPos, l[f]);
                    }
                }
            }
            else if ((SOSss == 5) || (SOSss == 6)) {
                for (int lIncX = 2; lIncX <= SOFxdim; lIncX++) {
                    for (int f = 1; f <= SOFnf; f++) {
                        lPredicted[f] = lImgRA8[lPx[f] - lPredA] +
                                        ((lImgRA8[lPx[f] - lPredB] -
                                          lImgRA8[lPx[f] - lPredC]) >> 1);
                        lPx[f]++;
                        lImgRA8[lPx[f]] = lPredicted[f] +
                            decodePixelDifference(lRawRA, &lRawPos,
                                                  &lCurrentBitPos, l[f]);
                    }
                }
            }
            else if (SOSss == 7) {
                for (int lIncX = 2; lIncX <= SOFxdim; lIncX++) {
                    for (int f = 1; f <= SOFnf; f++) {
                        lPx[f]++;
                        lPredicted[f] = (lImgRA8[lPx[f] - 1] +
                                         lImgRA8[lPx[f] - SOFxdim]) >> 1;
                        lImgRA8[lPx[f]] = lPredicted[f] +
                            decodePixelDifference(lRawRA, &lRawPos,
                                                  &lCurrentBitPos, l[f]);
                    }
                }
            }
            else {
                // SOSss 1,2,3 read single values
                for (int lIncX = 2; lIncX <= SOFxdim; lIncX++) {
                    for (int f = 1; f <= SOFnf; f++) {
                        lPredicted[f] = lImgRA8[lPx[f] - lPredA];
                        lPx[f]++;
                        lImgRA8[lPx[f]] = lPredicted[f] +
                            decodePixelDifference(lRawRA, &lRawPos,
                                                  &lCurrentBitPos, l[f]);
                    }
                }
            }
        }
    }
    else {
        // 8-bit, 1 frame
        if (lItems < lImgSz) {
            // output array too small
            return JPEGSOF3_INVALID_OUTPUT;
        }
        ssize_t lPx = -1;  // pixel position
        int lPredicted = 1 << (SOFprecision - 1 - SOSpttrans);
        for (ssize_t i = 0; i < lItems; i++) {
            // zero array
            lImgRA8[i] = 0;
        }
        for (int lIncX = 1; lIncX <= SOFxdim; lIncX++) {
            // for first row - here we ALWAYS use LEFT as predictor
            lPx++;
            if (lIncX > 1) {
                lPredicted = lImgRA8[lPx - 1];
            }
            lImgRA8[lPx] = lPredicted + decodePixelDifference(
                lRawRA, &lRawPos, &lCurrentBitPos, l[1]);
        }
        for (int lIncY = 2; lIncY <= SOFydim; lIncY++) {
            // for all subsequent rows
            lPx++;
            lPredicted = lImgRA8[lPx - SOFxdim];  // use ABOVE
            lImgRA8[lPx] = lPredicted + decodePixelDifference(
                lRawRA, &lRawPos, &lCurrentBitPos, l[1]);
            if (SOSss == 4) {
                for (int lIncX = 2; lIncX <= SOFxdim; lIncX++) {
                    lPredicted = lImgRA8[lPx - lPredA] +
                                 lImgRA8[lPx - lPredB] -
                                 lImgRA8[lPx - lPredC];
                    lPx++;
                    lImgRA8[lPx] = lPredicted + decodePixelDifference(
                        lRawRA, &lRawPos, &lCurrentBitPos, l[1]);
                }
            }
            else if ((SOSss == 5) || (SOSss == 6)) {
                for (int lIncX = 2; lIncX <= SOFxdim; lIncX++) {
                    lPredicted = lImgRA8[lPx - lPredA] +
                                 ((lImgRA8[lPx - lPredB] -
                                   lImgRA8[lPx - lPredC]) >> 1);
                    lPx++;
                    lImgRA8[lPx] = lPredicted + decodePixelDifference(
                        lRawRA, &lRawPos, &lCurrentBitPos, l[1]);
                }
            }
            else if (SOSss == 7) {
                for (int lIncX = 2; lIncX <= SOFxdim; lIncX++) {
                    lPx++;
                    lPredicted = (lImgRA8[lPx - 1] +
                                  lImgRA8[lPx - SOFxdim]) >> 1;
                    lImgRA8[lPx] = lPredicted + decodePixelDifference(
                        lRawRA, &lRawPos, &lCurrentBitPos, l[1]);
                }
            }
            else {
                // SOSss 1,2,3 read single values
                for (int lIncX = 2; lIncX <= SOFxdim; lIncX++) {
                    lPredicted = lImgRA8[lPx - lPredA];
                    lPx++;
                    lImgRA8[lPx] = lPredicted + decodePixelDifference(
                        lRawRA, &lRawPos, &lCurrentBitPos, l[1]);
                }
            }
        }
    }
    return JPEGSOF3_OK;
}
