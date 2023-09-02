#ifndef MYPROJECT_H_
#define MYPROJECT_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"

// Prototype of top level function for C-synthesis
void myproject(
    hls::stream<input10_t> input_20[1],
    hls::stream<input10_t> input_21[1],
    hls::stream<layer11_t> layer11_out[1],
    hls::stream<layer12_t> layer12_out[1],
    weight12_t w120[786432/2],
    weight12_t w121[786432/2],
    recurrent_weight12_t wr120[786432/2],
    recurrent_weight12_t wr121[786432/2]
);

#endif
