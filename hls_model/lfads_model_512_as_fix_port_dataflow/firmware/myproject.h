#ifndef MYPROJECT_H_
#define MYPROJECT_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"

// Prototype of top level function for C-synthesis
void myproject(
    hls::stream<input_t> input_1[49], hls::stream<input10_t> input_2[512],
    hls::stream<result_t> layer16_out[49]
);

#endif
