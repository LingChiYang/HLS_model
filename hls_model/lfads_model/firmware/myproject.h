#ifndef MYPROJECT_H_
#define MYPROJECT_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"

// Prototype of top level function for C-synthesis
void myproject(
    input_t input_1[N_INPUT_1_1*N_INPUT_2_1], input6_t input_2[N_INPUT_1_6*N_INPUT_2_6],
    result_t layer9_out[N_OUT_8]
);

#endif
