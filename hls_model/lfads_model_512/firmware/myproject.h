#ifndef MYPROJECT_H_
#define MYPROJECT_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"

// Prototype of top level function for C-synthesis
void myproject(
    input_t input_1[N_INPUT_1_1*N_INPUT_2_1], input10_t input_2[N_INPUT_1_10*N_INPUT_2_10],
    result_t layer16_out[N_LAYER_1_14*N_LAYER_2_14]
);

#endif
