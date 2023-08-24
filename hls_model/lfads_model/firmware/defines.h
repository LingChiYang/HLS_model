#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 73
#define N_INPUT_2_1 70
#define N_OUT_2 128
#define N_OUT_2 128
#define N_LAYER_4 64
#define N_INPUT_1_6 73
#define N_INPUT_2_6 64
#define N_LAYER_4 64
#define N_OUT_8 64
#define N_OUT_8 64

// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<32,16> input_t;
typedef ap_fixed<31,9> accum2_t;
typedef ap_fixed<8,1> forward_weight2_t;
typedef ap_fixed<8,1> backward_weight2_t;
typedef ap_fixed<8,1> forward_recurrent_weight2_t;
typedef ap_fixed<8,1> backward_recurrent_weight2_t;
typedef ap_fixed<8,1> forward_bias2_t;
typedef ap_fixed<8,1> forward_recurrent_bias2_t;
typedef ap_fixed<8,1> backward_bias2_t;
typedef ap_fixed<8,1> backward_recurrent_bias2_t;
typedef ap_fixed<8,1,AP_RND_CONV,AP_SAT> act2_t;
typedef ap_ufixed<8,0,AP_RND_CONV,AP_SAT> recr_act2_t;
typedef ap_fixed<8,1,AP_RND_CONV,AP_SAT> state2_t;
typedef ap_ufixed<2,0> slope2_t;
typedef ap_ufixed<2,0> shift2_t;
typedef ap_fixed<17,2> layer2_t;
typedef ap_fixed<23,9> accum_dense2_t;
typedef ap_fixed<8,1,AP_RND_CONV,AP_SAT> layer3_t;
typedef ap_fixed<18,8> active_bits0_table_t;
typedef ap_fixed<32,16> model_default_t;
typedef ap_fixed<32,16> layer4_t;
typedef ap_fixed<8,1> weight4_t;
typedef ap_fixed<8,1> bias4_t;
typedef ap_uint<1> layer4_index;
typedef ap_fixed<32,16> input6_t;
typedef ap_fixed<8,1,AP_RND_CONV,AP_SAT> layer7_t;
typedef ap_fixed<18,8> active_bits1_table_t;
typedef ap_fixed<30,8> accum8_t;
typedef ap_fixed<8,1> weight8_t;
typedef ap_fixed<8,1> recurrent_weight8_t;
typedef ap_fixed<8,1> bias8_t;
typedef ap_fixed<8,1> recurrent_bias8_t;
typedef ap_fixed<8,1,AP_RND_CONV,AP_SAT> act8_t;
typedef ap_ufixed<8,0,AP_RND_CONV,AP_SAT> recr_act8_t;
typedef ap_fixed<8,1,AP_RND_CONV,AP_SAT> state8_t;
typedef ap_ufixed<2,0> slope8_t;
typedef ap_ufixed<2,0> shift8_t;
typedef ap_fixed<17,2> layer8_t;
typedef ap_fixed<22,8> accum_dense8_t;
typedef ap_fixed<8,1,AP_RND_CONV,AP_SAT> result_t;
typedef ap_fixed<18,8> active_bits2_table_t;

#endif
