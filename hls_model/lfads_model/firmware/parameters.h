#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_fixed.h"
#include "ap_int.h"

#include "nnet_utils/nnet_code_gen.h"
#include "nnet_utils/nnet_helpers.h"
// hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_array_stream.h"
#include "nnet_utils/nnet_activation_stream.h"
#include "nnet_utils/nnet_bidirectional.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_array_stream.h"
#include "nnet_utils/nnet_dense_compressed.h"
#include "nnet_utils/nnet_dense_stream.h"
#include "nnet_utils/nnet_recurrent.h"
#include "nnet_utils/nnet_recurrent_array_stream.h"

// hls-fpga-machine-learning insert weights
#include "weights/fw2.h"
#include "weights/bw2.h"
#include "weights/fwr2.h"
#include "weights/bwr2.h"
#include "weights/fb2.h"
#include "weights/fbr2.h"
#include "weights/bb2.h"
#include "weights/bbr2.h"
#include "weights/w4.h"
#include "weights/b4.h"
#include "weights/w8.h"
#include "weights/wr8.h"
#include "weights/b8.h"
#include "weights/br8.h"

// hls-fpga-machine-learning insert layer-config
// Encoder_BidirectionalGRU
struct config2_1 : nnet::dense_config {
    static const unsigned n_in = N_INPUT_2_1;
    static const unsigned n_out = N_OUT_2 * 3 /2;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 319;
    static const unsigned n_nonzeros = 13121;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef accum_dense2_t accum_t;
    typedef forward_bias2_t bias_t;
    typedef forward_weight2_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config2_2 : nnet::dense_config {
    static const unsigned n_in = N_OUT_2/2;
    static const unsigned n_out = N_OUT_2 * 3 /2;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 549;
    static const unsigned n_nonzeros = 11739;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef accum_dense2_t accum_t;
    typedef forward_bias2_t bias_t;
    typedef forward_weight2_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct hard_sigmoid_config2_recr  : nnet::hard_activ_config {
    static const unsigned n_in = N_OUT_2 * 2 /2;
    static const slope2_t slope;
    static const shift2_t shift;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
};
const slope2_t hard_sigmoid_config2_recr::slope = 0.5;
const shift2_t hard_sigmoid_config2_recr::shift = 0.5;

struct hard_tanh_config2 : nnet::hard_activ_config{
    static const unsigned n_in = N_OUT_2/2;
    static const slope2_t slope;
    static const shift2_t shift;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
};
const slope2_t hard_tanh_config2::slope = 0.5;
const shift2_t hard_tanh_config2::shift = 0.5;

struct config2_f : nnet::gru_config {
    typedef accum_dense2_t accum_dense_t;
    typedef accum2_t accum_t;
    typedef forward_weight2_t weight_t;  // Matrix
    typedef forward_bias2_t bias_t;  // Vector
    typedef config2_1 mult_config1;
    typedef config2_2 mult_config2;
    typedef hard_sigmoid_config2_recr ACT_CONFIG_GRU;
    template<class x_T, class y_T, class config_T>
    using activation_recr = nnet::activation::hard_sigmoid<x_T, y_T, config_T>;
    typedef hard_tanh_config2 ACT_CONFIG_T;
    template<class x_T, class y_T, class config_T>
    using activation = nnet::activation::hard_tanh<x_T, y_T, config_T>;
    static const unsigned n_in  = N_INPUT_2_1;
    static const unsigned n_out = 64;
    static const unsigned n_state = 64;
    static const unsigned n_sequence = N_INPUT_1_1;
    static const unsigned n_sequence_out = 1;
    static const unsigned io_type = nnet::resource;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    static const bool use_static = true;
    typedef state2_t state_t;
    typedef act2_t act_t;
    typedef recr_act2_t recr_act_t;
    static const unsigned merge_mode = nnet::concat;
};

struct config2_b : nnet::gru_config {
    typedef accum_dense2_t accum_dense_t;
    typedef accum2_t accum_t;
    typedef backward_weight2_t weight_t;  // Matrix
    typedef backward_bias2_t bias_t;  // Vector
    typedef config2_1 mult_config1;
    typedef config2_2 mult_config2;
    typedef hard_sigmoid_config2_recr ACT_CONFIG_GRU;
    template<class x_T, class y_T, class config_T>
    using activation_recr = nnet::activation::hard_sigmoid<x_T, y_T, config_T>;
    typedef hard_tanh_config2 ACT_CONFIG_T;
    template<class x_T, class y_T, class config_T>
    using activation = nnet::activation::hard_tanh<x_T, y_T, config_T>;
    static const unsigned n_in  = N_INPUT_2_1;
    static const unsigned n_out = 64;
    static const unsigned n_state = 64;
    static const unsigned n_sequence = N_INPUT_1_1;
    static const unsigned n_sequence_out = 1;
    static const unsigned io_type = nnet::resource;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    static const bool use_static = true;
    typedef state2_t state_t;
    typedef act2_t act_t;
    typedef recr_act2_t recr_act_t;
    static const unsigned merge_mode = nnet::concat;
};

struct config2 : nnet::bidirectional_config {
    typedef accum_dense2_t accum_dense_t;
    typedef accum2_t accum_t;
    typedef backward_weight2_t weight_t;  // Matrix
    typedef backward_bias2_t bias_t;  // Vector

    typedef config2_f config_rnn_layer_f;
    typedef config2_b config_rnn_layer_b;
    static const unsigned n_in  = N_INPUT_2_1;
    static const unsigned n_out = N_OUT_2;
    static const unsigned n_state = N_OUT_2/2;
    static const unsigned n_sequence = N_INPUT_1_1;
    static const unsigned n_sequence_out = 1;
    static const unsigned io_type = nnet::resource;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    static const unsigned merge_mode = nnet::concat;
};

// active_bits0
struct linear_config3 : nnet::activ_config {
    static const unsigned n_in = 128;
    static const unsigned n_chan = N_OUT_2;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef active_bits0_table_t table_t;
};

// dense_latent
struct config4 : nnet::dense_config {
    static const unsigned n_in = 128;
    static const unsigned n_out = 64;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 183;
    static const unsigned n_nonzeros = 8009;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef bias4_t bias_t;
    typedef weight4_t weight_t;
    typedef layer4_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// active_bits1
struct linear_config7 : nnet::activ_config {
    static const unsigned n_in = 64;
    static const unsigned n_chan = N_LAYER_4;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef active_bits1_table_t table_t;
};

// DecoderGRU
struct config8_1 : nnet::dense_config {
    static const unsigned n_in = N_INPUT_2_6;
    static const unsigned n_out = N_OUT_8 * 3;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 299;
    static const unsigned n_nonzeros = 11989;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef accum_dense8_t accum_t;
    typedef bias8_t bias_t;
    typedef weight8_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config8_2 : nnet::dense_config {
    static const unsigned n_in = N_OUT_8;
    static const unsigned n_out = N_OUT_8 * 3;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 536;
    static const unsigned n_nonzeros = 11752;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef accum_dense8_t accum_t;
    typedef bias8_t bias_t;
    typedef weight8_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct hard_sigmoid_config8_recr  : nnet::hard_activ_config {
    static const unsigned n_in = N_OUT_8 * 2;
    static const slope8_t slope;
    static const shift8_t shift;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
};
const slope8_t hard_sigmoid_config8_recr::slope = 0.5;
const shift8_t hard_sigmoid_config8_recr::shift = 0.5;

struct hard_tanh_config8 : nnet::hard_activ_config{
    static const unsigned n_in = N_OUT_8;
    static const slope8_t slope;
    static const shift8_t shift;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
};
const slope8_t hard_tanh_config8::slope = 0.5;
const shift8_t hard_tanh_config8::shift = 0.5;

struct config8 : nnet::gru_config {
    typedef accum_dense8_t accum_dense_t;
    typedef accum8_t accum_t;
    typedef weight8_t weight_t;  // Matrix
    typedef bias8_t bias_t;  // Vector
    typedef config8_1 mult_config1;
    typedef config8_2 mult_config2;
    typedef hard_sigmoid_config8_recr ACT_CONFIG_GRU;
    template<class x_T, class y_T, class config_T>
    using activation_recr = nnet::activation::hard_sigmoid<x_T, y_T, config_T>;
    typedef hard_tanh_config8 ACT_CONFIG_T;
    template<class x_T, class y_T, class config_T>
    using activation = nnet::activation::hard_tanh<x_T, y_T, config_T>;
    static const unsigned n_in  = N_INPUT_2_6;
    static const unsigned n_out = N_OUT_8;
    static const unsigned n_state = N_OUT_8;
    static const unsigned n_sequence = N_INPUT_1_6;
    static const unsigned n_sequence_out = 1;
    static const unsigned io_type = nnet::resource;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    static const bool use_static = true;
    static const bool use_initial = 1;
    typedef state8_t state_t;
    typedef act8_t act_t;
    typedef recr_act8_t recr_act_t;
};

// active_bits2
struct linear_config9 : nnet::activ_config {
    static const unsigned n_in = 64;
    static const unsigned n_chan = N_OUT_8;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef active_bits2_table_t table_t;
};


#endif
