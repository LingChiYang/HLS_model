#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    input_t input_1[N_INPUT_1_1*N_INPUT_2_1], input6_t input_2[N_INPUT_1_6*N_INPUT_2_6],
    result_t layer9_out[N_OUT_8]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=input_1 complete dim=0
    #pragma HLS ARRAY_RESHAPE variable=input_2 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=input_1,input_2,layer9_out 
    #pragma HLS DATAFLOW 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<forward_weight2_t, 13440>(fw2, "fw2.txt");
        nnet::load_weights_from_txt<backward_weight2_t, 13440>(bw2, "bw2.txt");
        nnet::load_weights_from_txt<forward_recurrent_weight2_t, 12288>(fwr2, "fwr2.txt");
        nnet::load_weights_from_txt<backward_recurrent_weight2_t, 12288>(bwr2, "bwr2.txt");
        nnet::load_weights_from_txt<forward_bias2_t, 192>(fb2, "fb2.txt");
        nnet::load_weights_from_txt<forward_recurrent_bias2_t, 192>(fbr2, "fbr2.txt");
        nnet::load_weights_from_txt<backward_bias2_t, 192>(bb2, "bb2.txt");
        nnet::load_weights_from_txt<backward_recurrent_bias2_t, 192>(bbr2, "bbr2.txt");
        nnet::load_weights_from_txt<weight4_t, 8192>(w4, "w4.txt");
        nnet::load_weights_from_txt<bias4_t, 64>(b4, "b4.txt");
        nnet::load_weights_from_txt<weight8_t, 12288>(w8, "w8.txt");
        nnet::load_weights_from_txt<recurrent_weight8_t, 12288>(wr8, "wr8.txt");
        nnet::load_weights_from_txt<bias8_t, 192>(b8, "b8.txt");
        nnet::load_weights_from_txt<recurrent_bias8_t, 192>(br8, "br8.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    layer2_t layer2_out[N_OUT_2];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    nnet::bidirectional<input_t, layer2_t, config2>(input_1, layer2_out, bw2, bwr2, bb2, bbr2, fw2, fwr2, fb2, fbr2); // Encoder_BidirectionalGRU
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer2_t>(layer2_out, "Encoder_BidirectionalGRU", N_OUT_2);
#endif

    layer3_t layer3_out[N_OUT_2];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0
    nnet::linear<layer2_t, layer3_t, linear_config3>(layer2_out, layer3_out); // active_bits0
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer3_t>(layer3_out, "active_bits0", N_OUT_2);
#endif

    layer4_t layer4_out[N_LAYER_4];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
    nnet::dense<layer3_t, layer4_t, config4>(layer3_out, layer4_out, w4, b4); // dense_latent
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer4_t>(layer4_out, "dense_latent", N_LAYER_4);
#endif

    layer7_t layer7_out[N_LAYER_4];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
    nnet::linear<layer4_t, layer7_t, linear_config7>(layer4_out, layer7_out); // active_bits1
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer7_t>(layer7_out, "active_bits1", N_LAYER_4);
#endif

    layer8_t layer8_out[N_OUT_8];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0
    nnet::gru_stack<input6_t, layer7_t, layer8_t, config8>(input_2, layer7_out, layer8_out, w8, wr8, b8, br8); // DecoderGRU
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer8_t>(layer8_out, "DecoderGRU", N_OUT_8);
#endif

    nnet::linear<layer8_t, result_t, linear_config9>(layer8_out, layer9_out); // active_bits2
#ifndef __SYNTHESIS__
    nnet::save_layer_output<result_t>(layer9_out, "active_bits2", N_OUT_8);
#endif

}
