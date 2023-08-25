#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    hls::stream<input_t> input_1[49], hls::stream<input10_t> input_2[512],
    hls::stream<result_t> layer16_out[49]
) {

    // hls-fpga-machine-learning insert IO

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<forward_weight2_t, 75264>(fw2, "fw2.txt");
        nnet::load_weights_from_txt<backward_weight2_t, 75264>(bw2, "bw2.txt");
        nnet::load_weights_from_txt<forward_recurrent_weight2_t, 786432>(fwr2, "fwr2.txt");
        nnet::load_weights_from_txt<backward_recurrent_weight2_t, 786432>(bwr2, "bwr2.txt");
        nnet::load_weights_from_txt<forward_bias2_t, 1536>(fb2, "fb2.txt");
        nnet::load_weights_from_txt<forward_recurrent_bias2_t, 1536>(fbr2, "fbr2.txt");
        nnet::load_weights_from_txt<backward_bias2_t, 1536>(bb2, "bb2.txt");
        nnet::load_weights_from_txt<backward_recurrent_bias2_t, 1536>(bbr2, "bbr2.txt");
        nnet::load_weights_from_txt<weight4_t, 524288>(w4, "w4.txt");
        nnet::load_weights_from_txt<bias4_t, 512>(b4, "b4.txt");
        nnet::load_weights_from_txt<weight6_t, 524288>(w6, "w6.txt");
        nnet::load_weights_from_txt<bias6_t, 512>(b6, "b6.txt");
        nnet::load_weights_from_txt<weight12_t, 786432>(w12, "w12.txt");
        nnet::load_weights_from_txt<recurrent_weight12_t, 786432>(wr12, "wr12.txt");
        nnet::load_weights_from_txt<bias12_t, 1536>(b12, "b12.txt");
        nnet::load_weights_from_txt<recurrent_bias12_t, 1536>(br12, "br12.txt");
        nnet::load_weights_from_txt<nerual_dense_weight_t, 25088>(w17, "w17.txt");
        nnet::load_weights_from_txt<bias17_t, 49>(b17, "b17.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    hls::stream<layer2_t> layer2_out[1024];
    #pragma HLS STREAM variable=layer2_out depth=1
    nnet::bidirectional<input_t, layer2_t, config2>(input_1, layer2_out, bw2, bwr2, bb2, bbr2, fw2, fwr2, fb2, fbr2); // Encoder_BidirectionalGRU
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer2_t,1024>(layer2_out, "Encoder_BidirectionalGRU", N_OUT_2);
#endif

    hls::stream<layer3_t> layer3_out[1024];
    #pragma HLS STREAM variable=layer3_out depth=1
    nnet::linear<layer2_t, layer3_t, linear_config3>(layer2_out, layer3_out); // active_bits0
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer3_t,1024>(layer3_out, "active_bits0", N_OUT_2);
#endif

    hls::stream<layer18_t> layer18_cpy1[1024];
    #pragma HLS STREAM variable=layer18_cpy1 depth=1
    hls::stream<layer18_t> layer18_cpy2[1024];
    #pragma HLS STREAM variable=layer18_cpy2 depth=1
    nnet::clone_stream<layer3_t, layer18_t, 1024, 1024>(layer3_out, layer18_cpy1, layer18_cpy2); // clone_active_bits0

    hls::stream<layer4_t> layer4_out[512];
    #pragma HLS STREAM variable=layer4_out depth=1
    nnet::dense<layer18_t, layer4_t, config4>(layer18_cpy1, layer4_out, w4, b4); // dense_mean
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer4_t,512>(layer4_out, "dense_mean", N_LAYER_4);
#endif

    hls::stream<layer6_t> layer6_out[512];
    #pragma HLS STREAM variable=layer6_out depth=1
    nnet::dense<layer18_t, layer6_t, config6>(layer18_cpy2, layer6_out, w6, b6); // dense_latent2
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer6_t,512>(layer6_out, "dense_latent2", N_LAYER_6);
#endif

    hls::stream<layer8_t> layer8_out[512];
    #pragma HLS STREAM variable=layer8_out depth=1
    nnet::linear<layer4_t, layer8_t, linear_config8>(layer4_out, layer8_out); // active_bits1
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer8_t,512>(layer8_out, "active_bits1", N_LAYER_4);
#endif

    hls::stream<layer9_t> layer9_out[512];
    #pragma HLS STREAM variable=layer9_out depth=1
    nnet::linear<layer6_t, layer9_t, linear_config9>(layer6_out, layer9_out); // active_bits
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer9_t,512>(layer9_out, "active_bits", N_LAYER_6);
#endif

    hls::stream<layer11_t> layer11_out[512];
    #pragma HLS STREAM variable=layer11_out depth=1
    nnet::add<layer8_t, layer9_t, layer11_t, config11>(layer8_out, layer9_out, layer11_out); // add
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer11_t,512>(layer11_out, "add", N_LAYER_4);
#endif

    hls::stream<layer12_t> layer12_out[512];
    #pragma HLS STREAM variable=layer12_out depth=73
    nnet::gru_stack<input10_t, layer11_t, layer12_t, config12>(input_2, layer11_out, layer12_out, w12, wr12, b12, br12); // DecoderGRU
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer12_t,512>(layer12_out, "DecoderGRU", N_TIME_STEPS_12*N_OUT_12);
#endif

    hls::stream<layer13_t> layer13_out[512];
    #pragma HLS STREAM variable=layer13_out depth=73
    nnet::linear<layer12_t, layer13_t, linear_config13>(layer12_out, layer13_out); // active_bits2
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer13_t,512>(layer13_out, "active_bits2", N_TIME_STEPS_12*N_OUT_12);
#endif

    hls::stream<layer17_t> layer17_out[49];
    #pragma HLS STREAM variable=layer17_out depth=73
    nnet::pointwise_conv_1d_cl<layer13_t, layer17_t, config17>(layer13_out, layer17_out, w17, b17); // nerual_dense
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer17_t,49>(layer17_out, "nerual_dense", N_OUTPUTS_17*N_FILT_17);
#endif

    nnet::linear<layer17_t, result_t, linear_config16>(layer17_out, layer16_out); // active_bits4
#ifndef __SYNTHESIS__
    nnet::save_layer_output<result_t,49>(layer16_out, "active_bits4", N_LAYER_1_14*N_LAYER_2_14);
#endif

}
