#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    hls::stream<input10_t> input_20[1],
    hls::stream<input10_t> input_21[1],
    hls::stream<layer11_t> layer11_out[1],
    hls::stream<layer12_t> layer12_out[1],
    weight12_t w120[786432/2],
    weight12_t w121[786432/2],
    recurrent_weight12_t wr120[786432/2],
    recurrent_weight12_t wr121[786432/2]
) {

    // hls-fpga-machine-learning insert IO

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        //nnet::load_weights_from_txt<weight12_t, 786432>(w12, "w12.txt");
        //nnet::load_weights_from_txt<recurrent_weight12_t, 786432>(wr12, "wr12.txt");
        nnet::load_weights_from_txt<bias12_t, 1536>(b12, "b12.txt");
        nnet::load_weights_from_txt<recurrent_bias12_t, 1536>(br12, "br12.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers
std::cout<<"8"<<std::endl;
    nnet::gru_stack_single<input10_t, layer11_t, layer12_t, config12>(input_20, input_21, layer11_out, layer12_out, w120, w121, wr120, wr121, b12, br12); // DecoderGRU
}
