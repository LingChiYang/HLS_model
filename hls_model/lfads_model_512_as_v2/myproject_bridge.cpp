#ifndef MYPROJECT_BRIDGE_H_
#define MYPROJECT_BRIDGE_H_

#include "firmware/myproject.h"
#include "firmware/nnet_utils/nnet_helpers.h"
#include <algorithm>
#include <map>

// hls-fpga-machine-learning insert bram

namespace nnet {
bool trace_enabled = false;
std::map<std::string, void *> *trace_outputs = NULL;
size_t trace_type_size = sizeof(double);
} // namespace nnet

extern "C" {

struct trace_data {
    const char *name;
    void *data;
};

void allocate_trace_storage(size_t element_size) {
    nnet::trace_enabled = true;
    nnet::trace_outputs = new std::map<std::string, void *>;
    nnet::trace_type_size = element_size;
    nnet::trace_outputs->insert(std::pair<std::string, void *>("Encoder_BidirectionalGRU", (void *) malloc(N_OUT_2 * element_size)));
    nnet::trace_outputs->insert(std::pair<std::string, void *>("active_bits0", (void *) malloc(N_OUT_2 * element_size)));
    nnet::trace_outputs->insert(std::pair<std::string, void *>("dense_mean", (void *) malloc(N_LAYER_4 * element_size)));
    nnet::trace_outputs->insert(std::pair<std::string, void *>("dense_latent2", (void *) malloc(N_LAYER_6 * element_size)));
    nnet::trace_outputs->insert(std::pair<std::string, void *>("active_bits1", (void *) malloc(N_LAYER_4 * element_size)));
    nnet::trace_outputs->insert(std::pair<std::string, void *>("active_bits", (void *) malloc(N_LAYER_6 * element_size)));
    nnet::trace_outputs->insert(std::pair<std::string, void *>("add", (void *) malloc(N_LAYER_4 * element_size)));
    nnet::trace_outputs->insert(std::pair<std::string, void *>("DecoderGRU", (void *) malloc(N_TIME_STEPS_12*N_OUT_12 * element_size)));
    nnet::trace_outputs->insert(std::pair<std::string, void *>("active_bits2", (void *) malloc(N_TIME_STEPS_12*N_OUT_12 * element_size)));
    nnet::trace_outputs->insert(std::pair<std::string, void *>("nerual_dense", (void *) malloc(N_OUTPUTS_17*N_FILT_17 * element_size)));
    nnet::trace_outputs->insert(std::pair<std::string, void *>("active_bits4", (void *) malloc(N_LAYER_1_14*N_LAYER_2_14 * element_size)));
}

void free_trace_storage() {
    for (std::map<std::string, void *>::iterator i = nnet::trace_outputs->begin(); i != nnet::trace_outputs->end(); i++) {
        void *ptr = i->second;
        free(ptr);
    }
    nnet::trace_outputs->clear();
    delete nnet::trace_outputs;
    nnet::trace_outputs = NULL;
    nnet::trace_enabled = false;
}

void collect_trace_output(struct trace_data *c_trace_outputs) {
    int ii = 0;
    for (std::map<std::string, void *>::iterator i = nnet::trace_outputs->begin(); i != nnet::trace_outputs->end(); i++) {
        c_trace_outputs[ii].name = i->first.c_str();
        c_trace_outputs[ii].data = i->second;
        ii++;
    }
}

// Wrapper of top level function for Python bridge
void myproject_float(
    float input_1[N_INPUT_1_1*N_INPUT_2_1], float input_2[N_INPUT_1_10*N_INPUT_2_10],
    float layer16_out[N_LAYER_1_14*N_LAYER_2_14]
) {

    hls::stream<input_t> input_1_ap[49];
    nnet::convert_data<float, input_t, 49, N_INPUT_1_1*N_INPUT_2_1/49>(input_1, input_1_ap);
    hls::stream<input10_t> input_2_ap[512];
    nnet::convert_data<float, input10_t, 512, N_INPUT_1_10*N_INPUT_2_10/512>(input_2, input_2_ap);

    hls::stream<result_t> layer16_out_ap[49];

    myproject(input_1_ap,input_2_ap,layer16_out_ap);

    nnet::convert_data<result_t, float, 49, N_LAYER_1_14*N_LAYER_2_14/49>(layer16_out_ap, layer16_out);
}

void myproject_double(
    double input_1[N_INPUT_1_1*N_INPUT_2_1], double input_2[N_INPUT_1_10*N_INPUT_2_10],
    double layer16_out[N_LAYER_1_14*N_LAYER_2_14]
) {
    hls::stream<input_t> input_1_ap[49];
    nnet::convert_data<double, input_t, 49, N_INPUT_1_1*N_INPUT_2_1/49>(input_1, input_1_ap);
    hls::stream<input10_t> input_2_ap[512];
    nnet::convert_data<double, input10_t, 512, N_INPUT_1_10*N_INPUT_2_10/512>(input_2, input_2_ap);

    hls::stream<result_t> layer16_out_ap[49];

    myproject(input_1_ap,input_2_ap,layer16_out_ap);

    nnet::convert_data<result_t, double, 49, N_LAYER_1_14*N_LAYER_2_14/49>(layer16_out_ap, layer16_out);
}
}

#endif
