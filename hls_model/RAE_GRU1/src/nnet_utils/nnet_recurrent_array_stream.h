#ifndef NNET_RECURSIVE_ARRAY_STREAM_H_
#define NNET_RECURSIVE_ARRAY_STREAM_H_

#include "hls_stream.h"
#include "nnet_activation.h"
#include "nnet_common.h"
#include "nnet_dense.h"
#include "nnet_recr_activations.h"
#include "nnet_recurrent.h"

namespace nnet {

template<class data_T, class init_T, class res_T, typename CONFIG_T>
  void gru_stack_single(
      hls::stream<data_T> data_stream0[1],
      hls::stream<data_T> data_stream1[1],
      hls::stream<init_T> initial_state[1],
      hls::stream<res_T>  res_stream[1],
      typename CONFIG_T::weight_t     param0   [CONFIG_T::n_state*3*CONFIG_T::n_in/2],
      typename CONFIG_T::weight_t     param1   [CONFIG_T::n_state*3*CONFIG_T::n_in/2],
      typename CONFIG_T::weight_t     param_zr0[CONFIG_T::n_state*3*CONFIG_T::n_state/2],
      typename CONFIG_T::weight_t     param_zr1[CONFIG_T::n_state*3*CONFIG_T::n_state/2],
      typename CONFIG_T::bias_t       param_b [CONFIG_T::n_state*3],
      typename CONFIG_T::bias_t       param_br [CONFIG_T::n_state*3]
      ) {

    res_T  h_newstate[CONFIG_T::n_state];
    //#pragma HLS ARRAY_PARTITION variable=h_newstate complete
    
    if (CONFIG_T::use_initial==1){
        for(int ii = 0; ii < CONFIG_T::n_state; ii++) {
            #pragma HLS PIPELINE II=1
            h_newstate[ii] = initial_state[0].read();
        }
    }else{
        for(int ii = 0; ii < CONFIG_T::n_state; ii++) {
            #pragma HLS PIPELINE II=1
            h_newstate[ii] = 0;
        }
    }

    data_T data_in0[CONFIG_T::n_in/2];
    data_T data_in1[CONFIG_T::n_in/2];
    #pragma HLS ARRAY_RESHAPE variable=data_in0 complete
    #pragma HLS ARRAY_RESHAPE variable=data_in1 complete
    bool reset_state = false;
    DataPropagation: for(int i_in = 0; i_in < CONFIG_T::n_sequence; i_in++) {
      if (CONFIG_T::n_sequence*CONFIG_T::n_in / CONFIG_T::n_in > 1) {
          // #pragma HLS PIPELINE
      }
      DataPack1: for (int i_pack = 0; i_pack < CONFIG_T::n_in/2; i_pack++) {
          #pragma HLS PIPELINE
          data_in0[i_pack] = data_stream0[0].read();
      }
      DataPack0: for (int i_pack = 0; i_pack < CONFIG_T::n_in/2; i_pack++) {
          #pragma HLS PIPELINE
          data_in1[i_pack] = data_stream1[0].read();
      }
      if (CONFIG_T::use_static)
        nnet::gru_static<data_T, res_T, CONFIG_T>(reset_state,data_in0,data_in1,h_newstate,param0,param1,param_zr0,param_zr1,param_b, param_br);
      else
        nnet::gru<data_T, res_T, CONFIG_T>(reset_state,data_in0,h_newstate,param0,param_zr0,param_b, param_br);
      if (CONFIG_T::n_sequence_out > 1){
        ResPack_sequences: for (int i_pack = 0; i_pack < CONFIG_T::n_out; i_pack++) {
            #pragma HLS PIPELINE
            res_stream[0].write(h_newstate[i_pack]);
        }
      }
      reset_state = false;
    }

    if (CONFIG_T::n_sequence_out == 1){
        ResPack: for (int i_pack = 0; i_pack < CONFIG_T::n_out; i_pack++) {
            #pragma HLS PIPELINE
            res_stream[0].write(h_newstate[i_pack]);
        }
    }

}


template<class data_T, class init_T, class res_T, typename CONFIG_T>
  void gru_stack_array(
      hls::stream<data_T> data_stream[CONFIG_T::n_in],
      hls::stream<init_T> initial_state[CONFIG_T::n_state],
      hls::stream<res_T>  res_stream[CONFIG_T::n_out],
      typename CONFIG_T::weight_t     param   [CONFIG_T::n_state*3*CONFIG_T::n_in],
      typename CONFIG_T::weight_t     param_zr[CONFIG_T::n_state*3*CONFIG_T::n_state],
      typename CONFIG_T::bias_t       param_b [CONFIG_T::n_state*3],
      typename CONFIG_T::bias_t       param_br [CONFIG_T::n_state*3]
      ) {

    res_T  h_newstate[CONFIG_T::n_state];
    #pragma HLS ARRAY_PARTITION variable=h_newstate complete
    
    if (CONFIG_T::use_initial==1){
        for(int ii = 0; ii < CONFIG_T::n_state; ii++) {
            #pragma HLS UNROLL
            h_newstate[ii] = res_T(initial_state[ii].read());
        }
    }else{
        for(int ii = 0; ii < CONFIG_T::n_state; ii++) {
            #pragma HLS UNROLL
            h_newstate[ii] = 0;
        }
    }

    data_T data_in[CONFIG_T::n_in];
    #pragma HLS ARRAY_RESHAPE variable=data_in complete
    bool reset_state = false;

    DataPropagation: for(int i_in = 0; i_in < CONFIG_T::n_sequence; i_in++) {
      if (CONFIG_T::n_sequence*CONFIG_T::n_in / CONFIG_T::n_in > 1) {
          // #pragma HLS PIPELINE
      }
      DataPack: for (int i_pack = 0; i_pack < CONFIG_T::n_in; i_pack++) {
          #pragma HLS UNROLL
          data_in[i_pack] = data_stream[i_pack].read();
      }
      if (CONFIG_T::use_static)
        nnet::gru_static<data_T, res_T, CONFIG_T>(reset_state,data_in,h_newstate,param,param_zr,param_b, param_br);
      else
        nnet::gru<data_T, res_T, CONFIG_T>(reset_state,data_in,h_newstate,param,param_zr,param_b, param_br);
      if (CONFIG_T::n_sequence_out > 1){
        ResPack_sequences: for (int i_pack = 0; i_pack < CONFIG_T::n_out; i_pack++) {
            #pragma HLS UNROLL
            res_stream[i_pack].write(h_newstate[i_pack]);
        }
      }
      reset_state = false;
    }

    if (CONFIG_T::n_sequence_out == 1){
        ResPack: for (int i_pack = 0; i_pack < CONFIG_T::n_out; i_pack++) {
            #pragma HLS UNROLL
            res_stream[i_pack].write(h_newstate[i_pack]);
        }
    }

}

template<class data_T, class res_T, typename CONFIG_T>
  void gru_stack(
      hls::stream<data_T> data_stream[CONFIG_T::n_in],
      hls::stream<res_T>  res_stream[CONFIG_T::n_out],
      typename CONFIG_T::weight_t     param   [CONFIG_T::n_state*3*CONFIG_T::n_in],
      typename CONFIG_T::weight_t     param_zr[CONFIG_T::n_state*3*CONFIG_T::n_state],
      typename CONFIG_T::bias_t       param_b [CONFIG_T::n_state*3],
      typename CONFIG_T::bias_t       param_br [CONFIG_T::n_state*3]
      ) {

    res_T  h_newstate[CONFIG_T::n_state];
    #pragma HLS ARRAY_PARTITION variable=h_newstate complete
    
    for(int ii = 0; ii < CONFIG_T::n_state; ii++) {
        #pragma HLS UNROLL
        h_newstate[ii] = 0;
      }

    data_T data_in[CONFIG_T::n_in];
    #pragma HLS ARRAY_RESHAPE variable=data_in complete
    bool reset_state = false;

    DataPropagation: for(int i_in = 0; i_in < CONFIG_T::n_sequence; i_in++) {
      if (CONFIG_T::n_sequence*CONFIG_T::n_in / CONFIG_T::n_in > 1) {
          // #pragma HLS PIPELINE
      }
      DataPack: for (int i_pack = 0; i_pack < CONFIG_T::n_in; i_pack++) {
          #pragma HLS UNROLL
          data_in[i_pack] = data_stream[i_pack].read();
      }
      if (CONFIG_T::use_static)
        nnet::gru_static<data_T, res_T, CONFIG_T>(reset_state,data_in,h_newstate,param,param_zr,param_b, param_br);
      else
        nnet::gru<data_T, res_T, CONFIG_T>(reset_state,data_in,h_newstate,param,param_zr,param_b, param_br);
      if (CONFIG_T::n_sequence_out > 1){
        ResPack_sequences: for (int i_pack = 0; i_pack < CONFIG_T::n_out; i_pack++) {
            #pragma HLS UNROLL
            res_stream[i_pack].write(h_newstate[i_pack]);
        }
      }
      reset_state = false;
    }

    if (CONFIG_T::n_sequence_out == 1){
        ResPack: for (int i_pack = 0; i_pack < CONFIG_T::n_out; i_pack++) {
            #pragma HLS UNROLL
            res_stream[i_pack].write(h_newstate[i_pack]);
        }
    }

}

} // namespace nnet

#endif
