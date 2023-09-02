/**********
Copyright (c) 2018, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/

/*******************************************************************************
Description:
    HLS pragmas can be used to optimize the design : improve throughput, reduce latency and 
    device resource utilization of the resulting RTL code
    This is a wrapper to be used with an hls4ml project to enable proper handling by SDAccel
*******************************************************************************/
#include <iostream>
#include "myproject.h"
#include "kernel_params.h"

template<unsigned N> 
void fillWeights(const bigdata_t iWeightsIn[N], weight12_t weights[N]) { 
  for(int i0 = 0; i0 < N; i0++) { 
    weights[i0] = iWeightsIn[i0];
  }
}

template<unsigned N> 
void fillWeights1(const bigdata_t iWeightsIn[N], recurrent_weight12_t weights[N]) { 
  for(int i0 = 0; i0 < N; i0++) { 
    weights[i0] = iWeightsIn[i0];
  }
}

extern "C" {

void alveo_hls4ml(
    const bigdata_t *in0, // Read-Only Vector
    const bigdata_t *in1, // Read-Only Vector
    const bigdata_t *initial, // Read-Only Vector
	const bigdata_t *in_w120,
  const bigdata_t *in_w121,
	const bigdata_t *in_wr120,
  const bigdata_t *in_wr121,
    bigdata_t *out       // Output Result
    )
{
    #pragma HLS INTERFACE m_axi port=in0  offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=in1  offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=initial  offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi port=in_w120  offset=slave bundle=gmem3
    #pragma HLS INTERFACE m_axi port=in_w121  offset=slave bundle=gmem4
    #pragma HLS INTERFACE m_axi port=in_wr120  offset=slave bundle=gmem5
    #pragma HLS INTERFACE m_axi port=in_wr121  offset=slave bundle=gmem6
    #pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem4
    #pragma HLS INTERFACE s_axilite port=in0   bundle=control
    #pragma HLS INTERFACE s_axilite port=in1   bundle=control
    #pragma HLS INTERFACE s_axilite port=initial   bundle=control
    #pragma HLS INTERFACE s_axilite port=in_w120   bundle=control
    #pragma HLS INTERFACE s_axilite port=in_w121   bundle=control
    #pragma HLS INTERFACE s_axilite port=in_wr120   bundle=control
    #pragma HLS INTERFACE s_axilite port=in_wr121   bundle=control
    #pragma HLS INTERFACE s_axilite port=out  bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    #pragma HLS DATAFLOW

    //weight file, which are stored in URAM
    static weight12_t w120[786432/2];
    static weight12_t w121[786432/2];
    static recurrent_weight12_t wr120[786432/2];
    static recurrent_weight12_t wr121[786432/2];
    static bool fillWeights_ = false;

    if(!fillWeights_) {
      fillWeights<786432/2>(in_w120,w120);
      fillWeights<786432/2>(in_w121,w121);
      fillWeights1<786432/2>(in_wr120,wr120);
      fillWeights1<786432/2>(in_wr121,wr121);
      fillWeights_ = true;
    }else {

    input10_t in_bigbuf0[N_INPUT_1_10*N_INPUT_2_10/2];
    input10_t in_bigbuf1[N_INPUT_1_10*N_INPUT_2_10/2];
    layer11_t initial_bigbuf[512];
    layer12_t out_bigbuf;
    
    hls::stream<input10_t> in_buf0[1];
    hls::stream<input10_t> in_buf1[1];
    hls::stream<layer11_t> initial_buf[1];
    hls::stream<layer12_t> out_buf[1];

    //If input or output variable is array
    //#pragma HLS ARRAY_PARTITION   variable=in_buf  complete dim=0
    //#pragma HLS ARRAY_PARTITION   variable=out_buf complete dim=0
    #pragma HLS STREAM   variable=in_buf0  depth=256
    #pragma HLS STREAM   variable=in_buf1  depth=256
    #pragma HLS STREAM   variable=initial_buf  depth=512
    #pragma HLS STREAM   variable=out_buf depth=73
    
    //Get data from DRAM
    for (int i = 0; i < N_INPUT_1_10*N_INPUT_2_10/2; i++) {
        #pragma HLS PIPELINE II=1
        in_bigbuf0[i] = in0[i];
    }

    for (int i = 0; i < N_INPUT_1_10*N_INPUT_2_10/2; i++) {
        #pragma HLS PIPELINE II=1
        in_bigbuf1[i] = in1[i];
    }
    
    for (int i = 0; i < 512; i++) {
        #pragma HLS PIPELINE II=1
        initial_bigbuf[i] = initial[i];
    }
    //=============================================
    //Input
    //=============================================
    for(int i0 = 0; i0 < N_INPUT_1_10*N_INPUT_2_10/2; i0++) { 
        #pragma HLS PIPELINE II=1
        input10_t tmp = in_bigbuf0[i0];
        in_buf0[0].write(tmp);
    }

    for(int i0 = 0; i0 < N_INPUT_1_10*N_INPUT_2_10/2; i0++) { 
        #pragma HLS PIPELINE II=1
        input10_t tmp = in_bigbuf1[i0];
        in_buf1[0].write(tmp);
    }

    for(int i1 = 0; i1 < 512; i1++) { 
        #pragma HLS PIPELINE II=1
        layer11_t tmp1 = initial_bigbuf[i1];
        initial_buf[0].write(tmp1);
    }

    //=============================================
    //Start computation
    //=============================================

    std::cout<<"inf start"<<std::endl;
    myproject(in_buf0,in_buf1,initial_buf,out_buf,w120,w121,wr120,wr121);
    std::cout<<"inf end"<<std::endl;

    //=============================================
    //Output
    //=============================================
    for(int i0 = 0; i0 < 37376; i0++) {
        #pragma HLS PIPELINE II=1
        layer12_t tmp2 = out_buf[0].read();
        out[i0] = tmp2;
    }
}
}
}
