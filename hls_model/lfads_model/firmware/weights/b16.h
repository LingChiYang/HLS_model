//Numpy array shape [4]
//Min 0.000000000000
//Max 0.000000000000
//Number of zeros 4

#ifndef B16_H_
#define B16_H_

#ifndef __SYNTHESIS__
bias16_t b16[4];
#else
bias16_t b16[4] = {0, 0, 0, 0};
#endif

#endif
