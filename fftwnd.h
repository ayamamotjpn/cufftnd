#ifndef FFTND_H_
#define FFTND_H_
#include <fftw3.h>

int getnt(int *n, int rank);
void copy_fftwc(fftwf_complex *x, fftwf_complex *y,int nt);
void copy_fftwz(fftw_complex *x, fftw_complex *y,int nt);
void fftwnd_c(fftwf_complex *x,int rank,int *n,int sign);
void fftwnd_z(fftw_complex *x,int rank,int *n,int sign);
#endif /* FFTND_H_ */