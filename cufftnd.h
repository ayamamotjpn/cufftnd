/*
 *cufftnd.h
 *
 *  Created on: 2017/10/18
 *      Author: yamamoto
 */

#ifndef GPU_FFTND_H_
#define GPU_FFTND_H_
#include <complex.h>

//void cufftnd_c(float complex *x, int dim, int n[], int mode);
//void cufftnd_z(double complex *x, int dim, int n[], int mode);
int get_nt(int *n, int dim);

void cufftnd_c(float complex *x, int dim, int n[], int mode);
void cufftnd_z(double complex *x, int dim, int n[], int mode);
void cufft2md_c(float complex *x, int dim, int n[], int mode);
void cufft2md_z(double complex *x, int dim, int n[], int mode);
void cufft3md_c(float complex *x, int dim, int n[], int mode);
void cufft3md_z(double complex *x, int dim, int n[], int mode);

void cufft4d_c(float complex *x, int n[], int mode);
void cufft4d_z(double complex *x, int n[], int mode);
void cufft5d_c(float complex *x, int n[], int mode);
void cufft5d_z(double complex *x, int n[], int mode);
void cufft6d_c(float complex *x, int n[], int mode);
void cufft6d_z(double complex *x, int n[], int mode);

#endif /* GPU_FFTND_H_ */
