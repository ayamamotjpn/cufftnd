/* Copyright (c) 2020, Akiji Yamamoto. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include <complex.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <cuda_runtime.h>

#include "cufftnd.h"

//void cufftnd_c(float complex *x, int dim, int n[], int mode){};  // for test

void cufftnd_c(float complex* x, int dim, int n[], int mode) {
	// x should be a complex 1D array of size n[0]*n[1]..n[idm-1]
	// dim is the dimension
	// array indices of n should be a fortran order
	// mode 0 for forward 1 for inverse transformation
	int nt = 1;
	int ni[dim];
	for (int i = 0; i < dim; i++) {
		nt *= n[i];
	}
	for (int i = 1; i < dim; i++) {
		ni[i] = nt / n[i];
	}

	cufftComplex* x_d;
	cufftComplex *y_d;
	const cuComplex alpha={1.0,0.0};
	const cuComplex beta={0.0,0.0};
	cublasHandle_t handle;
	cublasCreate(&handle);
	cudaMalloc((void **) &x_d, sizeof(cufftComplex) * nt);
	cudaMalloc((void **) &y_d, sizeof(cufftComplex) * nt);
	cudaMemcpy(x_d, x, sizeof(cufftComplex) * nt, cudaMemcpyHostToDevice); // copy x to GPU mem
	cufftHandle plan;
	for (int i = 0; i < dim; i++) {
		cufftPlan1d(&plan, n[i], CUFFT_C2C, ni[i]);        // 1D complex Fourier transformation
		if(mode==0) {
		    cufftExecC2C(plan, x_d, y_d, CUFFT_FORWARD);   // y_d : Fourier coefficients
		} else {
			cufftExecC2C(plan, x_d, y_d, CUFFT_INVERSE);   // y_d : Fourier coefficients
		}

		if (i < dim - 1) {
			cublasCgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, n[i], ni[i], &alpha,
					y_d, n[i], &beta, x_d, ni[i], y_d, ni[i]); // y_d : transpose of x_d
			//x_d = y_d;  // pointer copy
		}
	}
	cudaMemcpy(x, y_d, sizeof(cufftComplex) * nt, cudaMemcpyDeviceToHost); // copy back y_d to CPU mem
	cudaFree(x_d);
	cudaFree(y_d);
}

int main() {
	double complex cacos(double complex z);
	float complex cacosf(float complex z);
	long double complex cacosl(long double complex z);
}

