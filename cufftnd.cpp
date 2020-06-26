/* Copyright (c) 2020, Akiji Yamamoto.
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


#include <iostream>
#include <string>
#include <cuda.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>

#include "cufftnd.h"

// nd fft using cufft 1d and transpose matrix
void cufftnd_c(thrust::complex<float>* x, int dim, int n[], int mode) {
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

	cufftComplex *x_d;
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


// nd fft using cufft 1d and transpose matrix
void cufftnd_z(thrust::complex<double>* x, int dim, int n[], int mode) {
	// x should be a complex<double> 1D array of size n[0]*n[1]..n[idm-1]
	// dim is the dimension
	// array indices of n should be a fortran order
	// mode 0 for forward 1 for inverse transformation
	//cufftResult istat;
	int nt = 1;
	int ni[dim];
	for (int i = 0; i < dim; i++) {
		nt *= n[i];
	}
	for (int i = 1; i < dim; i++) {
		ni[i] = nt / n[i];
	}

	cufftDoubleComplex* x_d;
	cufftDoubleComplex *y_d;
	const cuDoubleComplex alpha={1.0,0.0};
	const cuDoubleComplex beta={0.0,0.0};
	cublasHandle_t handle;
	cublasCreate(&handle);
	cudaMalloc((void **) &x_d, sizeof(cufftDoubleComplex) * nt);
	cudaMalloc((void **) &y_d, sizeof(cufftDoubleComplex) * nt);
	cudaMemcpy(x_d, x, sizeof(cufftDoubleComplex) * nt, cudaMemcpyHostToDevice); // copy x to GPU mem
	cufftHandle plan;
	for (int i = 0; i < dim; i++) {
		cufftPlan1d(&plan,n[i], CUFFT_Z2Z, ni[i]); // 1D complex<double> Fourier transformation
		if(mode==0) {
		    cufftExecZ2Z(plan, x_d, y_d, CUFFT_FORWARD);   // y_d : Fourier coefficients
		} else {
			cufftExecZ2Z(plan, x_d, y_d, CUFFT_INVERSE);   // y_d : inverse Fourier coefficients
		}

		if (i < dim - 1) {
			cublasZgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, n[i], ni[i], &alpha,
					y_d, n[i], &beta, x_d, ni[i], y_d, ni[i]); // y_d : transpose of x_d
			//x_d = y_d;  // pointer copy
		}
	}
	cudaMemcpy(x, y_d, sizeof(cufftDoubleComplex) * nt, cudaMemcpyDeviceToHost); // copy back y_d to CPU mem
	cudaFree(x_d);
	cudaFree(y_d);
}


// 2^m d fft using cufft 2d and transpose matrix
void cufft2md_c(thrust::complex<float>* x, int dim, int n[], int mode) {
	// x should be a complex 1D array of size n[0]*n[1]..n[idm-1]
	// dim is the dimension
	// array indices of n should be a fortran order
	// mode 0 for forward 1 for inverse transformation
	int nt = 1;
	int ni[dim];

	//cufftResult istat;
	for (int i = 0; i < dim; i++) {
		nt *= n[i];
	}
	for (int i = 0; i < dim/2; i++) {
		ni[i] = nt / (n[i*2]*n[i*2+1]);
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
	for (int i = 0; i < dim/2; i++) {
		cufftPlanMany(&plan, 2, &n[i*2],NULL,1,0,NULL,1,0,CUFFT_C2C, ni[i]);  // 2D complex Fourier transformation
		if(mode==0) {
		    cufftExecC2C(plan, x_d, y_d, CUFFT_FORWARD);   // y_d : Fourier coefficients
		} else {
			cufftExecC2C(plan, x_d, y_d, CUFFT_INVERSE);   // y_d : inverse Fourier coefficients
		}

		if (i < dim/2 - 1) {
			cublasCgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, n[i], ni[i], &alpha,
					y_d, n[i], &beta, x_d, ni[i], y_d, ni[i]); // y_d : transpose of x_d
			//x_d = y_d;  // pointer copy
		}
	}
	cudaMemcpy(x, y_d, sizeof(cufftComplex) * nt, cudaMemcpyDeviceToHost); // copy back y_d to CPU mem
	cudaFree(x_d);
	cudaFree(y_d);
}

// 2^m d fft using cufft 2d and transpose matrix
void cufft2md_z(thrust::complex<double>* x, int dim, int n[], int mode) {
	// x should be a complex<double> 1D array of size n[0]*n[1]..n[idm-1]
	// dim is the dimension
	// array indices of n should be a fortran order
	// mode 0 for forward 1 for inverse transformation
	int nt = 1;
	int ni[dim];
	for (int i = 0; i < dim; i++) {
		nt *= n[i];
	}

	for (int i = 0; i < dim/2; i++) {
		ni[i] = nt / (n[i*2]*n[i*2+1]);
	}

	cufftDoubleComplex* x_d;
	cufftDoubleComplex *y_d;
	const cuDoubleComplex alpha={1.0,0.0};
	const cuDoubleComplex beta={0.0,0.0};
	cublasHandle_t handle;
	cublasCreate(&handle);
	cudaMalloc((void **) &x_d, sizeof(cufftDoubleComplex) * nt);
	cudaMalloc((void **) &y_d, sizeof(cufftDoubleComplex) * nt);
	cudaMemcpy(x_d, x, sizeof(cufftDoubleComplex) * nt, cudaMemcpyHostToDevice); // copy x to GPU mem
	cufftHandle plan;
	for (int i = 0; i < dim/2; i++) {
		cufftPlanMany(&plan, 2, &n[i*2],NULL,1,0,NULL,1,0,CUFFT_Z2Z,ni[i]); //2D complex<double> Fourier transformation
		if(mode==0) {
		    cufftExecZ2Z(plan, x_d, y_d, CUFFT_FORWARD);   // y_d : Fourier coefficients
		} else {
			cufftExecZ2Z(plan, x_d, y_d, CUFFT_INVERSE);   // y_d : inverse Fourier coefficients
		}

		if (i < dim/2 - 1) {
			cublasZgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, n[i], ni[i], &alpha,
					y_d, n[i], &beta, x_d, ni[i], y_d, ni[i]); // y_d : transpose of x_d
			//x_d = y_d;  // pointer copy
		}
	}
	cudaMemcpy(x, y_d, sizeof(cufftDoubleComplex) * nt, cudaMemcpyDeviceToHost); // copy back y_d to CPU mem
	cudaFree(x_d);
	cudaFree(y_d);
}

// 5d  fft using cufft 3d, 2d  and transpose matrix
void cufft5d_c(thrust::complex<float>* x, int n[], int mode) {
	// x should be a complex 1D array of size n[0]*n[1]..n[idm-1]
	// array indices of n should be a fortran order
	// mode 0 for forward 1 for inverse transformation
	int dim=5;
	int nt = 1;
	int ni[dim];
	for (int i = 0; i < dim; i++) {
		nt *= n[i];
	}

	ni[0] = nt / (n[0]*n[1]*n[2]);
	ni[1] = nt / (n[3]*n[4]);

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
	for (int i = 0; i < 2; i++) {
		cufftPlanMany(&plan, 2, &n[i*2],NULL,1,0,NULL,1,0,CUFFT_Z2Z,ni[i]); //2D  complex Fourier transformation
		if(mode==0) {
		    cufftExecC2C(plan, x_d, y_d, CUFFT_FORWARD);   // y_d : Fourier coefficients
		} else {
			cufftExecC2C(plan, x_d, y_d, CUFFT_INVERSE);   // y_d : inverse Fourier coefficients
		}

		if (i == 0) {
			cublasCgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, n[i], ni[i], &alpha,
					y_d, n[i], &beta, x_d, ni[i], y_d, ni[i]); // y_d : transpose of x_d
			//x_d = y_d;  // pointer copy
		}
	}
	cudaMemcpy(x, y_d, sizeof(cufftComplex) * nt, cudaMemcpyDeviceToHost); // copy back y_d to CPU mem
	cudaFree(x_d);
	cudaFree(y_d);
}


// 5d  fft using cufft 3d, 2d  and transpose matrix
void cufft5d_z(thrust::complex<double>* x, int n[], int mode) {
	// x should be a complex<double> 1D array of size n[0]*n[1]..n[idm-1]
	// array indices of n should be a fortran order
	// mode 0 for forward 1 for inverse transformation
	int dim=5;
	int nt = 1;
	int ni[dim];
	for (int i = 0; i < dim; i++) {
		nt *= n[i];
	}

	ni[0] = nt / (n[0]*n[1]*n[2]);
	ni[1] = nt / (n[3]*n[4]);

	cufftDoubleComplex* x_d;
	cufftDoubleComplex *y_d;
	const cuDoubleComplex alpha={1.0,0.0};
	const cuDoubleComplex beta={0.0,0.0};
	cublasHandle_t handle;
	cublasCreate(&handle);
	cudaMalloc((void **) &x_d, sizeof(cufftDoubleComplex) * nt);
	cudaMalloc((void **) &y_d, sizeof(cufftDoubleComplex) * nt);
	cudaMemcpy(x_d, x, sizeof(cufftDoubleComplex) * nt, cudaMemcpyHostToDevice); // copy x to GPU mem
	cufftHandle plan;
	for (int i = 0; i < 2; i++) {
		cufftPlanMany(&plan, 2, &n[i*2],NULL,1,0,NULL,1,0,CUFFT_Z2Z,ni[i]); //2D complex<double> Fourier transformation
		if(mode==0) {
		    cufftExecZ2Z(plan, x_d, y_d, CUFFT_FORWARD);   // y_d : Fourier coefficients
		} else {
			cufftExecZ2Z(plan, x_d, y_d, CUFFT_INVERSE);   // y_d : inverse Fourier coefficients
		}

		if (i == 0) {
			cublasZgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, n[i], ni[i], &alpha,
					y_d, n[i], &beta, x_d, ni[i], y_d, ni[i]); // y_d : transpose of x_d
			//x_d = y_d;  // pointer copy
		}
	}
	cudaMemcpy(x, y_d, sizeof(cufftDoubleComplex) * nt, cudaMemcpyDeviceToHost); // copy back y_d to CPU mem
	cudaFree(x_d);
	cudaFree(y_d);
}

// 3^m d fft using cufft 3d and transpose matrix
void cufft3md_c(thrust::complex<float>* x, int dim, int n[], int mode) {
	// x should be a complex 1D array of size n[0]*n[1]..n[idm-1]
	// dim is the dimension
	// array indices of n should be a fortran order
	// mode 0 for forward 1 for inverse transformation
	int nt = 1;
	int ni[dim];

	//cufftResult istat;
	for (int i = 0; i < dim; i++) {
		nt *= n[i];
	}
	for (int i = 0; i < dim/3; i++) {
		ni[i] = nt / (n[i*3]*n[i*3+1]);
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
	for (int i = 0; i < dim/3; i++) {
		cufftPlanMany(&plan, 3, &n[i*3],NULL,1,0,NULL,1,0,CUFFT_C2C, ni[i]);  // 3D complex Fourier transformation
		if(mode==0) {
		    cufftExecC2C(plan, x_d, y_d, CUFFT_FORWARD);   // y_d : Fourier coefficients
		} else {
			cufftExecC2C(plan, x_d, y_d, CUFFT_INVERSE);   // y_d : inverse Fourier coefficients
		}

		if (i < dim/3 - 1) {
			cublasCgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, n[i], ni[i], &alpha,
					y_d, n[i], &beta, x_d, ni[i], y_d, ni[i]); // y_d : transpose of x_d
			//x_d = y_d;  // pointer copy
		}
	}
	cudaMemcpy(x, y_d, sizeof(cufftComplex) * nt, cudaMemcpyDeviceToHost); // copy back y_d to CPU mem
	cudaFree(x_d);
	cudaFree(y_d);
}

// 3^m d fft using cufft 3d and transpose matrix
void cufft3md_z(thrust::complex<double>* x, int dim, int n[], int mode) {
	// x should be a complex<double> 1D array of size n[0]*n[1]..n[idm-1]
	// dim is the dimension
	// array indices of n should be a fortran order
	// mode 0 for forward 1 for inverse transformation
	int nt = 1;
	int ni[dim];
	for (int i = 0; i < dim; i++) {
		nt *= n[i];
	}

	for (int i = 0; i < dim/2; i++) {
		ni[i] = nt / (n[i*2]*n[i*2+1]);
	}

	cufftDoubleComplex* x_d;
	cufftDoubleComplex *y_d;
	const cuDoubleComplex alpha={1.0,0.0};
	const cuDoubleComplex beta={0.0,0.0};
	cublasHandle_t handle;
	cublasCreate(&handle);
	cudaMalloc((void **) &x_d, sizeof(cufftDoubleComplex) * nt);
	cudaMalloc((void **) &y_d, sizeof(cufftDoubleComplex) * nt);
	cudaMemcpy(x_d, x, sizeof(cufftDoubleComplex) * nt, cudaMemcpyHostToDevice); // copy x to GPU mem
	cufftHandle plan;
	for (int i = 0; i < dim/2; i++) {
		cufftPlanMany(&plan, 2, &n[i*2],NULL,1,0,NULL,1,0,CUFFT_Z2Z,ni[i]); //2D complex<double> Fourier transformation
		if(mode==0) {
		    cufftExecZ2Z(plan, x_d, y_d, CUFFT_FORWARD);   // y_d : Fourier coefficients
		} else {
			cufftExecZ2Z(plan, x_d, y_d, CUFFT_INVERSE);   // y_d : inverse Fourier coefficients
		}

		if (i < dim/2 - 1) {
			cublasZgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, n[i], ni[i], &alpha,
					y_d, n[i], &beta, x_d, ni[i], y_d, ni[i]); // y_d : transpose of x_d
			//x_d = y_d;  // pointer copy
		}
	}
	cudaMemcpy(x, y_d, sizeof(cufftDoubleComplex) * nt, cudaMemcpyDeviceToHost); // copy back y_d to CPU mem
	cudaFree(x_d);
	cudaFree(y_d);
}

// wrapper for 4D fft
void cufft4d_c(thrust::complex<float>* x, int n[], int mode) {
	cufft2md_c(x, 4, n, mode);
}

// wrapper for 4D fft
void cufft4d_z(thrust::complex<double>* x, int n[], int mode) {
	cufft2md_z(x, 4, n, mode);
}

// wrapper for 6D fft
void cufft6d_c(thrust::complex<float>* x, int n[], int mode) {
	cufft3md_c(x, 6, n, mode);
}

// wrapper for 6D fft
void cufft6d_z(thrust::complex<double>* x, int n[], int mode) {
	cufft3md_z(x, 6, n, mode);
}
