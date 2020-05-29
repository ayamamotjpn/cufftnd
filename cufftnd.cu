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

#include <iostream>
#include <string>
#include <cuda.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <cuda_runtime.h>

#include "mtrns_gpu2.h"

extern "C" {

void testwt_c1(cuComplex *x, int n) {
	for (int i=0; i<5; i++) {
		std::cout<< x[i].x <<" "<< x[i].y <<" ";
	}
	printf("\n");
	for (int i=n-5; i<n; i++) {
		std::cout<< x[i].x <<" "<< x[i].y <<" ";
	}
	printf("\n");
}

void testwt_z1(cuDoubleComplex *x, int n) {
	for (int i=0; i<5; i++) {
		std::cout<< x[i].x <<" "<<x[i].x<<" ";
	}
	printf("\n");
	for (int i=n-5; i<n; i++) {
		std::cout<< x[i].x <<" "<< x[i].y<<" ";
	}
	printf("\n");
}

void testout(int dim, int dim0, int dim1, int mode, int nt, int *n, int *n0, int *ni) {
	std::cout << "dim " << dim << " mode " << mode 	<< " nt " << nt << "\n";
	std::cout << "n ";
	for (int i=0; i<dim; i++) {
		std::cout << n[i]<<" ";
	}
	std::cout << "\n";
	std::cout << "n0 ";
	for (int i=0; i<dim0; i++) {
		std::cout << n0[i]<<" ";
	}
	std::cout << "\n";

	std::cout << "ni ";
	for (int i=0; i<dim1; i++) {
		std::cout << ni[i]<<" ";
	}
	std::cout << "\n";
}

int get_nt(int *n, int dim) {
	int nt = 1;
	for (int i = 0; i < dim; i++) {
		nt *= n[i];
	}
	return nt;
}

void getn0i(int *n,int *n0, int *ni, int dim, int nt) {
	for (int i = 0; i < dim; i++) {
		n0[i] = n[i];
		ni[i] = nt / n0[i];
	}
}

// nd fft using cufft 1d and transpose matrix
void cufftnd_c(cuComplex* x, int dim, int n[], int mode) {
	// x should be a complex 1D array of size n[0]*n[1]..n[idm-1]
	// dim is the dimension
	// array indices of n should be a fortran order
	// mode 0 for forward 1 for inverse transformation
	int ni[dim];
	int n0[dim];

	int nt = get_nt(n,dim);
	getn0i(n, n0, ni, dim,nt);

	//std::cout << "x in cufftnd_c\n";
	//testwt_c1(x, nt);  // for test

	//testout(dim, dim, dim, mode, nt, n, n0, ni);

	cuComplex* x_d;
	//cuComplex *y_d;
	cuComplex *z_d;
	const cuComplex alpha={1.0,0.0};
	const cuComplex beta={0.0,0.0};
	cublasHandle_t handle;
	cublasCreate(&handle);

	int memsize = sizeof(cuComplex) * nt;
	cudaMalloc((void **) &x_d, memsize);
	//cudaMalloc((void **) &y_d, memsize);
	cudaMalloc((void **) &z_d, memsize);  // for transpose
	cudaMemcpy(x_d, x, memsize, cudaMemcpyHostToDevice); // copy x to GPU mem
	cufftHandle plan;

	for (int i = 0; i < dim; i++) {
		cufftPlan1d(&plan, n[i], CUFFT_C2C, ni[i]);        // 1D complex Fourier transformation
		if(mode==1) {
		    cufftExecC2C(plan, x_d, x_d, CUFFT_FORWARD);   // x_d : Fourier coefficients
		} else if(mode==-1) {
			cufftExecC2C(plan, x_d, x_d, CUFFT_INVERSE);   // x_d : Fourier coefficients
		}
		cudaDeviceSynchronize();
		cudaMemcpy(x, x_d, memsize, cudaMemcpyDeviceToHost); // copy back x_d to CPU mem
		//testwt_c1(x, nt);  // for test

		cublasCgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, ni[i], n0[i], &alpha,
			x_d, n0[i], &beta, z_d, ni[i], z_d, ni[i]); // z_d : transpose of x_d
		//x_d = z_d;  // pointer copy
		cudaMemcpy(x_d, z_d, memsize, cudaMemcpyDeviceToDevice); // copy z_d to x_d
	}
	//std::cout <<"end of for loop\n";
	cudaMemcpy(x, x_d, memsize, cudaMemcpyDeviceToHost); // copy back x_d to CPU mem
	cudaFree(x_d);
	//cudaFree(y_d);
	cudaFree(z_d);
}


// nd fft using cufft 1d and transpose matrix
void cufftnd_z(cuDoubleComplex* x, int dim, int n[], int mode) {
	// x should be a cuDoubleComplex 1D array of size n[0]*n[1]..n[idm-1]
	// dim is the dimension
	// array indices of n should be a fortran order
	// mode 0 for forward 1 for inverse transformation
	//cufftResult istat;
	int ni[dim];
	int n0[dim];
	int nt = get_nt(n,dim);
	getn0i(n, n0, ni, dim, nt);

	//std::cout << "x in cufftnd_z\n";
	//testwt_z1(x, nt);  // for test

	//testout(dim, dim, dim, mode, nt, n, n0, ni);

	cuDoubleComplex* x_d;
	//cuDoubleComplex *y_d;
	cuDoubleComplex *z_d;
	const cuDoubleComplex alpha={1.0,0.0};
	const cuDoubleComplex beta={0.0,0.0};
	cublasHandle_t handle;
	cublasCreate(&handle);
	int memsize = sizeof(cuDoubleComplex) * nt;
	cudaMalloc((void **) &x_d, memsize);
	//cudaMalloc((void **) &y_d, memsize);
	cudaMalloc((void **) &z_d, memsize);  // for transpose
	cudaMemcpy(x_d, x, memsize, cudaMemcpyHostToDevice); // copy x to GPU mem
	cufftHandle plan;
	for (int i = 0; i < dim; i++) {
		cufftPlan1d(&plan,n[i], CUFFT_Z2Z, ni[i]); // 1D cuDoubleComplex Fourier transformation
		if(mode==1) {
		    cufftExecZ2Z(plan, x_d, x_d, CUFFT_FORWARD);   // x_d : Fourier coefficients
		} else if(mode==-1) {
			cufftExecZ2Z(plan, x_d, x_d, CUFFT_INVERSE);   // x_d : inverse Fourier coefficients
		}
		cudaDeviceSynchronize();
		cudaMemcpy(x, x_d, memsize, cudaMemcpyDeviceToHost); // copy back x_d to CPU mem
		//testwt_z1(x, nt);  // for test

		//std::cout << i << " dim " << dim <<"\n";
		cublasZgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, ni[i], n0[i], &alpha,
			x_d, n0[i], &beta, z_d, ni[i], z_d, ni[i]); // z_d : transpose of x_d
		//x_d = z_d;  // pointer copy
		cudaMemcpy(x_d, z_d, memsize, cudaMemcpyDeviceToDevice); // copy z_d to x_d
	}
	//std::cout <<"end of for loop\n";
	cudaMemcpy(x, x_d, memsize, cudaMemcpyDeviceToHost); // copy back x_d to CPU mem
	cudaFree(x_d);
	//cudaFree(y_d);
	cudaFree(z_d);
}

void get_4d_ni0(int *n,int *n0,int *ni,int dim,int nt) {
	for (int i = 0; i < dim/2; i++) {
		n0[i] = n[i*2]*n[i*2+1];
		ni[i] = nt / n0[i];
	}
}

// 2^m d fft using cufft 2d and transpose matrix
void cufft2md_c(cuComplex* x, int dim, int n[], int mode) {
	// x should be a complex 1D array of size n[0]*n[1]..n[idm-1]
	// dim is the dimension
	// array indices of n should be a fortran order
	// mode 0 for forward 1 for inverse transformation

	int ni[dim];
	int n0[dim];

	//cufftResult istat;
	int nt = get_nt(n,dim);
	get_4d_ni0(n,n0,ni,dim,nt);

	//testout(dim, 2, 2, mode, nt, n, n0, ni);

	cuComplex *x_d;
	cuComplex *y_d;
	cuComplex *z_d;
	const cuComplex alpha={1.0,0.0};
	const cuComplex beta={0.0,0.0};
	cublasHandle_t handle;
	cublasCreate(&handle);
	int memsize = sizeof(cuComplex) * nt;
	cudaMalloc((void **) &x_d, memsize);
	cudaMalloc((void **) &y_d, memsize);
	cudaMalloc((void **) &z_d, memsize);  // for transpose
	cudaMemcpy(x_d, x, memsize, cudaMemcpyHostToDevice); // copy x to GPU mem
	cufftHandle plan;
	for (int i = 0; i < dim/2; i++) {
		cufftPlanMany(&plan, 2, &n[i*2],NULL,1,0,NULL,1,0,CUFFT_C2C, ni[i]);  // 2D complex Fourier transformation
		if(mode==1) {
		    cufftExecC2C(plan, x_d, y_d, CUFFT_FORWARD);   // y_d : Fourier coefficients of x_d
		} else if(mode==-1) {
			cufftExecC2C(plan, x_d, y_d, CUFFT_INVERSE);   // y_d : inverse Fourier coefficients of x_d
		}
		cudaDeviceSynchronize();
		cudaMemcpy(x, y_d, memsize, cudaMemcpyDeviceToHost); // copy back y_d to CPU mem
		//testwt_c1(x, nt);  // for test

		//std::cout<< "i "<< i << " dim/2 " << dim/2 <<"\n";
		cublasCgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, ni[i], n0[i], &alpha,
			y_d, n0[i], &beta, z_d, ni[i], z_d, ni[i]); // z_d : transpose of y_d
		cudaMemcpy(x_d, z_d, memsize, cudaMemcpyDeviceToDevice); // copy z_d to x_d
	}
	cudaMemcpy(x, x_d, memsize, cudaMemcpyDeviceToHost); // copy back x_d to CPU mem
	cudaFree(x_d);
	cudaFree(y_d);
	cudaFree(z_d);
}

// 2^m d fft using cufft 2d and transpose matrix
void cufft2md_z(cuDoubleComplex* x, int dim, int n[], int mode) {
	// x should be a cuDoubleComplex 1D array of size n[0]*n[1]..n[idm-1]
	// dim is the dimension
	// array indices of n should be a fortran order
	// mode 0 for forward 1 for inverse transformation
	int ni[dim];
	int n0[dim];

	int nt = get_nt(n,dim);
	get_4d_ni0(n,n0,ni,dim,nt);
	//testout(dim, 2, 2, mode, nt, n, n0, ni);

	cuDoubleComplex *x_d;
	cuDoubleComplex *y_d;
	cuDoubleComplex *z_d;
	const cuDoubleComplex alpha={1.0,0.0};
	const cuDoubleComplex beta={0.0,0.0};
	cublasHandle_t handle;
	cublasCreate(&handle);
	int memsize = sizeof(cuDoubleComplex) * nt;
	cudaMalloc((void **) &x_d, memsize);
	cudaMalloc((void **) &y_d, memsize);
	cudaMalloc((void **) &z_d, memsize);  // for transpose
	cudaMemcpy(x_d, x, memsize, cudaMemcpyHostToDevice); // copy x to GPU mem
	cufftHandle plan;
	for (int i = 0; i < dim/2; i++) {
		cufftPlanMany(&plan, 2, &n[i*2],NULL,1,0,NULL,1,0,CUFFT_Z2Z,ni[i]); //2D cuDoubleComplex Fourier transformation
		if(mode==1) {
		    cufftExecZ2Z(plan, x_d, y_d, CUFFT_FORWARD);   // y_d : Fourier coefficients of x_d
		} else if(mode==-1) {
			cufftExecZ2Z(plan, x_d, y_d, CUFFT_INVERSE);   // y_d : inverse Fourier coefficients of x_d
		}
		cudaDeviceSynchronize();
		cudaMemcpy(x, y_d, memsize, cudaMemcpyDeviceToHost); // copy back x_d to CPU mem
		//testwt_z1(x, nt);  // for test

		cublasZgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, ni[i], n0[i], &alpha,
			y_d, n0[i], &beta, z_d, ni[i], z_d, ni[i]); // z_d : transpose of y_d
		cudaMemcpy(x_d, z_d, memsize, cudaMemcpyDeviceToDevice); // copy z_d to x_d
	}
	cudaMemcpy(x, x_d, memsize, cudaMemcpyDeviceToHost); // copy back x_d to CPU mem
	cudaFree(x_d);
	cudaFree(y_d);
	cudaFree(z_d);
}


void get_5d_n0i(int *n,int *n0,int *ni,int dim,int nt) {
	n0[0] = n[0]*n[1]*n[2];
	n0[1] = n[3]*n[4];
	ni[0] = nt / n0[0];
	ni[1] = nt / n0[1];
}

void get_idim(int *idim) {
	idim[0]=3;
	idim[1]=2;
}

/*
	n0[0] = n[0]*n[1]*n[2];
	n0[1] = n[3]*n[4];
	ni[0] = nt / n0[0];
	ni[1] = nt / n0[1];
*/

// 5d  fft using cufft 3d, 2d  and transpose matrix
void cufft5d_c(cuComplex* x, int n[], int mode) {
	// x should be a complex 1D array of size n[0]*n[1]..n[idm-1]
	// array indices of n should be a fortran order
	// mode 0 for forward 1 for inverse transformation
	int dim=5;

	int ni[2];
	int n0[2];
	int idim[2];

	get_idim(idim);
	int nt = get_nt(n,dim);
	get_5d_n0i(n,n0,ni,dim,nt);
	//testout(dim, 2, 2, mode, nt, n, n0, ni);

	cuComplex *x_d;
	cuComplex *y_d;
	cuComplex *z_d;
	const cuComplex alpha={1.0,0.0};
	const cuComplex beta={0.0,0.0};
	cublasHandle_t handle;
	cublasCreate(&handle);
	int memsize = sizeof(cuComplex) * nt;
	cudaMalloc((void **) &x_d, memsize);
	cudaMalloc((void **) &y_d, memsize);
	cudaMalloc((void **) &z_d, memsize);  // for transpose
	cudaMemcpy(x_d, x, memsize, cudaMemcpyHostToDevice); // copy x to GPU mem
	cufftHandle plan;
	for (int i = 0; i < 2; i++) {
		cufftPlanMany(&plan, idim[i], &n[i*3],NULL,1,0,NULL,1,0,CUFFT_Z2Z,ni[i]); //2D  complex Fourier transformation
		if(mode==1) {
		    cufftExecC2C(plan, x_d, y_d, CUFFT_FORWARD);   // y_d : Fourier coefficients of x_d
		} else if(mode==-1) {
			cufftExecC2C(plan, x_d, y_d, CUFFT_INVERSE);   // y_d : inverse Fourier coefficients of x_d
		}
		cudaDeviceSynchronize();
		cublasCgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, ni[i], n0[i], &alpha,
			y_d, n0[i], &beta, z_d, ni[i], z_d, ni[i]); // z_d : transpose of y_d
		cudaMemcpy(x_d, z_d, memsize, cudaMemcpyDeviceToDevice); // copy z_d to x_d
	}
	cudaMemcpy(x, x_d, memsize, cudaMemcpyDeviceToHost); // copy back x_d to CPU mem
	cudaFree(x_d);
	cudaFree(y_d);
	cudaFree(z_d);
}

// 5d  fft using cufft 3d, 2d  and transpose matrix
void cufft5d_z(cuDoubleComplex* x, int n[], int mode) {
	// x should be a cuDoubleComplex 1D array of size n[0]*n[1]..n[idm-1]
	// array indices of n should be a fortran order
	// mode 0 for forward 1 for inverse transformation
	int dim=5;
	int ni[2];
	int n0[2];
	int idim[2];

	get_idim(idim);
	int nt = get_nt(n,dim);
	get_5d_n0i(n,n0,ni,dim,nt);
	//testout(dim, 2, 2, mode, nt, n, n0, ni);

	cuDoubleComplex *x_d;
	cuDoubleComplex *y_d;
	cuDoubleComplex *z_d;
	const cuDoubleComplex alpha={1.0,0.0};
	const cuDoubleComplex beta={0.0,0.0};
	cublasHandle_t handle;
	cublasCreate(&handle);
	int memsize = sizeof(cuDoubleComplex) * nt;
	cudaMalloc((void **) &x_d, memsize);
	cudaMalloc((void **) &y_d, memsize);
	cudaMalloc((void **) &z_d, memsize);  // for transpose
	cudaMemcpy(x_d, x, memsize, cudaMemcpyHostToDevice); // copy x to GPU mem
	cufftHandle plan;
	for (int i = 0; i < 2; i++) {
		cufftPlanMany(&plan, idim[i], &n[i*3],NULL,1,0,NULL,1,0,CUFFT_Z2Z,ni[i]); //2D cuDoubleComplex Fourier transformation
		if(mode==1) {
		    cufftExecZ2Z(plan, x_d, y_d, CUFFT_FORWARD);   // y_d : Fourier coefficients of x_d
		} else if(mode==-1) {
			cufftExecZ2Z(plan, x_d, y_d, CUFFT_INVERSE);   // y_d : inverse Fourier coefficients of x_d
		}
		cudaDeviceSynchronize();
		cublasZgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, ni[i], n0[i], &alpha,
			y_d, n0[i], &beta, z_d, ni[i], z_d, ni[i]); // z_d : transpose of y_d
		cudaMemcpy(x_d, z_d, memsize, cudaMemcpyDeviceToDevice); // copy z_d to x_d
	}
	cudaMemcpy(x, z_d, memsize, cudaMemcpyDeviceToHost); // copy back x_d to CPU mem
	cudaFree(x_d);
	cudaFree(y_d);
	cudaFree(z_d);
}


void get_3md_n0i(int *n,int *n0,int *ni,int dim,int nt) {
	for (int i = 0; i < dim/3; i++) {
		n0[i] = n[i*3]*n[i*3+1]*n[i*3+2];
		ni[i] = nt / n0[i];
	}
}

// 3^m d fft using cufft 3d and transpose matrix
void cufft3md_c(cuComplex* x, int dim, int n[], int mode) {
	// x should be a complex 1D array of size n[0]*n[1]..n[idm-1]
	// dim is the dimension
	// array indices of n should be a fortran order
	// mode 0 for forward 1 for inverse transformation
	int ni[dim];
	int n0[dim];

	//cufftResult istat;
	int nt = get_nt(n,dim);
	get_3md_n0i(n,n0,ni,dim,nt);
	//testout(dim, 2, 2, mode, nt, n, n0, ni);

	cuComplex *x_d;
	cuComplex *y_d;
	cuComplex *z_d;
	const cuComplex alpha={1.0,0.0};
	const cuComplex beta={0.0,0.0};
	cublasHandle_t handle;
	cublasCreate(&handle);
	int memsize = sizeof(cuComplex) * nt;
	cudaMalloc((void **) &x_d, memsize);
	cudaMalloc((void **) &y_d, memsize);
	cudaMalloc((void **) &z_d, memsize);  // for transpose
	cudaMemcpy(x_d, x, memsize, cudaMemcpyHostToDevice); // copy x to GPU mem
	cufftHandle plan;
	for (int i = 0; i < dim/3; i++) {
		cufftPlanMany(&plan, 3, &n[i*3],NULL,1,0,NULL,1,0,CUFFT_C2C, ni[i]);  // 3D complex Fourier transformation
		if(mode==1) {
		    cufftExecC2C(plan, x_d, y_d, CUFFT_FORWARD);   // y_d : Fourier coefficients of x_d
		} else if(mode==-1) {
			cufftExecC2C(plan, x_d, y_d, CUFFT_INVERSE);   // y_d : inverse Fourier coefficients of x_d
		}
		cudaDeviceSynchronize();
		cublasCgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, ni[i], n0[i], &alpha,
			y_d, n0[i], &beta, z_d, ni[i], z_d, ni[i]); // z_d : transpose of y_d
		cudaMemcpy(x_d, z_d, memsize, cudaMemcpyDeviceToDevice); // copy z_d to x_d
	}
	cudaMemcpy(x, x_d, memsize, cudaMemcpyDeviceToHost); // copy back x_d to CPU mem
	cudaFree(x_d);
	cudaFree(y_d);
	cudaFree(z_d);
}

// 3^m d fft using cufft 3d and transpose matrix
void cufft3md_z(cuDoubleComplex* x, int dim, int n[], int mode) {
	// x should be a cuDoubleComplex 1D array of size n[0]*n[1]..n[idm-1]
	// dim is the dimension
	// array indices of n should be a fortran order
	// mode 0 for forward 1 for inverse transformation
	int ni[dim];
	int n0[dim];
	int nt = get_nt(n,dim);
	get_3md_n0i(n,n0,ni,dim,nt);
	//testout(dim, 2, 2, mode, nt, n, n0, ni);

	cuDoubleComplex *x_d;
	cuDoubleComplex *y_d;
	cuDoubleComplex *z_d;
	const cuDoubleComplex alpha={1.0,0.0};
	const cuDoubleComplex beta={0.0,0.0};
	cublasHandle_t handle;
	cublasCreate(&handle);
	int memsize = sizeof(cuDoubleComplex) * nt;
	cudaMalloc((void **) &x_d, memsize);
	cudaMalloc((void **) &y_d, memsize);
	cudaMalloc((void **) &z_d, memsize);  // for transpose
	cudaMemcpy(x_d, x, memsize, cudaMemcpyHostToDevice); // copy x to GPU mem
	cufftHandle plan;
	for (int i = 0; i < dim/3; i++) {
		cufftPlanMany(&plan, 3, &n[i*3],NULL,1,0,NULL,1,0,CUFFT_Z2Z,ni[i]); //3D cuDoubleComplex Fourier transformation
		if(mode==1) {
		    cufftExecZ2Z(plan, x_d, y_d, CUFFT_FORWARD);   // y_d : Fourier coefficients of x_d
		} else if (mode==-1) {
			cufftExecZ2Z(plan, x_d, y_d, CUFFT_INVERSE);   // y_d : inverse Fourier coefficients of x_d
		}
		cudaDeviceSynchronize();
		cublasZgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, ni[i], n0[i], &alpha,
			y_d, n0[i], &beta, z_d, ni[i], z_d, ni[i]); // z_d : transpose of y_d
		cudaMemcpy(x_d, z_d, memsize, cudaMemcpyDeviceToDevice); // copy z_d to x_d
	}
	cudaMemcpy(x, x_d, memsize, cudaMemcpyDeviceToHost); // copy back x_d to CPU mem
	cudaFree(x_d);
	cudaFree(y_d);
	cudaFree(z_d);
}

// wrapper for 4D fft
void cufft4d_c(cuComplex* x, int n[], int mode) {
	cufft2md_c(x, 4, n, mode);
}

// wrapper for 4D fft
void cufft4d_z(cuDoubleComplex* x, int n[], int mode) {
	cufft2md_z(x, 4, n, mode);
}

// wrapper for 6D fft
void cufft6d_c(cuComplex* x, int n[], int mode) {
	cufft3md_c(x, 6, n, mode);
}

// wrapper for 6D fft
void cufft6d_z(cuDoubleComplex* x, int n[], int mode) {
	cufft3md_z(x, 6, n, mode);
}

}  // end of extern "C"