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
#include <stdlib.h>
#include <stdio.h>
#include "cufftnd.h"
#include "fftwnd.h"

void Uniform_c(float complex x[], int n){
	for(int i=0; i<n; i++) {
    	float xr = (float)rand()/(float)RAND_MAX;
    	float xi = (float)rand()/(float)RAND_MAX;
    	//printf("xr %f xi %f",xr,xi);
    	x[i] = CMPLXF(xr,xi);  // set complex value
	}
}

void Uniform_z(double complex x[], int n){
	for(int i=0; i<n; i++) {
    	double xr = (double)rand()/(double)RAND_MAX;
    	double xi = (double)rand()/(double)RAND_MAX;
    	x[i] = CMPLX(xr,xi);  // set complex value
	}
}

void copy_c(float complex *y, float complex *x,int n) {
	for(int i=0; i<n; i++) {
		y[i] = x[i];
	}
}

void copy_z(double complex *y, double complex *x,int n) {
	for(int i=0; i<n; i++) {
		y[i] = x[i];
	}
}

int check_c(float complex *x,float complex *y,int n) {
	float eps = 0.00001;
	for(int i=0; i<n; i++) {
		if (abs(x[i]-y[i]) > eps) {return 0;};  // false
	}
	return 1;  // true
}

int check_z(double complex *x,double complex *y,int n) {
	double eps = 1.e-12;
	for(int i=0; i<n; i++) {
		if (abs(x[i]-y[i]) > eps) {return 0;};  // false
	}
	return 1;  // true
}

void testwt_c(float complex *x, int n) {
	for (int i=0; i<5; i++) {
		printf("%f %f ",creal(x[i]),cimag(x[i]));
	}
	printf("\n");
	for (int i=n-5; i<n; i++) {
		printf("%f %f ",creal(x[i]),cimag(x[i]));
	}
	printf("\n");
}

void testwt_z(double complex *x, int n) {
	for (int i=0; i<5; i++) {
		printf("%f %f ",creal(x[i]),cimag(x[i]));
	}
	printf("\n");
	for (int i=n-5; i<n; i++) {
		printf("%f %f ",creal(x[i]),cimag(x[i]));
	}
	printf("\n");
}

void scale_c(float complex *x, int nt) {
	for(int i=0; i<nt; i++) {
		x[i]=x[i]/nt;
	}
}

void scale_z(double complex *x, int nt) {
	for(int i=0; i<nt; i++) {
		x[i]=x[i]/nt;
	}
}

void test_nd_fft(int *n, int dim) {
	// read dim and n
	int nt = get_nt(n,dim);
	float complex x_c[nt];
	float complex y_c[nt];
	float complex x_c0[nt];

	Uniform_c(x_c,nt);  // get uniform random complex numbers
	copy_c(y_c,x_c,nt);
	copy_c(x_c0,x_c,nt);
	int mode = 1;  // for forward fft

	// original x_c0
	printf("original x_c0\n");
	printf("%s","x_c0\n");
	testwt_c(x_c0, nt);
	printf("\n");

	// forward FFT
	fftwnd_c(x_c0,dim,n,-mode); // sign in fftw is -mode
	printf("%s","x_c0\n");
	testwt_c(x_c0, nt);
	printf("\n");

	cufftnd_c(x_c, dim, n, mode);
	if(dim==4) {
		cufft4d_c(y_c, n, mode);
	} else if(dim==5) {
		cufft5d_c(y_c, n, mode);
	} else if(dim==6) {
	cufft6d_c(y_c, n, mode);
	}

	// 0 : false 1 : true
	printf("x_c = x_c0 fft? %d\n",check_c(x_c,x_c0,nt));
	printf("y_c = x_c0 fft? %d\n",check_c(y_c,x_c0,nt));

	// inverse FFT
	printf("inverse FFT of x_c0\n");
	fftwnd_c(x_c0,dim,n,mode);  // sign in fftw is -mode
	scale_c(x_c0,nt);
	printf("%s","x_c0\n");
	testwt_c(x_c0, nt);
	printf("\n");

	cufftnd_c(x_c, dim, n, -mode);
	scale_c(x_c,nt);
	printf("inverse FFT of x_c\n");
	testwt_c(x_c, nt);
	printf("\n");

	double complex x_z[nt];
	double complex y_z[nt];
	double complex x_z0[nt];

	Uniform_z(x_z,nt);  // get uniform random complex numbers
	copy_z(y_z,x_z,nt);
	copy_z(x_z0,x_z,nt);

	// original  x_z0
	printf("original x_z0\n");
	printf("%s","x_z0\n");
	testwt_z(x_z0, nt);
	printf("\n");

	// forward FFT
	fftwnd_z(x_z0,dim,n,-mode);
	printf("%s","x_z0\n");
	testwt_z(x_z0, nt);
	printf("\n");

	//testwt_z(x_z, nt);  // for test
	//printf("x_z = y_z after copy? %d\n",check_z(x_z,y_z,nt));
	cufftnd_z(x_z, dim, n, mode);
	if(dim==4) {
		cufft4d_z(y_z, n, mode);
	} else if(dim==5) {
		cufft5d_z(y_z, n, mode);
	} else if(dim==6) {
		cufft6d_z(y_z, n, mode);
	}
	// 0 : false 1 : true
	printf("x_z = x_z0 fft? %d\n",check_z(x_z,x_z0,nt));
	printf("y_z = x_z0 fft? %d\n",check_z(y_z,x_z0,nt));
	//testwt_z(x_z, nt);

	// inverse FFT
	printf("inverse FFT of x_z0\n");
	fftwnd_z(x_z0,dim,n,mode);
	scale_z(x_z0,nt);
	printf("%s","x_z0\n");
	testwt_z(x_z0, nt);
	printf("\n");

	cufftnd_z(x_z, dim, n, -mode);
	scale_z(x_z,nt);
	printf("inverse FFT of x_z\n");
	testwt_z(x_z, nt);
	printf("\n");
}

int main(int argc, char **argv) {
	// read test 4D data here
	//float complex *x;
	int dim = 4;
	int n4[]={4,4,4,4};  // nt=256
	test_nd_fft(n4,dim);

	dim = 6;
	int n6[]={4,4,4,4,4,4};
	test_nd_fft(n6,dim);

	dim = 5;
	int n5[]={4,4,4,4,4};
	test_nd_fft(n5,dim);

}
