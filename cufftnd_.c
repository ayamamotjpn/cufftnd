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


// fortran wrapper of cufftnd.cu written in cuda
// but this can be replaced by new interface
// using isoc_binding

#include "cufftnd.h"

void cufftnd_c_(float complex *x, int *dim, int n[], int *mode) {
	int dim_ = *dim;
	int mode_ = *mode;
	cufftnd_c(x, dim_, n, mode_);
}

void cufftnd_z_(double complex *x, int *dim, int n[], int *mode) {
	int dim_ = *dim;
	int mode_ = *mode;
	cufftnd_z(x, dim_, n, mode_);
}

void cufft2md_c_(float complex *x, int *dim, int n[], int *mode) {
	int dim_ = *dim;
	int mode_ = *mode;
	cufft2md_c(x, dim_, n, mode_);
}

void cufft2md_z_(double complex *x, int *dim, int n[], int *mode) {
	int dim_ = *dim;
	int mode_ = *mode;
	cufft2md_z(x, dim_, n, mode_);
}

void cufft3md_c_(float complex *x, int *dim, int n[], int *mode) {
	int dim_ = *dim;
	int mode_ = *mode;
	cufft3md_c(x, dim_, n, mode_);
}

void cufft3md_z_(double complex *x, int *dim, int n[], int *mode) {
	int dim_ = *dim;
	int mode_ = *mode;
	cufft3md_z(x, dim_, n, mode_);
}


void cufft4d_c_(float complex *x, int n[], int *mode) {
	int mode_ = *mode;
	cufft4d_c(x, n, mode_);
}

void cufft4d_z_(double complex *x, int n[], int *mode) {
	int mode_ = *mode;
	cufft4d_z(x, n, mode_);
}

void cufft5d_c_(float complex *x, int n[], int *mode) {
	int mode_ = *mode;
	cufft5d_c(x, n, mode_);
}

void cufft5d_z_(double complex *x, int n[], int *mode) {
	int mode_ = *mode;
	cufft5d_z(x, n, mode_);
}

void cufft6d_c_(float complex *x, int n[], int *mode) {
	int mode_ = *mode;
	cufft6d_c(x, n, mode_);
}

void cufft6d_z_(double complex *x, int n[], int *mode) {
	int mode_ = *mode;
	cufft6d_z(x, n, mode_);
}


