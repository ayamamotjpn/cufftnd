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

#include <fftw3.h>

int getnt(int *n, int rank) {
	int nt=1;
	for(int i=0; i<rank; i++) {
		nt *= n[i];
	}
	printf("nt %d\n",nt);
	return nt;
}

void copy_fftwc(fftwf_complex *x, fftwf_complex *y,int nt) {
	for(int i=0; i<nt; i++) {
		x[i][0] = y[i][0];
		x[i][1] = y[i][1];
	}
}

void copy_fftwz(fftw_complex *x, fftw_complex *y,int nt) {
	for(int i=0; i<nt; i++) {
		x[i][0] = y[i][0];
		x[i][1] = y[i][1];
	}
}

void fftwnd_c(fftwf_complex *x,int rank,int *n,int sign) {
	int nt = getnt(n, rank);
	fftwf_complex *y;
	y = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex) * nt);
	fftwf_plan p;
	p = fftwf_plan_dft(rank,n,x,y,sign,FFTW_ESTIMATE);
	fftwf_execute(p);
	copy_fftwc(x,y,nt);
	//fftwf_free(y);
}

void fftwnd_z(fftw_complex *x,int rank,int *n,int sign) {
	int nt = getnt(n, rank);
	fftw_complex *y;
	y = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * nt);
	fftw_plan p;
	p = fftw_plan_dft(rank,n,x,y,sign,FFTW_ESTIMATE);
	fftw_execute(p);
	copy_fftwz(x,y,nt);
	//fftw_free(y);
}

