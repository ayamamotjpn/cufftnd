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


#include <stdio.h>
#include <assert.h>
#include "mtrns_gpu2.h"

// Check errors and print GB/s
void postprocess(const float *ref, const float *res, int n, float ms)
{
  bool passed = true;
  for (int i = 0; i < n; i++)
    if (res[i] != ref[i]) {
      printf("%d %f %f\n", i, res[i], ref[i]);
      printf("%25s\n", "*** FAILED ***");
      passed = false;
      break;
    }
  if (passed)
    printf("%20.2f\n", 2 * n * sizeof(float) * 1e-6 * NUM_REPS / ms );
}

// simple copy kernel
// Used as reference case representing best effective bandwidth.
__global__ void copy(float *odata, const float *idata, int nx, int ny)
{
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  //int width = gridDim.x * TILE_DIM;
  int width = nx;

  for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS) {
    if( x < nx && (y+j) < ny) {
      odata[(y+j)*width + x] = idata[(y+j)*width + x];
    }
  }
}

// naive transpose
// Simplest transpose; doesn't use shared memory.
// Global memory reads are coalesced but writes are not.
__global__ void transposeNaive(float *odata, const float *idata, int nx, int ny)
{
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  //int width = gridDim.x * TILE_DIM;
  int width = nx;

  for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS) {
    if( x < nx && (y+j) < ny) {
      odata[x*width + (y+j)] = idata[(y+j)*width + x];
    }
  }
}

// coalesced transpose
// Uses shared memory to achieve coalesing in both reads and writes
// Tile width == #banks causes shared memory bank conflicts.
__global__ void transposeCoalesced(float *odata, const float *idata, int nx, int ny)
{
  __shared__ float tile[TILE_DIM][TILE_DIM];
    
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  //int width = gridDim.x * TILE_DIM;
  int width = nx;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    if( x < nx && (y+j) < ny) {
      tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];
    }
  }

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;
  int x0 = blockIdx.x * TILE_DIM + threadIdx.x;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    if( x0 < nx && (y+j) < ny) {
      odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
    }
  }
}
   


// No bank-conflict transpose
// Same as transposeCoalesced except the first tile dimension is padded 
// to avoid shared memory bank conflicts.

__global__ void transposeNoBankConflicts(float *odata, const float *idata, int nx, int ny)
{
  __shared__ float tile[TILE_DIM][TILE_DIM+1];

  // blockIdx.x, blockIdx.y are the block indices
  // threadidx.x and threadidx.y run between 0 and blockIdx.x-1 or blockIdx.y-1
  // blockIdx.x * TILE_DIM : offset for the block = tile index
  // blockIdx.y * TILE_DIM : offset for the block != tile index
  // in one tile, there are 32/8=4 blocks along y-direction
    
  int xt = blockIdx.x * TILE_DIM;
  int yt = blockIdx.y * TILE_DIM;
  int x = xt + threadIdx.x;
  int y = yt + threadIdx.y;

  // in the transposed matrix tile indices are swapped
  // and indices in a tile are swapped

  //int width = gridDim.x * TILE_DIM;  //????
  int width = nx;

  // copy a tile data
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
  	if( x < nx && (y+j) < ny) {
    	tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];
  	}
  }

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;
  int x0 = blockIdx.x * TILE_DIM + threadIdx.x;
 
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
  	if( x0 < nx && (y+j) < ny) {
    	odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
    }
  }
}
