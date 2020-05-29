const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;
const int NUM_REPS = 100;

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result);

// Check errors and print GB/s
void postprocess(const float *ref, const float *res, int n, float ms);

__global__ void copy(float *odata, const float *idata, int nx, int ny);

// naive transpose
// Simplest transpose; doesn't use shared memory.
// Global memory reads are coalesced but writes are not.
__global__ void transposeNaive(float *odata, const float *idata, int nx, int ny);

// coalesced transpose
// Uses shared memory to achieve coalesing in both reads and writes
// Tile width == #banks causes shared memory bank conflicts.
__global__ void transposeCoalesced(float *odata, const float *idata, int nx, int ny);

// No bank-conflict transpose
// Same as transposeCoalesced except the first tile dimension is padded 
// to avoid shared memory bank conflicts.
__global__ void transposeNoBankConflicts(float *odata, const float *idata, int nx, int ny);

