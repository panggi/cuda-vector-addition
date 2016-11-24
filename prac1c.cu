//
// include files
//

#include <stdio.h>
#include <cuda_runtime.h>
#include "cutil_inline.h"
#include <cuda_runtime_api.h>

//
// kernel routine
//

__global__ void my_first_kernel (float* x, float* y, float* z)
{
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        printf("%d\n", tid);
        z[tid] = x[tid] + y[tid];
}

//
// main code
//

int main(int argc, char **argv)
{
    //Initialising inputs
    float* h_x;
    float* h_y;
    float* h_z;
    float* d_x;
    float* d_y;
    float* d_z;
    int nblocks, nthreads, nsize;

    // initialise card
    cutilDeviceInit(argc, argv);

    // set number of blocks, and threads per block

    nblocks = 2;
    nthreads = 8;
    nsize = nblocks*nthreads;

    //Allocating memory on the host
    h_x = (float*)malloc(nsize*sizeof(float));
    h_y = (float*)malloc(nsize*sizeof(float));
    h_z = (float*)malloc(nsize*sizeof(float));

    for (int i = 0; i < nsize; ++i)
        {
            h_x[i] = (float)i;
            h_y[i] = (float)i;
            h_z[i] = 0.0;
        }

    // allocate memory
    cudaSafeCall(cudaMalloc( (void**)&d_x, nsize*sizeof(float) ));
    cudaSafeCall(cudaMalloc( (void**)&d_y, nsize*sizeof(float) ));
    cudaSafeCall(cudaMalloc( (void**)&d_z, nsize*sizeof(float) ));

    // copy data from host to device
    cudaSafeCall(cudaMemcpy(d_x, h_x, nsize*sizeof(float), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(d_y, h_y, nsize*sizeof(float), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(d_z, h_z, nsize*sizeof(float), cudaMemcpyHostToDevice));

    // execute kernel
    my_first_kernel<<<nblocks,nthreads>>>(d_x, d_y, d_z);
    cudaCheckMsg("my_first_kernel execution failed\n");
    cudaThreadSynchronize();

    // copy back results and print them out
    cudaSafeCall(cudaMemcpy(h_z, d_z, nsize*sizeof(float), cudaMemcpyDeviceToHost));
    for(int n = 0; n < nsize; n++)
        {
            printf("n, x = %d  %f\n",n,h_z[n]);
        }

    // free memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    free(h_x);
    free(h_y);
    free(h_z);
	
    // CUDA exit -- needed to flush printf write buffer

    cudaDeviceReset();

    return 0;
}