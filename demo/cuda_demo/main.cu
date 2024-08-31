#include "common.h"
#include <stdio.h>
#include <map>

/*
 * A simple introduction to programming in CUDA. This program prints "Hello
 * World from GPU! from 10 CUDA threads running on the GPU.
 */


#define TENSOR_MAX_DIM 4

enum tensor_type {
    TENSOR_TYPE_f32,
    TENSOR_TYPE_f16,
    TENSOR_TYPE_Q8,
    TENSOR_TYPE_Q4,
};

enum tensor_backend {
    TENSOR_BACKEND_CPU,
    TENSOR_BACKEND_CUDA,
    TENSOR_BACKEND_VULKEN,
};

struct tensor {
    int dims[TENSOR_MAX_DIM];
    tensor_type type;
    tensor_backend backend;
    void* data;
};

struct op_config_cuda {
};


struct cuda_device_info {
    char device_name[128] ;

    int cc;
    float memory_bus_width;
    int max_ith_sm;
    int max_ith_block;

    int l2_cache_size;
    int l1_cache_size;
    int smem_size;

    int mem_clock_rate;
    int gpu_clock_rate;

    void init() {
    }
};


struct cpu_device_info {

};



void op_turing() {

}

// __global__ void mul_mat_f32_simple(const struct *tensor src0, const struct *tensor src1, struct *tensor dst) {
// 
//     
// }
// 
// __global__ void mul_mat_f32_(const struct *tensor src0, const struct *tensor src1, struct *tensor dst) {
//     
// }
// 
// 
// __global__ void mul_mat_f32_cublas(const struct *tensor src0, const struct *tensor src1, struct *tensor dst) {
//     
// }
// 
// 
// void mul_mat_f32_cpu(const struct *tensor src0, const struct *tensor src1, struct *tensor dst) {
// 
// }
// 


void get_cuda_device_info() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        printf("There are no available device(s) that support CUDA\n");
    }
    else
    {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

    int dev = 0, driverVersion = 0, runtimeVersion = 0;
    CHECK_CUDA(cudaSetDevice(dev));
    cudaDeviceProp deviceProp;
    CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Device %d: \"%s\"\n", dev, deviceProp.name);

    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
           driverVersion / 1000, (driverVersion % 100) / 10,
           runtimeVersion / 1000, (runtimeVersion % 100) / 10);
    printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
           deviceProp.major, deviceProp.minor);
    printf("  Total amount of global memory:                 %.2f MBytes (%llu "
           "bytes)\n", (float)deviceProp.totalGlobalMem / pow(1024.0, 3),
           (unsigned long long)deviceProp.totalGlobalMem);
    printf("  GPU Clock rate:                                %.0f MHz (%0.2f "
           "GHz)\n", deviceProp.clockRate * 1e-3f,
           deviceProp.clockRate * 1e-6f);
    printf("  Memory Clock rate:                             %.0f Mhz\n",
           deviceProp.memoryClockRate * 1e-3f);
    printf("  Memory Bus Width:                              %d-bit\n",
           deviceProp.memoryBusWidth);

    if (deviceProp.l2CacheSize)
    {
        printf("  L2 Cache Size:                                 %d bytes\n",
               deviceProp.l2CacheSize);
    }

    printf("  Max Texture Dimension Size (x,y,z)             1D=(%d), "
           "2D=(%d,%d), 3D=(%d,%d,%d)\n", deviceProp.maxTexture1D,
           deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
           deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1],
           deviceProp.maxTexture3D[2]);
    printf("  Max Layered Texture Size (dim) x layers        1D=(%d) x %d, "
           "2D=(%d,%d) x %d\n", deviceProp.maxTexture1DLayered[0],
           deviceProp.maxTexture1DLayered[1], deviceProp.maxTexture2DLayered[0],
           deviceProp.maxTexture2DLayered[1],
           deviceProp.maxTexture2DLayered[2]);
    printf("  Total amount of constant memory:               %lu bytes\n",
           deviceProp.totalConstMem);
    printf("  Total amount of shared memory per block:       %lu bytes\n",
           deviceProp.sharedMemPerBlock);
    printf("  Total number of registers available per block: %d\n",
           deviceProp.regsPerBlock);
    printf("  Warp size:                                     %d\n",
           deviceProp.warpSize);
    printf("  Maximum number of threads per multiprocessor:  %d\n",
           deviceProp.maxThreadsPerMultiProcessor);
    printf("  Maximum number of threads per block:           %d\n",
           deviceProp.maxThreadsPerBlock);


    //printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
    //       deviceProp.maxThreadsDim[0],
    //       deviceProp.maxThreadsDim[1],
    //       deviceProp.maxThreadsDim[2]);
    //printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
    //       deviceProp.maxGridSize[0],
    //       deviceProp.maxGridSize[1],
    //       deviceProp.maxGridSize[2]);
    //printf("  Maximum memory pitch:                          %lu bytes\n",
    //       deviceProp.memPitch);

}

__global__ void helloFromGPU()
{
    printf("Hello World from GPU!\n");
}



struct host_tensor_allocator {
    struct mem_block {
        char* begin;
        size_t size;
        bool used;
    };

    char* data;
    std::map<int, mem_block> dict; // 
    // std::map<int, int> dict; // 

    void init() {

    }

    void* alloc(size_t size) {
        if (dict.size() == 0) {
        }
        void * ptr = calloc(size, sizeof(char));
        return ptr;
    }
};

void test_host_mem_pool() {

    size_t s0 = 151851 * 32;
    size_t s1 = 4096;
    size_t s2 = 128;
    size_t s3 = 4;

    


}


// todo: template
void initial_data(void* ptr, size_t size) {
    float* p = (float*)ptr;
    time_t t;
    srand((unsigned) time(&t));

    for (int i = 0; i < size; ++i) {
        p[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}


void sum_array_f32(struct *tensor src0, struct *tensor src1, struct *tensor dst, size_t size) {
    float* p_src0= (float*)src0->data;
    float* p_src1 = (float*)src1->data;
    float* p_dst = (float*)dst->data;
    for (int idx = 0; idx < size; ++size) {
        p_dst[idx] = p_src0[idx] + p_src1[idx];
    }
}

__global__ sum_array_f32(struct *tensor src0, struct *tensor src1, struct *tensor dst) {
    int idx = threadIdx.x;
    dst[idx] = src0[idx] + src1[idx];
}


float tensor_distance_cos(const struct *tensor src0, const struct *tensor src1) {

}


void test_array_sum() {

    int dev = 0;
    CHECK(cudaSetDevice(dev));

    // set up data size of vectors
    int nElem = 1 << 5;
    printf("Vector size %d\n", nElem);

    // malloc host memory
    size_t nBytes = nElem * sizeof(float);

    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A     = (float *)malloc(nBytes);
    h_B     = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef  = (float *)malloc(nBytes);

    // initialize data at host side
    initial_data(h_A, nElem);
    initial_data(h_B, nElem);

    memset(hostRef, 0, nBytes);
    memset(gpuRef,  0, nBytes);

    // malloc device global memory
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_B, nBytes));
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    // transfer data from host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice));

    // invoke kernel at host side
    dim3 block (nElem);
    dim3 grid  (1);
}







int main(int argc, char **argv)
{
    // printf("Hello World from CPU!\n");
    // helloFromGPU<<<1, 10>>>();
    // CHECK_CUDA(cudaDeviceReset());
    get_cuda_device_info();

    return 0;
}


