#include <assert.h>
#include <fstream>
#include <stdio.h>
#include <vector>

#include <cuda_runtime.h>
#include <helper_functions.h>

//--------------------------PARAMETERS---------------------------------
// Compile and run only one version at a time: 3, 5
#define CODE_VERSION 3

std::ofstream file;

const int block_size = 16;   // size of one dimension
const int matrix_size = 864; // size of one dimension
const int nStreams = 10;
const double epsilon = 1e-3; // corectness checking tolerance
//---------------------------------------------------------------------

template <int BLOCK_SIZE>
__global__ void
matrixMulCUDA(float *C, float *A, float *B, int wA, int wB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += aStep, b += bStep)
    {

        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

void constantInit(float *data, int size, float val)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = val;
    }
}

#if CODE_VERSION == 3
/**
 * Version 3
 */
int matrixMultiply(dim3 &dimsA, dim3 &dimsB)
{
    // Allocate host memory for matrices A and B
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    dim3 dimsC(dimsB.x, dimsA.y, 1);
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);

    // Allocate and Initialize host memory
    const float valB = 0.01f;
    std::vector<float *> h_A(nStreams);
    std::vector<float *> h_B(nStreams);
    std::vector<float *> h_C(nStreams);
    for (int i = 0; i < nStreams; i++)
    {
        h_A[i] = (float *)malloc(mem_size_A);
        if (h_A[i] == NULL)
        {
            fprintf(stderr, "Failed to allocate host matrix C!\n");
            exit(EXIT_FAILURE);
        }
        constantInit(h_A[i], size_A, 1.0f);
        h_B[i] = (float *)malloc(mem_size_B);
        if (h_B[i] == NULL)
        {
            fprintf(stderr, "Failed to allocate host matrix C!\n");
            exit(EXIT_FAILURE);
        }
        constantInit(h_B[i], size_B, valB);
        h_C[i] = (float *)malloc(mem_size_B); // no need to fill with values
        if (h_C[i] == NULL)
        {
            fprintf(stderr, "Failed to allocate host matrix C!\n");
            exit(EXIT_FAILURE);
        }
    }

    cudaError_t error;

    // Allocate device memory
    std::vector<float *> d_A(nStreams);
    std::vector<float *> d_B(nStreams);
    std::vector<float *> d_C(nStreams);

    for (int i = 0; i < nStreams; i++)
    {
        error = cudaMalloc((void **)&d_A[i], mem_size_A);
        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }

        error = cudaMalloc((void **)&d_B[i], mem_size_B);
        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_B returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }

        error = cudaMalloc((void **)&d_C[i], mem_size_C);
        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_C returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }
    }

    // Setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);
    assert(("Wrong matrix size", grid.y * threads.y == dimsA.y));

    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start;
    error = cudaEventCreate(&start);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    cudaEvent_t stop;
    error = cudaEventCreate(&stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Record the start event
    error = cudaEventRecord(start, NULL);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Process all matrices
    for (int i = 0; i < nStreams; i++)
    {
        // copy host memory to device
        error = cudaMemcpy(d_A[i], h_A[i], mem_size_A, cudaMemcpyHostToDevice);

        if (error != cudaSuccess)
        {
            printf("cudaMemcpy (d_A,h_A) returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }

        error = cudaMemcpy(d_B[i], h_B[i], mem_size_B, cudaMemcpyHostToDevice);

        if (error != cudaSuccess)
        {
            printf("cudaMemcpy (d_B,h_B) returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }

        // Execute the kernel
        matrixMulCUDA<block_size><<<grid, threads>>>(d_C[i], d_A[i], d_B[i], dimsA.x, dimsB.x);

        // Copy result from device to host
        error = cudaMemcpy(h_C[i], d_C[i], mem_size_C, cudaMemcpyDeviceToHost);

        if (error != cudaSuccess)
        {
            printf("cudaMemcpy (h_C,d_C) returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }
    }

    // Record the stop event
    error = cudaEventRecord(stop, NULL);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Wait for the stop event to complete
    error = cudaEventSynchronize(stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    float msecTotal = 0.0f;
    error = cudaEventElapsedTime(&msecTotal, start, stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal;
    double flopsPerMatrixMul = nStreams * 2.0 * (double)dimsA.x * (double)dimsA.y * (double)dimsB.x;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf(
        "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops, WorkgroupSize= %u threads/block\n",
        gigaFlops,
        msecPerMatrixMul,
        flopsPerMatrixMul,
        threads.x * threads.y);

    file << "Performance: " << gigaFlops << " GFlop/s" << std::endl;
    file << "Time: " << msecPerMatrixMul << " msec" << std::endl;
    file << "Size: " << flopsPerMatrixMul << " Ops" << std::endl;

    printf("Checking computed result for correctness: ");
    bool correct = true;

    for (int idx = 0; idx < nStreams; idx++)
    {
        for (int i = 0; i < (int)(dimsC.x * dimsC.y); i++)
        {
            if (fabs(h_C[idx][i] - (dimsA.x * valB)) > epsilon)
            {
                printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %.2e\n", i, h_C[idx][i], dimsA.x * valB, epsilon);
                correct = false;
            }
        }
    }

    printf("%s\n", correct ? "OK" : "FAIL");

    // Clean up memory
    for (int i = 0; i < nStreams; i++)
    {
        free(h_A[i]);
        free(h_B[i]);
        free(h_C[i]);
        cudaFree(d_A[i]);
        cudaFree(d_B[i]);
        cudaFree(d_C[i]);
    }

    cudaDeviceReset();

    if (correct)
    {
        return EXIT_SUCCESS;
    }
    else
    {
        return EXIT_FAILURE;
    }
}
#endif // CODE_VERSION == 3

#if CODE_VERSION == 5
/**
 * Version 5
 */
int asyncMatrixMultiply(dim3 &dimsA, dim3 &dimsB)
{
    // Allocate host memory for matrices A and B
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    dim3 dimsC(dimsB.x, dimsA.y, 1);
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);

    cudaError_t error;

    // Allocate and Initialize pinned host memory
    const float valB = 0.01f;
    std::vector<float *> h_A(nStreams);
    std::vector<float *> h_B(nStreams);
    std::vector<float *> h_C(nStreams);
    for (int i = 0; i < nStreams; i++)
    {
        error = cudaMallocHost((void **)&h_A[i], mem_size_A);
        if (error != cudaSuccess)
        {
            printf("cudaMallocHost h_A returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }
        constantInit(h_A[i], size_A, 1.0f);

        error = cudaMallocHost((void **)&h_B[i], mem_size_A);
        if (error != cudaSuccess)
        {
            printf("cudaMallocHost h_B returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }
        constantInit(h_B[i], size_B, valB);

        error = cudaMallocHost((void **)&h_C[i], mem_size_A); // no need to fill with values
        if (error != cudaSuccess)
        {
            printf("cudaMallocHost h_C returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }
    }

    // Allocate device memory
    std::vector<float *> d_A(nStreams);
    std::vector<float *> d_B(nStreams);
    std::vector<float *> d_C(nStreams);

    for (int i = 0; i < nStreams; i++)
    {
        error = cudaMalloc((void **)&d_A[i], mem_size_A);
        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }

        error = cudaMalloc((void **)&d_B[i], mem_size_B);
        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_B returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }

        error = cudaMalloc((void **)&d_C[i], mem_size_C);
        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_C returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }
    }

    // Setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);
    assert(("Wrong matrix size", grid.y * threads.y == dimsA.y));

    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start;
    error = cudaEventCreate(&start);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    cudaEvent_t stop;
    error = cudaEventCreate(&stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Create streams
    cudaStream_t stream[nStreams];
    for (int i = 0; i < nStreams; ++i)
    {
        error = cudaStreamCreate(&stream[i]);
        if (error != cudaSuccess)
        {
            fprintf(stderr, "Failed to create stream (error code %s)!\n", cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }
    }

    // Record the start event
    error = cudaEventRecord(start, NULL);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Process all matrices
    {
        // copy host memory to device
        for (int i = 0; i < nStreams; ++i)
        {
            error = cudaMemcpyAsync(d_A[i], h_A[i], mem_size_A, cudaMemcpyHostToDevice, stream[i]);
            if (error != cudaSuccess)
            {
                printf("cudaMemcpyAsync (d_A,h_A) returned error code %d, line(%d)\n", error, __LINE__);
                exit(EXIT_FAILURE);
            }
            error = cudaMemcpyAsync(d_B[i], h_B[i], mem_size_B, cudaMemcpyHostToDevice, stream[i]);
            if (error != cudaSuccess)
            {
                printf("cudaMemcpyAsync (d_B,h_B) returned error code %d, line(%d)\n", error, __LINE__);
                exit(EXIT_FAILURE);
            }
        }

        // Execute the kernel
        for (int i = 0; i < nStreams; ++i)
        {
            matrixMulCUDA<block_size><<<grid, threads, 0, stream[i]>>>(d_C[i], d_A[i], d_B[i], dimsA.x, dimsB.x);
        }

        // Copy result from device to host
        for (int i = 0; i < nStreams; ++i)
        {
            error = cudaMemcpyAsync(h_C[i], d_C[i], mem_size_C, cudaMemcpyDeviceToHost, stream[i]);

            if (error != cudaSuccess)
            {
                printf("cudaMemcpyAsync (h_C,d_C) returned error code %d, line(%d)\n", error, __LINE__);
                exit(EXIT_FAILURE);
            }
        }
    }

    // Record the stop event
    error = cudaEventRecord(stop, NULL);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Wait for the stop event to complete
    error = cudaEventSynchronize(stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    float msecTotal = 0.0f;
    error = cudaEventElapsedTime(&msecTotal, start, stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal;
    double flopsPerMatrixMul = nStreams * 2.0 * (double)dimsA.x * (double)dimsA.y * (double)dimsB.x;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf(
        "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops, WorkgroupSize= %u threads/block\n",
        gigaFlops,
        msecPerMatrixMul,
        flopsPerMatrixMul,
        threads.x * threads.y);

    file << "Performance: " << gigaFlops << " GFlop/s" << std::endl;
    file << "Time: " << msecPerMatrixMul << " msec" << std::endl;
    file << "Size: " << flopsPerMatrixMul << " Ops" << std::endl;

    printf("Checking computed result for correctness: ");
    bool correct = true;

    for (int idx = 0; idx < nStreams; idx++)
    {
        for (int i = 0; i < (int)(dimsC.x * dimsC.y); i++)
        {
            if (fabs(h_C[idx][i] - (dimsA.x * valB)) > epsilon)
            {
                printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %.2e\n", i, h_C[idx][i], dimsA.x * valB, epsilon);
                correct = false;
            }
        }
    }

    printf("%s\n", correct ? "OK" : "FAIL");

    // Clean up memory
    for (int i = 0; i < nStreams; i++)
    {
        cudaFreeHost(h_A[i]);
        cudaFreeHost(h_B[i]);
        cudaFreeHost(h_C[i]);
        cudaFree(d_A[i]);
        cudaFree(d_B[i]);
        cudaFree(d_C[i]);
    }

    cudaDeviceReset();

    if (correct)
    {
        return EXIT_SUCCESS;
    }
    else
    {
        return EXIT_FAILURE;
    }
}
#endif // CODE_VERSION == 5

int main()
{
    file.open("resultsGPU.txt", std::ofstream::out | std::ofstream::app);
    printf("[Matrix Multiply Using CUDA] - Starting...\n");

    // By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
    int devID = 0;

    cudaError_t error;
    cudaDeviceProp deviceProp;
    error = cudaGetDevice(&devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
    }

    error = cudaGetDeviceProperties(&deviceProp, devID);

    if (deviceProp.computeMode == cudaComputeModeProhibited)
    {
        fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
        exit(EXIT_SUCCESS);
    }

    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
    }
    else
    {
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
    }

    dim3 dimsA(matrix_size, matrix_size, 1);
    dim3 dimsB(matrix_size, matrix_size, 1);

    if (dimsA.x != dimsB.y)
    {
        printf("Error: outer matrix dimensions must be equal. (%d != %d)\n",
               dimsA.x, dimsB.y);
        exit(EXIT_FAILURE);
    }

    printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y);

    file << "--------------------------START--------------------------------\n";
    file << "Matrix size: " << matrix_size << std::endl
         << "Block size: " << block_size << std::endl
         << "Code version: " << CODE_VERSION << std::endl
         << "nMatrices: " << nStreams << std::endl;

#if CODE_VERSION == 3
    int matrix_result = matrixMultiply(dimsA, dimsB);
#elif CODE_VERSION == 5
    int matrix_result = asyncMatrixMultiply(dimsA, dimsB);
#endif

    file << std::endl;
    if (matrix_result == EXIT_SUCCESS)
    {
        file << "Status: SUCCESS" << std::endl;
    }
    else
    {
        file << "Status: FAILURE" << std::endl;
    }

    file << "---------------------------END---------------------------------\n";
    file.close();
    exit(matrix_result);
}
