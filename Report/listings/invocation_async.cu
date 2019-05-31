cudaStream_t stream[nStreams];
for (int i = 0; i < nStreams; ++i)
    cudaStreamCreate(&stream[i]);

for (int i = 0; i < nStreams; ++i)
{
    cudaMemcpyAsync(d_A[i], h_A[i], mem_size_A, cudaMemcpyHostToDevice, stream[i]);
    cudaMemcpyAsync(d_B[i], h_B[i], mem_size_B, cudaMemcpyHostToDevice, stream[i]);
}

for (int i = 0; i < nStreams; ++i)
    matrixMulCUDA<block_size><<<grid, threads, 0, stream[i]>>>(d_C[i], d_A[i], d_B[i], dimsA.x, dimsB.x);

for (int i = 0; i < nStreams; ++i)
    cudaMemcpyAsync(h_C[i], d_C[i], mem_size_C, cudaMemcpyDeviceToHost, stream[i]);