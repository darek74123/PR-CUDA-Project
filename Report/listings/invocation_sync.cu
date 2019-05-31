for (int i = 0; i < nStreams; i++)
{
    cudaMemcpy(d_A[i], h_A[i], mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B[i], h_B[i], mem_size_B, cudaMemcpyHostToDevice);

    matrixMulCUDA<block_size><<<grid, threads>>>(d_C[i], d_A[i], d_B[i], dimsA.x, dimsB.x);
    
    cudaMemcpy(h_C[i], d_C[i], mem_size_C, cudaMemcpyDeviceToHost);
}