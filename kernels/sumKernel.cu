__global__ void sumKernel(float* d_minArray, float* d_basicParaResList){
    __shared__ float partialSum[2*PARTIAlBLOCKSIZE];
    unsigned int t = threadIdx.x;
    unsigned int start = 2*blockIdx.x*blockDim.x;
    //loading data from global memory to share memory
    partialSum[t] = d_minArray[start+t];
    partialSum[blockDim.x+t] = d_minArray[start + blockDim.x+t];
    //compute
    for(unsigned int stride = 1; stride <= blockDim.x; stride <<= 1){
        __syncthreads();
        //if t % stride == 0 the thread will run the computation
        if(t % stride == 0)
            partialSum[2*t] += partialSum[2*t+stride];
    } 
    if(t==0) d_basicParaResList[blockIdx.x] = partialSum[0];
}