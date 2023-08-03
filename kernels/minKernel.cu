__global__ void minKernel(float* d_normArray, float* d_minArray){
    int indexBase = blockIdx.x*blockDim.x+threadIdx.x;
    float min = 1;
    for (int i = 0; i < CLOUDSIZE; i++)
        if(d_normArray[indexBase*CLOUDSIZE+i]<min) min = d_normArray[indexBase*CLOUDSIZE+i];
    d_minArray[indexBase] = min;
}