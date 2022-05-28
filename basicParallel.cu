#include "basicParallel.h"
using namespace std;

//o(1)
__global__ void normKernel(Point* d_basePointCloud, Point* d_targetPointCloud, float* d_normArray){
    int indexBase = blockIdx.x*blockDim.x+threadIdx.x;
    int indexTarget = blockIdx.y*blockDim.y+threadIdx.y;
    d_normArray[indexBase*CLOUDSIZE+indexTarget] = norm3d(d_basePointCloud[indexBase].x-d_targetPointCloud[indexTarget].x,\
                                                          d_basePointCloud[indexBase].y-d_targetPointCloud[indexTarget].y,\
                                                          d_basePointCloud[indexBase].z-d_targetPointCloud[indexTarget].z);
}

//o(n)
__global__ void minKernel(float* d_normArray, float* d_minArray){
    int indexBase = blockIdx.x*blockDim.x+threadIdx.x;
    float min = 1;
    for (int i = 0; i < CLOUDSIZE; i++)
        if(d_normArray[indexBase*CLOUDSIZE+i]<min) min = d_normArray[indexBase*CLOUDSIZE+i];
    d_minArray[indexBase] = min;
}

//o(logn)
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


void basicParaCompute(Point* basePointcloud, Point* targetPointcloud, float* basicParaRes, int dev){
    cudaSetDevice(dev);
    Point* d_basePointcloud, *d_targetPointcloud;
    float* d_normArray, *d_minArray, *d_basicParaRes, *d_basicParaResList;
    float* normArray, *minArray, *resList;
    int size = CLOUDSIZE*sizeof(Point);

    normArray = (float *)malloc(CLOUDSIZE*CLOUDSIZE*sizeof(float));
    minArray = (float *)malloc(CLOUDSIZE*sizeof(float));
    resList = (float *)malloc((CLOUDSIZE/PARTIAlBLOCKSIZE+1)*sizeof(float));

    //allocate device memory
    //and mv basePointcloud and target Pointcloud to device memory
    cudaMalloc((void **)&d_basePointcloud, size);
    cudaMemcpy(d_basePointcloud, basePointcloud, size, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_targetPointcloud, size);
    cudaMemcpy(d_targetPointcloud, targetPointcloud, size, cudaMemcpyHostToDevice);
    //normArray and result space
    cudaMalloc((void **)&d_normArray, CLOUDSIZE*CLOUDSIZE*sizeof(float));
    cudaMalloc((void **)&d_minArray, CLOUDSIZE*sizeof(float));
    cudaMalloc((void **)&d_basicParaResList, (CLOUDSIZE/PARTIAlBLOCKSIZE+1)*sizeof(float));
    cudaMalloc((void **)&d_basicParaRes, sizeof(float));

    //kernel
    //compute 2-Norm
    dim3 normDimGrid(CLOUDSIZE/16+1,CLOUDSIZE/16+1,1);
    dim3 normDimBlock(16,16,1);
    normKernel <<< normDimGrid, normDimBlock >>> (d_basePointcloud,d_targetPointcloud,d_normArray);
    cudaMemcpy(normArray, d_normArray,CLOUDSIZE*CLOUDSIZE*sizeof(float), cudaMemcpyDeviceToHost);

    //compute min of norm array
    dim3 minDimGrid(CLOUDSIZE/16+1,1,1);
    dim3 minDimBlock(16,1,1);
    minKernel <<< minDimGrid, minDimBlock >>> (d_normArray, d_minArray);
    cudaMemcpy(minArray, d_minArray, CLOUDSIZE*sizeof(float), cudaMemcpyDeviceToHost);

    //compute sum
    dim3 sumDimGrid(CLOUDSIZE/PARTIAlBLOCKSIZE+1,1,1);
    dim3 sumDimBlock(PARTIAlBLOCKSIZE,1,1);
    sumKernel <<< sumDimGrid, sumDimBlock>>> (d_minArray, d_basicParaResList);
    cudaMemcpy(resList, d_basicParaResList, (CLOUDSIZE/PARTIAlBLOCKSIZE+1)*sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < CLOUDSIZE/PARTIAlBLOCKSIZE+1; i++)
    {
        *basicParaRes+=resList[i];
    }
    
    //get result from device
    *basicParaRes /= CLOUDSIZE;
    
    //free memory
    cudaFree(d_basePointcloud);
    cudaFree(d_targetPointcloud);
    cudaFree(d_normArray);
    cudaFree(d_minArray);
    cudaFree(d_basicParaRes);
    free(normArray);
    free(minArray);
}