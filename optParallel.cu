#include "optParallel.h"
#include <iostream>
using namespace std;
//o(1)
__global__ void optNormKernel(Point* d_basePointCloud, Point* d_targetPointCloud, float* d_normArray){
    int indexBase = blockIdx.x*blockDim.x+threadIdx.x;
    int indexTarget = blockIdx.y*blockDim.y+threadIdx.y;
    d_normArray[indexBase*CLOUDSIZE+indexTarget] =\
    norm3d(d_basePointCloud[indexBase].x-d_targetPointCloud[indexTarget].x,\
           d_basePointCloud[indexBase].y-d_targetPointCloud[indexTarget].y,\
           d_basePointCloud[indexBase].z-d_targetPointCloud[indexTarget].z);
}
//o(logn)
__global__ void optMinKernel(float* d_normArray, float* d_minArray){
    //shared memeory for partial min of every thread in the same 2D block
    __shared__ float partialMin[PARTIAlBLOCKSIZE][2*PARTIAlBLOCKSIZE];
    unsigned int rowIndex = blockIdx.x*blockDim.x+threadIdx.x;

    unsigned int t = threadIdx.y;
    unsigned int start = 2*blockIdx.y*blockDim.y;
    //loading data from global memory to share memory
    partialMin[threadIdx.x][t] = d_normArray[rowIndex*CLOUDSIZE + start+t];
    partialMin[threadIdx.x][blockDim.y+t] = d_normArray[rowIndex*CLOUDSIZE + start+blockDim.y+t];
    //compute
    for(unsigned int stride = blockDim.y; stride > 0; stride >>= 1){
        __syncthreads();
        //if t % stride == 0 the thread will run the computation
        if(t < stride &&partialMin[threadIdx.x][t] > partialMin[threadIdx.x][t+stride])
            partialMin[threadIdx.x][t] = partialMin[threadIdx.x][t+stride];
    } 
    if(t==0) d_minArray[rowIndex*(CLOUDSIZE/PARTIAlBLOCKSIZE+1)+blockIdx.y] = partialMin[threadIdx.x][0];
}
//o(1)
__global__ void optFixMinKernel(float* d_minArray, float* d_minFixMinArray){
    int RowIndex = blockIdx.x*blockDim.x+threadIdx.x;
    d_minFixMinArray[RowIndex]=1;
    for (int i = 0; i <(CLOUDSIZE/PARTIAlBLOCKSIZE+1); i++){
        if(d_minArray[RowIndex*(CLOUDSIZE/PARTIAlBLOCKSIZE+1)+i] < d_minFixMinArray[RowIndex])
            d_minFixMinArray[RowIndex] = d_minArray[RowIndex*(CLOUDSIZE/PARTIAlBLOCKSIZE+1)+i];
    }
}

//o(logn)
__global__ void optSumKernel(float* d_minArray, float* d_basicParaResList){
    __shared__ float partialSum[2*PARTIAlBLOCKSIZE];
    unsigned int t = threadIdx.x;
    unsigned int start = 2*blockIdx.x*blockDim.x;
    //loading data from global memory to share memory
    partialSum[t] = d_minArray[start+t];
    partialSum[blockDim.x+t] = d_minArray[start + blockDim.x+t];
    //compute
    for(unsigned int stride = blockDim.x; stride > 0; stride >>= 1){
        __syncthreads();
        if(t < stride)
            partialSum[t] += partialSum[t+stride];
    } 
    if(t==0) d_basicParaResList[blockIdx.x] = partialSum[0];
}


void optParaCompute(Point* basePointcloud, Point* targetPointcloud, float* basicParaRes, int dev){
    cudaSetDevice(dev);
    Point* d_basePointcloud, *d_targetPointcloud;
    float* d_normArray, *d_minArray, *d_fixMinArray, *d_basicParaRes, *d_basicParaResList;
    float* normArray, *minArray, *fixMinArray, *resList;
    int size = CLOUDSIZE*sizeof(Point);

    normArray = (float *)malloc(CLOUDSIZE*CLOUDSIZE*sizeof(float));
    minArray = (float *)malloc(CLOUDSIZE*(CLOUDSIZE/PARTIAlBLOCKSIZE+1)*sizeof(float));
    fixMinArray = (float *)malloc(CLOUDSIZE*sizeof(float));
    resList = (float *)malloc((CLOUDSIZE/PARTIAlBLOCKSIZE+1)*sizeof(float));

    //allocate device memory
    //and mv basePointcloud and target Pointcloud to device memory
    cudaMalloc((void **)&d_basePointcloud, size);
    cudaMemcpy(d_basePointcloud, basePointcloud, size, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_targetPointcloud, size);
    cudaMemcpy(d_targetPointcloud, targetPointcloud, size, cudaMemcpyHostToDevice);
    //normArray and result space
    cudaMalloc((void **)&d_normArray, CLOUDSIZE*CLOUDSIZE*sizeof(float));
    cudaMalloc((void **)&d_minArray, CLOUDSIZE*(CLOUDSIZE/PARTIAlBLOCKSIZE+1)*sizeof(float));
    cudaMalloc((void **)&d_fixMinArray, CLOUDSIZE*sizeof(float));
    cudaMalloc((void **)&d_basicParaResList, (CLOUDSIZE/PARTIAlBLOCKSIZE+1)*sizeof(float));
    cudaMalloc((void **)&d_basicParaRes, sizeof(float));

    //kernel
    //compute 2-Norm
    dim3 normDimBlock(PARTIAlBLOCKSIZE,PARTIAlBLOCKSIZE);
    dim3 normDimGrid((CLOUDSIZE + normDimBlock.x - 1) / normDimBlock.x, (CLOUDSIZE + normDimBlock.y - 1) / normDimBlock.y);
    optNormKernel <<< normDimGrid, normDimBlock >>> (d_basePointcloud,d_targetPointcloud,d_normArray);
    cudaMemcpy(normArray, d_normArray,CLOUDSIZE*CLOUDSIZE*sizeof(float), cudaMemcpyDeviceToHost);
    for(int i=0; i<40; i++) cout << normArray[i]<<" ";
    cout << endl;
    //compute min of norm array
    dim3 minDimBlock(PARTIAlBLOCKSIZE,PARTIAlBLOCKSIZE);
    dim3 minDimGrid((CLOUDSIZE + minDimBlock.x - 1) / minDimBlock.x);
    optMinKernel <<< minDimGrid, minDimBlock >>> (d_normArray, d_minArray);
    cudaMemcpy(minArray, d_minArray, CLOUDSIZE*(CLOUDSIZE/PARTIAlBLOCKSIZE+1)*sizeof(float), cudaMemcpyDeviceToHost);
    for(int i=0; i<40; i++) cout << minArray[i]<<" ";
    cout << endl;
    //fix min of norm array
    dim3 fixMinDimBlock(PARTIAlBLOCKSIZE);
    dim3 fixMinDimGrid((CLOUDSIZE+fixMinDimBlock.x-1)/fixMinDimBlock.x);
    optFixMinKernel <<< fixMinDimGrid, fixMinDimBlock >>> (d_minArray, d_fixMinArray);
    cudaMemcpy(fixMinArray, d_fixMinArray, CLOUDSIZE*sizeof(float), cudaMemcpyDeviceToHost);

    //compute sum
    dim3 sumDimBlock(PARTIAlBLOCKSIZE);
    dim3 sumDimGrid((CLOUDSIZE+sumDimBlock.x-1)/sumDimBlock.x);
    optSumKernel <<< sumDimGrid, sumDimBlock>>> (d_fixMinArray, d_basicParaResList);
    cudaMemcpy(resList, d_basicParaResList, (CLOUDSIZE/PARTIAlBLOCKSIZE+1)*sizeof(float), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < CLOUDSIZE/PARTIAlBLOCKSIZE+1; i++) *basicParaRes+=resList[i];
    
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