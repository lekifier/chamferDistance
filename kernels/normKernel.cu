__global__ void normKernel(Point* d_basePointCloud, Point* d_targetPointCloud, float* d_normArray){
    int indexBase = blockIdx.x*blockDim.x+threadIdx.x;
    int indexTarget = blockIdx.y*blockDim.y+threadIdx.y;
    d_normArray[indexBase*CLOUDSIZE+indexTarget] = \
    norm3d(d_basePointCloud[indexBase].x-d_targetPointCloud[indexTarget].x,\
           d_basePointCloud[indexBase].y-d_targetPointCloud[indexTarget].y,\
           d_basePointCloud[indexBase].z-d_targetPointCloud[indexTarget].z);
}