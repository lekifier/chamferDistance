#include "serial.h"
float twoNorm(Point point1, Point point2){    
    return sqrt(pow((point1.x - point2.x),2)+\
                pow((point1.y - point2.y),2)+\
                pow((point1.z - point2.z),2));
}
float serialCompute(Point* basePointcloud, Point* targetPointcloud){
    float res = 0;
    for(int i = 0; i < CLOUDSIZE; i++){
        float min = 1;
        for(int j = 0; j < CLOUDSIZE; j++){
            float cur = twoNorm(basePointcloud[i], targetPointcloud[j]);
            if(cur < min) min = cur;
        }
        res += min;
        std::cout<< "Serial compute process: " <<100*i/CLOUDSIZE<< "%" <<"\r"<<std::flush;
    }
    res /= CLOUDSIZE;
    return res;
}