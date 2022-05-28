#include "utils.h"
#include "optParallel.h"
using namespace std;

int main(void){
    string basePath = "./dataset/basePointcloud.xyz";
    string targetPath = "./dataset/targetPointcloud.xyz";
    Point basePointCloud[CLOUDSIZE];
    Point targetPointCloud[CLOUDSIZE];
    ifstream baseFile(basePath);
    ifstream targetFile(targetPath);
    clock_t start,end;
    int dev = 3;

    float optParaRes;

    int count = 0;
    while (count<CLOUDSIZE)
    {
        baseFile >> basePointCloud[count].x \
                 >> basePointCloud[count].y \
                 >> basePointCloud[count].z;
        
        targetFile >> targetPointCloud[count].x \
                   >> targetPointCloud[count].y \
                   >> targetPointCloud[count].z;
        count ++;
    }
    baseFile.close();
    targetFile.close();
    
    start = clock();
    optParaCompute(basePointCloud, targetPointCloud, &optParaRes, dev);
    end = clock();
    cout << "Optimal parallel program result: " << optParaRes << endl;
    cout << "Optimal parallel program costs " << (end - start)*1000/float(CLOCKS_PER_SEC) << " ms" << endl;

    return 0;
}