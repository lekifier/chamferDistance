#include "utils.h"
#include "basicParallel.h"
using namespace std;

int main(){
    string basePath = "./dataset/basePointcloud.xyz";
    string targetPath = "./dataset/targetPointcloud.xyz";
    Point basePointCloud[CLOUDSIZE];
    Point targetPointCloud[CLOUDSIZE];
    ifstream baseFile(basePath);
    ifstream targetFile(targetPath);
    clock_t start,end;
    int dev = 1;

    float serialRes;
    float basicParaRes;
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
    basicParaCompute(basePointCloud, targetPointCloud, &basicParaRes, dev);
    end = clock();
    cout << "Basic parallel program result: " << basicParaRes << endl;
    cout << "Basic parallel program costs " << (end - start)*1000/float(CLOCKS_PER_SEC) << " ms" << endl;

    return 0;
}