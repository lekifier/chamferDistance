import numpy as np
from tqdm import tqdm

targetPath = "./dataset/targetPointcloud.xyz"
basePath = "./dataset/basePointcloud.xyz"
targetData = []
baseData = []
count = 400000
for i in tqdm(range(count)):
    vec = np.random.rand(3)
    targetData.append(vec.tolist())
for i in tqdm(range(count)):
    vec = np.random.rand(3)
    baseData.append(vec.tolist())
with open(targetPath,"w") as f:
    for line in targetData:
        f.write(str(line[0])+" "+str(line[1])+" "+str(line[2])+"\n")
with open(basePath,"w") as f:
    for line in baseData:
        f.write(str(line[0])+" "+str(line[1])+" "+str(line[2])+"\n")