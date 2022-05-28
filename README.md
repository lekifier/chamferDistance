# CUDA practice: Chamfer distance

A little practice of CUDA program used to compute Chamfer distance between 2 pointclouds.

### Prepare work
Make sure ALL the .sh file is executable.
```
chmod +x *.sh
```

### Generate test dataset
init.sh file will make a folder called 
```
./init.sh
```

### Compile source code
```
make
```
if the code need to be deleted:
```
make clean
```
### Test
The test will compare the result of serial, basic parallel and optimal parallel program.

```
./test.sh
```

### Note
It is just a little practice as my homework, maybe I'll make it better later.

I'm very happy to talk about the program :)

Copyright(c) 2022 Lekifier