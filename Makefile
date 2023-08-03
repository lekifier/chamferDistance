all: analysatorSerial.cpp analysatorBasicParallel.cpp analysatorOptParallel.cpp serial.o basicParallel.o optParallel.o
	nvcc -O2 analysatorSerial.cpp serial.o -o analysatorSerial.o 
	nvcc analysatorBasicParallel.cpp basicParallel.o -o analysatorBasicParallel.o 
	nvcc analysatorOptParallel.cpp optParallel.o -o analysatorOptParallel.o 

serial.o: serial.cpp serial.h
	g++ -O2 -c serial.cpp -o serial.o 

basicParallel.o: basicParallel.cu basicParallel.h
	nvcc -lib -rdc=true  basicParallel.cu -o basicParallel.o

optParallel.o: optParallel.cu optParallel.h
	nvcc -lib -rdc=true  optParallel.cu -o optParallel.o

clean:
	rm *.o
