OPT = -O0

CUDA_CC = nvcc $(OPT) -G -g -std=c++11
CC = g++ -std=c++11 -g -Wall -Wfatal-errors -m64 $(OPT)

OPENMP = -fopenmp
# OPENMP = 

NTL = -Intl -lntl -lgmp

#LCUDA = -L/usr/local/cuda/lib64
LCUDA = -lcuda -lcudart -lcudadevrt -L/usr/local/cuda/lib64 -lcurand
ICUDA = -I/usr/local/cuda/include
CUDA_ARCH = -arch=sm_35

SRC = $(PWD)/src
BIN = $(PWD)/bin
OBJ = $(PWD)/obj

all: tests

tests: test.o operators.o polynomial.o cuda_bn.o cuda_distribution.o distribution.o logging.o cuda_bn.o yashe.o
	$(CUDA_CC) $(CUDA_ARCH) $(LCUDA) $(ICUDA) -o $(BIN)/test $(OBJ)/test.o $(OBJ)/polynomial.o $(OBJ)/operators.o $(OBJ)/cuda_distribution.o $(OBJ)/cuda_bn.o $(OBJ)/distribution.o $(OBJ)/logging.o $(OBJ)/log.o $(OBJ)/yashe.o -lcufft -lcurand  --relocatable-device-code true -Xcompiler $(OPENMP) $(NTL) -lboost_unit_test_framework

test.o: $(SRC)/test/test.cpp
	$(CC) -c $(SRC)/test/test.cpp -o $(OBJ)/test.o $(NTL) $(OPENMP) -lcurand  $(LCUDA) $(ICUDA)

operators.o:$(SRC)/cuda/operators.cu
	$(CUDA_CC) $(CUDA_ARCH) -c $(SRC)/cuda/operators.cu -o $(OBJ)/operators.o $(LCUDA) $(ICUDA) -lcufft --relocatable-device-code true -Xcompiler $(OPENMP)

polynomial.o:$(SRC)/aritmetic/polynomial.cu
	$(CUDA_CC) $(CUDA_ARCH) -c $(SRC)/aritmetic/polynomial.cu -o $(OBJ)/polynomial.o $(LCUDA) $(ICUDA) -lcufft --relocatable-device-code true -Xcompiler $(OPENMP) $(NTL)

logging.o: $(SRC)/logging/logging.cpp
	$(CC) -c $(SRC)/logging/log.c -o $(OBJ)/log.o
	$(CC) -c -w $(SRC)/logging/logging.cpp -o $(OBJ)/logging.o

# cuda_bn.o:$(SRC)/cuda/cuda_bn.cu
# 	$(CUDA_CC) $(CUDA_ARCH) -c $(SRC)/cuda/cuda_bn.cu -o $(OBJ)/cuda_bn.o (LCUDA) $(ICUDA) --relocatable-device-code true -Xcompiler $(NTL)

cuda_bn.o:$(SRC)/cuda/cuda_bn.cu
	$(CUDA_CC) $(CUDA_ARCH) -c $(SRC)/cuda/cuda_bn.cu -o $(OBJ)/cuda_bn.o $(LCUDA) $(ICUDA) --relocatable-device-code true -Xcompiler $(NTL)

cuda_distribution.o:$(SRC)/cuda/cuda_distribution.cu
	$(CUDA_CC) $(CUDA_ARCH) -c $(SRC)/cuda/cuda_distribution.cu -o $(OBJ)/cuda_distribution.o $(LCUDA) $(ICUDA) --relocatable-device-code true -lcurand

cuda_ciphertext.o:cuda_ciphertext.cu
	$(CUDA_CC) $(CUDA_ARCH) -c cuda_ciphertext.cu $(LCUDA) $(ICUDA) --relocatable-device-code true 

distribution.o:$(SRC)/distribution/distribution.cpp
	$(CC) -c $(SRC)/distribution/distribution.cpp -o $(OBJ)/distribution.o $(NTL) $(OPENMP) -lcurand  $(LCUDA) $(ICUDA)

yashe.o:$(SRC)/yashe/yashe.cpp
	$(CC) -c $(SRC)/yashe/yashe.cpp -o $(OBJ)/yashe.o $(NTL) $(OPENMP) $(LCUDA) $(ICUDA)

clean:
	rm -f $(OBJ)/*.o
