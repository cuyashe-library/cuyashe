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

all: tests benchmarks

tests: test.o operators.o polynomial.o ciphertext.o cuda_bn.o cuda_distribution.o distribution.o logging.o cuda_bn.o yashe.o cuda_ciphertext.o coprimes.o
	$(CUDA_CC) $(CUDA_ARCH) $(LCUDA) $(ICUDA) -o $(BIN)/test $(OBJ)/test.o $(OBJ)/polynomial.o $(OBJ)/ciphertext.o $(OBJ)/operators.o $(OBJ)/cuda_distribution.o $(OBJ)/cuda_bn.o $(OBJ)/distribution.o $(OBJ)/logging.o $(OBJ)/log.o $(OBJ)/coprimes.o $(OBJ)/yashe.o $(OBJ)/cuda_ciphertext.o -lcufft -lcurand  --relocatable-device-code true -Xcompiler $(OPENMP) $(NTL) -lboost_unit_test_framework

benchmarks: benchmark_poly.o operators.o benchmark_yashe.o cuda_bn.o polynomial.o logging.o distribution.o cuda_distribution.o yashe.o cuda_ciphertext.o coprimes.o
	$(CUDA_CC) $(CUDA_ARCH) $(LCUDA) $(ICUDA) -o $(BIN)/benchmark_poly $(OBJ)/benchmark_poly.o $(OBJ)/polynomial.o $(OBJ)/ciphertext.o $(OBJ)/yashe.o $(OBJ)/operators.o $(OBJ)/cuda_bn.o $(OBJ)/distribution.o $(OBJ)/cuda_distribution.o $(OBJ)/coprimes.o $(OBJ)/logging.o $(OBJ)/cuda_ciphertext.o $(OBJ)/log.o -lcufft -lcurand  --relocatable-device-code true $(NTL) -Xcompiler $(OPENMP) $(NTL) -lboost_unit_test_framework
	$(CUDA_CC) $(CUDA_ARCH) $(LCUDA) $(ICUDA) -o $(BIN)/benchmark_yashe $(OBJ)/benchmark_yashe.o $(OBJ)/polynomial.o $(OBJ)/ciphertext.o $(OBJ)/yashe.o $(OBJ)/operators.o $(OBJ)/cuda_bn.o $(OBJ)/distribution.o $(OBJ)/cuda_distribution.o $(OBJ)/coprimes.o $(OBJ)/cuda_ciphertext.o $(OBJ)/logging.o $(OBJ)/log.o -lcufft -lcurand  --relocatable-device-code true $(NTL) -Xcompiler $(OPENMP) $(NTL) -lboost_unit_test_framework

test.o: $(SRC)/test/test.cpp
	$(CC) -c $(SRC)/test/test.cpp -o $(OBJ)/test.o $(NTL) $(OPENMP) -lcurand  $(LCUDA) $(ICUDA)

benchmark_poly.o: $(SRC)/benchmark/polynomial.cpp
	$(CC) -c $(SRC)/benchmark/polynomial.cpp -o $(OBJ)/benchmark_poly.o $(NTL) $(OPENMP) -lcurand  $(LCUDA) $(ICUDA)

benchmark_yashe.o: $(SRC)/benchmark/yashe.cpp
	$(CC) -c $(SRC)/benchmark/yashe.cpp -o $(OBJ)/benchmark_yashe.o $(NTL) $(OPENMP) -lcurand  $(LCUDA) $(ICUDA)

operators.o:$(SRC)/cuda/operators.cu
	$(CUDA_CC) $(CUDA_ARCH) -c $(SRC)/cuda/operators.cu -o $(OBJ)/operators.o $(LCUDA) $(ICUDA) -lcufft --relocatable-device-code true $(NTL) -Xcompiler $(OPENMP)

polynomial.o:$(SRC)/aritmetic/polynomial.cu
	$(CUDA_CC) $(CUDA_ARCH) -c $(SRC)/aritmetic/polynomial.cu -o $(OBJ)/polynomial.o $(LCUDA) $(ICUDA) -lcufft --relocatable-device-code true $(NTL) -Xcompiler $(OPENMP) 

ciphertext.o:$(SRC)/yashe/ciphertext.cpp
	$(CC) -c $(SRC)/yashe/ciphertext.cpp -o $(OBJ)/ciphertext.o $(NTL) $(OPENMP)  $(LCUDA) $(ICUDA)

coprimes.o:$(SRC)/aritmetic/coprimes.cpp
	$(CC) -c $(SRC)/aritmetic/coprimes.cpp -o $(OBJ)/coprimes.o $(LCUDA) $(ICUDA) 

logging.o: $(SRC)/logging/logging.cpp
	$(CC) -c $(SRC)/logging/log.c -o $(OBJ)/log.o
	$(CC) -c -w $(SRC)/logging/logging.cpp -o $(OBJ)/logging.o

cuda_bn.o:$(SRC)/cuda/cuda_bn.cu
	$(CUDA_CC) $(CUDA_ARCH) -c $(SRC)/cuda/cuda_bn.cu -o $(OBJ)/cuda_bn.o $(LCUDA) $(ICUDA) --relocatable-device-code true $(NTL)

cuda_distribution.o:$(SRC)/cuda/cuda_distribution.cu
	$(CUDA_CC) $(CUDA_ARCH) -c $(SRC)/cuda/cuda_distribution.cu -o $(OBJ)/cuda_distribution.o $(LCUDA) $(ICUDA) --relocatable-device-code true -lcurand $(NTL)

cuda_ciphertext.o:$(SRC)/cuda/cuda_ciphertext.cu
	$(CUDA_CC) $(CUDA_ARCH) -c $(SRC)/cuda/cuda_ciphertext.cu -o $(OBJ)/cuda_ciphertext.o $(LCUDA) $(ICUDA) --relocatable-device-code true  $(NTL)

distribution.o:$(SRC)/distribution/distribution.cpp
	$(CC) -c $(SRC)/distribution/distribution.cpp -o $(OBJ)/distribution.o $(NTL) $(OPENMP) -lcurand  $(LCUDA) $(ICUDA) 

yashe.o:$(SRC)/yashe/yashe.cpp
	$(CC) -c $(SRC)/yashe/yashe.cpp -o $(OBJ)/yashe.o $(NTL) $(OPENMP) $(LCUDA) $(ICUDA)


# Special tests
test_distribution.o: $(SRC)/test/test_distribution.cu
	$(CUDA_CC) $(CUDA_ARCH) -c $(SRC)/test/test_distribution.cu -o $(OBJ)/test_distribution.o $(LCUDA) $(ICUDA)

test_distribution: test_distribution.o cuda_distribution.o distribution.o operators.o polynomial.o logging.o cuda_bn.o cuda_ciphertext.o yashe.o
	$(CUDA_CC) $(CUDA_ARCH) -o $(BIN)/test_distribution $(OBJ)/test_distribution.o $(OBJ)/operators.o $(OBJ)/polynomial.o $(OBJ)/cuda_ciphertext.o $(OBJ)/yashe.o $(OBJ)/distribution.o $(OBJ)/cuda_distribution.o $(OBJ)/cuda_bn.o $(OBJ)/logging.o $(OBJ)/log.o -lcufft -lcurand $(NTL)

clean:
	rm -f $(OBJ)/*.o
