# Compilers
CXX      := g++
MPICXX   := mpicxx
NVCC     := nvcc

# Flags & includes
INCLUDE       := -I .
CXXFLAGS      := -std=c++14
NVCCFLAGS     := -Xcompiler="-fopenmp"
CUSOLVER_LIBS := -lcusolver -lcublas

# Sources
SERIAL_SRC    := serial_only/spectral_serial.cpp
CUDA_SRC      := cuda_only/spectral_cuda.cpp cuda_only/cuda_eigen.cu
MPI_SRC       := mpi_only/spectral_mpi.cpp
CUDA_MPI_SRC  := cuda_mpi/spectral_cuda_mpi.cpp cuda_mpi/cuda.cu

# Binaries
SERIAL_BIN    := spec_ser
CUDA_BIN      := spec_cu
MPI_BIN       := spec_mpi
CUDA_MPI_BIN  := spec_cu_mpi

# Default parameters (override on make command-line)
INPUT    ?= input_100.txt
CLUSTERS ?= 5
RANKS    ?= 4

.PHONY: all build run clean

all: build

build: $(SERIAL_BIN) $(CUDA_BIN) $(MPI_BIN) $(CUDA_MPI_BIN)

$(SERIAL_BIN): $(SERIAL_SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $< -o $@

$(CUDA_BIN): $(word 1, $(CUDA_SRC)) $(word 2, $(CUDA_SRC))
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUSOLVER_LIBS)

$(MPI_BIN): $(MPI_SRC)
	$(MPICXX) $(CXXFLAGS) $< $(INCLUDE) -o $@ -lm

$(CUDA_MPI_BIN): $(CUDA_MPI_SRC)
	$(NVCC) -ccbin=$(MPICXX) $(NVCCFLAGS) -o $@ $^ $(CUSOLVER_LIBS)


# Run everything, capture outputs, then dump them
run: build
	@echo "Running on input: $(INPUT), clusters: $(CLUSTERS), MPI ranks: $(RANKS)"
	@echo

	@echo "--- Serial ---"          > serial.out
	@./$(SERIAL_BIN)  $(INPUT) $(CLUSTERS) >> serial.out

	@echo "--- CUDA-only ---"       > cuda.out
	@./$(CUDA_BIN)    $(INPUT) $(CLUSTERS) >> cuda.out

	@echo "--- MPI-only ($(RANKS)) ---" > mpi.out
	@mpirun -np $(RANKS) ./$(MPI_BIN)   $(INPUT) $(CLUSTERS) >> mpi.out

	@echo "--- CUDA+MPI ($(RANKS)) ---" > cu_mpi.out
	@mpirun -np $(RANKS) ./$(CUDA_MPI_BIN) $(INPUT) $(CLUSTERS) >> cu_mpi.out

	@echo; echo "=== Results ==="; echo
	@cat serial.out
	@cat cuda.out
	@cat mpi.out
	@cat cu_mpi.out

clean:
	rm -f $(SERIAL_BIN) $(CUDA_BIN) $(MPI_BIN) $(CUDA_MPI_BIN) serial.out cuda.out mpi.out cu_mpi.out