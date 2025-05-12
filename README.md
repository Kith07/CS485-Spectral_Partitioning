# Spectral Partitioning Project

By: Likhith Meesala

## Overview

This project implements spectral partitioning in four different versions:

* **Serial Only**: Completely sequential implementation using the C++ Eigen library
* **CUDA Only**: Parallelized eigen decomposition on GPU using cuSOLVER and cuBLAS CUDA libraries
* **MPI Only**: Parallel inertial matrix computation and projection  using MPI
* **CUDA+MPI**: Combines CUDA eigen computation with parallelized partitioning with MPI

All versions read can read from the 4 different input files and partition its nodes into *k* clusters based on user specifications

## Directory Structure

```
project/
├── Makefile
├── input_100.txt       # Default input file
├── source/
│   ├── serial_only/
│   │   └── spectral_serial.cpp
│   ├── cuda_only/
│   │   ├── spectral_cuda.cpp
│   │   └── cuda_eigen.cu
│   ├── mpi_only/
│   │   └── spectral_mpi.cpp
│   └── cuda_mpi/
│       ├── spectral_cuda_mpi.cpp
│       └── cuda.cu
├── doc/
│   ├── CS485_Spectral_Partitioning.pdf
└── README.md
```

## Prerequisites

* **Compilers:** Need to support g++, mpicxx, and nvcc
* **MPI library:** OpenMPI download necessary
* **NVIDIA CUDA Toolkit:** Includes `nvcc`, cuSOLVER, and cuBLAS
* **OpenMP support:** Enabled via `-fopenmp` flag
* **Eigen Library:** Already embedded within this folder so should work as is without any downloads necessary

Ensure environment variables (e.g., `PATH`, `LD_LIBRARY_PATH`) include CUDA and MPI installations so the programs know where to find them

## Building

From the project root directory, simply run:

```bash
make
```

This compiles all four files:

* `spec_ser` (serial)
* `spec_cu` (CUDA-only)
* `spec_mpi` (MPI-only)
* `spec_cu_mpi` (CUDA+MPI)

To remove compiled binary files:

```bash
make clean
```

## Running

The Makefile defines default variables:

```makefile
INPUT    := input_100.txt
CLUSTERS := 10
```

You can manually specify other input files and partition sizes:

```bash
make run-serial   INPUT=<FILE.txt> CLUSTERS=<NUMBER>
make run-cuda     INPUT=<FILE.txt> CLUSTERS=<NUMBER>
make run-mpi      INPUT=<FILE.txt> CLUSTERS=<NUMBER>
make run-cuda-mpi INPUT=<FILE.txt> CLUSTERS=<NUMBER>
```

Or run the compiled binary files directly:

```bash
./spec_ser     <input_file> <num_clusters>
./spec_cu      <input_file> <num_clusters>
mpirun -np 4 ./spec_mpi     <input_file> <num_clusters>
mpirun -np 4 ./spec_cu_mpi  <input_file> <num_clusters>
```

Adjust `-np` to specify the number of MPI processes

## Input Format

The input graph file must follow this structure:

```
n m idx
u1 v1
u2 v2
...
```

* `n`: number of nodes (0-based indexing)
* `m`: number of edges
* `idx`: either 0 or 1 indexed
* Each line after that lists an edge `(ui, vi)` connecting one node to the other

## Output

Each program prints partitions assignments:

```
Partition 0:
<node_ids>

Partition 1:
<node_ids>
...
```

## Performance Evaluation

Measure total execution time for each version using the `time` command:

```bash
time ./spec_ser input_100.txt 10
```

or

```bash
time mpirun -np 4 ./spec_mpi input_100.txt 10
```

If multiples machines are connected, the number of processes can exceed 4, however, running on this laptop has a maximum of 4 processes

## Future Improvements

* Swap out the adjacency matrix with a CSR for utility with exponentially larger graph inputs (> 100k nodes)
* Implement parallel radix sort to speed up the sorting of projected nodes onto the Principal Axis L for faster partitioning
* Implement recursive bisection to get optimal partitions rather than a simple even *k* partition split
* Extend to multi-machine parallel CUDA Eigensolver (Currently not widely available to parallelize eigenvalue decomposition)
---
