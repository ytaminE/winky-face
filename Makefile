# tools
CC = gcc
RM = rm -frd
CFLAGS = -g -Wall
NVCC = nvcc -ccbin clang++-3.8 -lcublas

# paths
GRAPH = ./graph/generateGraph
TARGET = matrix matrixCPU pagerank_hostalloc_tree pagerank_thrust_add pagerank_atomicadd $(GRAPH) pagerank_CPU_benchmark pagerank_GPU_benchmark

all: $(TARGET)

$(GRAPH): $(GRAPH).c
	$(CC) $(CFLAGS) -o $(GRAPH) $(GRAPH).c
matrix: matrix.cu
	$(NVCC) -o matrix matrix.cu
matrixCPU: matrixCPU.cu
	$(NVCC) -o matrixCPU matrixCPU.cu
pagerank_atomicadd: pagerank_atomicadd.cu
	$(NVCC) -o pagerank_atomicadd pagerank_atomicadd.cu
pagerank_CPU_benchmark: pagerank_CPU_benchmark.cu
	$(NVCC) -o pagerank_CPU_benchmark pagerank_CPU_benchmark.cu
pagerank_GPU_benchmark: pagerank_GPU_benchmark.cu
	$(NVCC) -o pagerank_GPU_benchmark pagerank_GPU_benchmark.cu
pagerank_hostalloc_tree: pagerank_hostalloc_tree.cu
	$(NVCC) -o pagerank_hostalloc_tree pagerank_hostalloc_tree.cu
pagerank_thrust_add: pagerank_thrust_add.cu
	$(NVCC) -o pagerank_thrust_add pagerank_thrust_add.cu



clean:
	$(RM) $(TARGET) *~
