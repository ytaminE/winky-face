# tools
CC = gcc
RM = rm -frd
CFLAGS = -g -Wall
NVCC = nvcc -ccbin clang++-3.8 -lcublas

# paths
GRAPH = ./graph/generateGraph
TARGET = matrix pagerank_hostalloc_tree pagerank_thrust_add $(GRAPH)

all: $(TARGET)

$(GRAPH): $(GRAPH).c
	$(CC) $(CFLAGS) -o $(GRAPH) $(GRAPH).c
matrix: matrix.cu
	$(NVCC) -o matrix matrix.cu
pagerank_hostalloc_tree: pagerank_hostalloc_tree.cu
	$(NVCC) -o pagerank_hostalloc_tree pagerank_hostalloc_tree.cu
pagerank_thrust_add: pagerank_thrust_add.cu
	$(NVCC) -o pagerank_thrust_add pagerank_thrust_add.cu

clean:
	$(RM) $(TARGET) *~
