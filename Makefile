# paths
GRAPH_PATH = ./graph/

# tools
CC = gcc
RM = rm -frd
CFLAGS = -g -Wall
NVCC = nvcc -ccbin clang-3.8 -arch sm_52

TARGET = $(GRAPH_PATH)/generateGraph matrix 

all: $(TARGET)
$(GRAPH_PATH)/generateGraph: $(GRAPH_PATH)/generateGraph.c
	$(CC) $(CFLAGS) -o $(TARGET) $(TARGET).c
matrix: matrix.cu
	$(NVCC) -o matirx matrix.cu 

clean:
	$(RM) $(TARGET) $(TARGET).o $(TARGET).dSYM


