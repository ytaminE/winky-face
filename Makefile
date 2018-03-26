# tools
CC = gcc
RM = rm -frd
CFLAGS = -g -Wall
NVCC = nvcc -ccbin clang-3.8 -arch sm_52

# paths
GRAPH = ./graph/generateGraph
MATRIX = matrix

# target
TARGET = $(GRAPH) $(MATRIX)

all: $(TARGET)

$(GRAPH): $(GRAPH).c
	$(CC) $(CFLAGS) -o $(GRAPH) $(GRAPH).c
$(MATRIX): matrix.cu
	$(NVCC) -o $(MATRIX) $(MATRIX).cu 

clean:
	$(RM) $(TARGET) $(TARGET).o $(TARGET).dSYM


