# Winky-Face :wink:
ECE1782 CUDA Project - Pagerank



## Experiment Results





## Useful Commands

#### Windows

```bash
// Compile the code and solve C4819 characters warning in Windows
nvcc .\matrix.cu -o matrix -Xcompiler "/wd 4819"

// Compile the code with LCUBLAS library
nvcc .\matrix.cu -lcublas -o matrix -Xcompiler "/wd 4819"

// Get runtime summary
nvprof .\xxx

// Run the programm
.\matrix <path to the graph file> <number of iterations>
.\matrix .\graph\graph10.gr 10

```

#### Linux (eecg)

```bash
// To Compile
nvcc -ccbin clang++-3.8 ./matrix.cu -lcublas -o matrix

// To Run
./matrix <path to the graph file> <number of iterations>
./matrix ./graph/graph10.gr 10
```
## MAKE
$ ./make
$ ./make clean


## .gitignore

```
.gitignore
*.o
*.exe
*.exp
*.lib
*.txt
```
#edited by tianyue
pagerank_hostalloc_tree: 在host端存储图的vertex+edge数据， convergence函数使用树形的结构相加得到diff，理论上和AtomicAdd相比效率会有所提高（实际没有）
pagerank_thrust: Atomic Add被替换，使用Thrust库实现
