# 文档地址https://docs.google.com/document/d/1kndbj2JstYt6JxsBvNEs3N39vql52wgme3pEofy8z3M/edit?ts=5aa32e5f
# Slides 地址https://docs.google.com/presentation/d/1DNdbA-ElAMaXhMcv2pCWXuujh0ByWkPNHcJpvMKqrg8/edit?ts=5abbd2ec#slide=id.g36c4ecdf99_0_69

# Winky-Face :wink:
ECE1782 CUDA Project - Pagerank



## Experiment Results

### Matrix

|             | Number of Vertices | Maximum number of edges per vertex | CPU Time | GPU Time  |
| ----------- | ------------------ | ---------------------------------- | -------- | --------- |
| graph100_20 | 100                | 20                                 | 1ms      | 356.152us |
| graph500_20 | 500                | 20                                 | 20ms     | 8.4748ms  |
| graph800_20 | 800                | 20                                 | 65ms     | 30.692ms  |



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

// To compile matrixSparse, please include the cusp library
nvcc -ccbin clang++-3.8 -I ./lib/ ./matrixSparse.cu -o matrixSparse

// To Run
./matrix <path to the graph file> <number of iterations>
./matrix ./graph/graph10.gr 10
```
#### MAKE
```
// Compile all
./make
// Clean all compiled files
./make clean
```

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
pagerank_tree: ， convergence函数使用树形的结构相加得到diff，理论上和AtomicAdd相比效率会有所提高（实际没有）
使用1000000*100的数据

## Test and Measure
### Input
#### Parameters
```
100 50/100/200 # drop this group as not obvious differences could be observed
1000 50/100/200
10000 50/100/200
100000 50/100/200
1000000 50/100
```
#### Functions
```
./matrix
./matirxCPU
./pagerank_CPU_benchmark
./pagerank_GPU_benchmark
./pagerank_atomicadd
./pagerank_hostalloc_tree
./pagerank_SOA             

About vertex based solutions:

Maximum data handled: pagerank_atomicadd < else
```
### Measure
#### notes
- 10 loops for each
- using "./top" to monitor current processes to make sure that all computing resources are available
- on the early Morning for accuracy
### Output
#### Downloadable
[test data archives(download here)](https://drive.google.com/drive/folders/1wK5NBYzm4pglYipjxyf7UYYFuBKYtxDy?usp=sharing)

*test results are updated*

[test results table](https://drive.google.com/file/d/12eNUXRoAMZVVQoU1iMkegG34BuiK5BQB/view?usp=sharing)

[test raw output(5 loops)](https://drive.google.com/open?id=1lkRBsVK3iXGaffRLih5S6GCZkLNf7u1i)

[test raw output(10 loops)](https://drive.google.com/drive/folders/1tBszowtItmUJvA8qKewTPLiHW8E-ZQVU?usp=sharing)

[matrix_test_results_table](https://docs.google.com/spreadsheets/d/1jsGo_Q_5oBVMeJ-dCmkEsnPyBsyqFZTZfi-oxXjkQ1k/edit?usp=sharing)

### Hardware(eecg)
Intel(R) Core(TM) i7-4790 CPU @ 3.60GHz
```
// show more info about CPU
./lscpu
```
NVIDIA Corporation GM204 [GeForce GTX 980]
```
// show more info about GPU
./nvidia-smi -q
```

## Final
### ppt
[pagerank](https://docs.google.com/presentation/d/1DNdbA-ElAMaXhMcv2pCWXuujh0ByWkPNHcJpvMKqrg8/edit#slide=id.g362d662672_0_25)
### report
## Reference
[CUDA_Image_Encryption, by Tong Zou, Jing Wang](https://github.com/DracoZT/CUDA_Image_Encryption)
