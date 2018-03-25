# Winky-Face
ECE1782 CUDA Project - Pagerank



## Experiment Results





## Useful Commands

```bash
// Compile the code and solve C4819 characters warning 
nvcc .\matrix.cu -o matrix -Xcompiler "/wd 4819" 

// Get runtime summary
nvprof .\xxx

// Run the programm
.\matrix .\graph\graph10.gr

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

