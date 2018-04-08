The vertex-centric computation consist of a sequence of iterations.  
During each iteration, a user-defined function is invoked for each vertex.   
For the GPU case, a kernel function is invoked for a thread.  
The function specifies behavior at a single vertex *V* and a single iteration *S*.  
It can read messages sent to *V* from iteration *S*-1, send other vertices that will be received at iteration *S*+1.

## Array of Structures
link structure file  

Source ID | Out Degree | Dest ID
------------ | ------------- | -------------
id 1 | 2 | id45; id189
id 2 | 1 | id832
id 3 | 0 | 
id 4 | 3 | id236; id2501; id12042
... | ... | ... 

Use a pointer points to array of verteces.  
```
struct vertex {  
    unsigned int vertex_id;  
    float pagerank;  
    float pagerank_next;  
    unsigned int n_successors;  
    vertex ** successors;  
};
```
## Structure of Arrays
Using coalesced memory, Structure of Array:
![Image of Coalesced Memory](https://tianyue1994.github.io/img/coalesced.JPG)
```
struct vertex {
    unsigned int* vertex_id;
    float* pagerank;
    float* pagerank_next;
    unsigned int* n_successors;
    unsigned int** successors;
};
```
## Flattened Structure
 Have separate arrays for pagerank, pagerank_next, etc.
```
float * pagerank_h, *pagerank_d;
float *pagerank_next_d;
int * n_successors_h, *n_successors_d;
int * successors_h, *successors_d;             
int * successor_offset_h;
int * successor_offset_d;
```
No pointer, the successors information is stored is a large array and queried by an offset.
```
if (i < n_vertices) {
    n_suc = n_successors_d[i];
    if(n_suc > 0) {
    for(j = 0; j < n_suc; j++) {
        atomicAdd(&(pagerank_next_d[successors_d[successor_offset_d[i]+j]]), 0.85*(pagerank_d[i])/n_suc);
    }
    } else {
        atomicAdd(dangling_value2, 0.85*pagerank_d[i]);
    }
}
```
