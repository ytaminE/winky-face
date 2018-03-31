#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>


typedef struct vertex vertex;

struct vertex {
    unsigned int* vertex_id;
    float* pagerank;
    float* pagerank_next;
    unsigned int* n_successors;
    unsigned int** successors;
};


__global__ void initializePageranks(vertex vertices, int n_vertices) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x; 

    if (i < n_vertices) {
        vertices.pagerank[i] = 1.0/(float)n_vertices;
        vertices.pagerank_next[i] = 0.0;
    }
}


__global__ void addToNextPagerank(vertex vertices, float * dangling_value, int n_vertices) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x; 
    int j;

    if(i < n_vertices) {
        if(vertices.n_successors[i] > 0) {
            for(j = 0; j < vertices.n_successors[i]; j++) {
                atomicAdd(&(vertices.pagerank_next[*(vertices.successors[i]+j)]),
				0.85*(vertices.pagerank[i])/vertices.n_successors[i]);
            }
        }else {
            atomicAdd(dangling_value, 0.85*vertices.pagerank[i]);
        }
    }
}

__global__ void finalPagerankForIteration(vertex vertices, int n_vertices, float dangling_value){
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (i < n_vertices) {
        vertices.pagerank_next[i] += (dangling_value + (1-0.85))/((float)n_vertices);
    }
}

__global__ void setPageranksFromNext(vertex vertices, int n_vertices) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (i < n_vertices) {
        vertices.pagerank[i] = vertices.pagerank_next[i];
        vertices.pagerank_next[i] = 0.0;
    }
}

__global__ void convergence(vertex vertices, float * reduced_sums_d, int n_vertices) {
// Each thread computes the diff for two vertexes (thus, half # of blocks needed for this function)
// Because of this, we need to handle the case where only one block is needed
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int i_thr = threadIdx.x;

    __shared__ float sums[1024];                       // blockDim.x == 1024
    
    float temp1, temp2;    
    
    if(i < 1024) {
        reduced_sums_d[i] = 0;
    }

    if(i < n_vertices) {
        temp1 = vertices.pagerank_next[i] - vertices.pagerank[i];
        if(i + (1024 * gridDim.x) < n_vertices) {
            temp2 = vertices.pagerank_next[i+ (1024 * gridDim.x)] - vertices.pagerank[i +(1024*gridDim.x)];
        }else{
            temp2 = 0;
        }

        if(temp1 < 0) {
            temp1 = temp1 * (-1);
        }
        if(temp2 < 0) {
            temp2 = temp2 * (-1);
        }

        sums[i_thr] = temp1 + temp2;
    } else {
        sums[i_thr] = 0;
    }
    __syncthreads();

    int j, index, index2;
    index = i_thr;

    for(j = 0; j < 10; j++) {                    // 10 times as 2^10 = 1024 threads
        if((index+1) % (2 * (1 << j)) == 0) {    // Note: 1 << j == 2^j
            index2 = index - (1 << j);
            sums[index] += sums[index2];
        }
        __syncthreads();
    }

    reduced_sums_d[blockIdx.x] = sums[1023];
}

__global__ void getConvergence(float * reduced_sums_d, float * diff) {
    int j, index, index2;
    index = threadIdx.x;
    
    for(j = 0; j < 10; j++) {                    // 10 times as 2^10 = 1024 threads
        if((index+1) % (2 * (1 << j)) == 0) {    // Note: 1 << j == 2^j
            index2 = index - (1 << j);
            reduced_sums_d[index] += reduced_sums_d[index2];
        }
        __syncthreads();
    }    

    *diff = reduced_sums_d[1023]; 
}


int main(int argc, char ** args) {
    if (argc != 2) {
	fprintf(stderr,"Wrong number of args. Provide input graph file.\n");
        exit(-1);
    } 

    size_t mem_total = 0;
    size_t mem_free = 0;

    cudaFree(0); // Initialize the cuda context
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    cudaMemGetInfo(&mem_free, &mem_total);
    printf("1. mem_total: %zu, mem_free: %zu\n",mem_total, mem_free);

/*************************************************************************/
    // Start CPU timer
    clock_t cycles_to_build, cycles_to_calc;


/*************************************************************************/
    // build up the graph
    int i;
    unsigned int n_vertices = 0;
    unsigned int vertex_from = 0, vertex_to = 0;

    vertex vertices;

    FILE * fp;
    if ((fp = fopen(args[1], "r")) == NULL) {
        fprintf(stderr,"ERROR: Could not open input file.\n");
        exit(-1);
     }

    // parse input file to count the number of vertices
    // expected format: vertex_from vertex_to
    while (fscanf(fp, "%u %u", &vertex_from, &vertex_to) != EOF) {
        if (vertex_from > n_vertices) {
            n_vertices = vertex_from;
	}
        else if (vertex_to > n_vertices) {
            n_vertices = vertex_to;
	}
    }
    n_vertices++;
    clock_t start = clock();
    // allocate memory for vertices
    //err = cudaMallocManaged((void **)&vertices, n_vertices*sizeof(vertex));
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    err = cudaMallocManaged((void **)&vertices.vertex_id, n_vertices*sizeof(vertices.vertex_id));
	err = cudaMallocManaged((void **)&vertices.pagerank, n_vertices*sizeof(vertices.pagerank));
	err = cudaMallocManaged((void **)&vertices.pagerank_next, n_vertices*sizeof(vertices.pagerank_next));
	err = cudaMallocManaged((void **)&vertices.n_successors, n_vertices*sizeof(vertices.n_successors));
	err = cudaMallocManaged((void **)&vertices.successors, n_vertices*sizeof(vertices.successors));
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //if (!vertices) {
    //    fprintf(stderr,"Malloc failed for vertices.\n");
    //    exit(-1);
    //}
    //memset((void *)vertices, 0, (size_t)(n_vertices*sizeof(vertex)));
	memset((void *)vertices.vertex_id, 0, (size_t)(n_vertices*sizeof(vertices.vertex_id)));
    memset((void *)vertices.pagerank, 0, (size_t)(n_vertices*sizeof(vertices.pagerank)));
	memset((void *)vertices.pagerank_next, 0, (size_t)(n_vertices*sizeof(vertices.pagerank_next)));
	memset((void *)vertices.n_successors, 0, (size_t)(n_vertices*sizeof(vertices.n_successors)));
	memset((void *)vertices.successors, 0, (size_t)(n_vertices*sizeof(vertices.successors)));
	
    // parse input file to count the number of successors of each vertex
    fseek(fp, 0L, SEEK_SET);
    while (fscanf(fp, "%u %u", &vertex_from, &vertex_to) != EOF) {
        vertices.n_successors[vertex_from]++;
    }
    printf("sizeof(vertex*): %d\n", sizeof(vertex*));
    printf("sizeof(vertex): %d\n", sizeof(vertex));

    cudaMemGetInfo(&mem_free, &mem_total);
    printf("mem_total: %zu, mem_free: %zu\n",mem_total, mem_free);
    // allocate memory for successor pointers
    for (i=0; i<n_vertices; i++) {
        vertices.vertex_id[i] = i;
        if (vertices.n_successors[i] > 0) {
            err = cudaMallocManaged((void***)&vertices.successors[i],vertices.n_successors[i]*sizeof(unsigned int*));
	    cudaMemGetInfo(&mem_free, &mem_total);
	    cudaDeviceSynchronize();
    	    //printf("i:%d, mem_total: %zu, mem_free: %zu\n",i, mem_total, mem_free);
            if (!vertices.successors[i]) {
                fprintf(stderr,"cudaMallocManaged failed for vertex %d successors (error: %s)\n",i,cudaGetErrorString(err));
		cudaMemGetInfo(&mem_free, &mem_total);
	    	cudaDeviceSynchronize();
    	    	printf("i:%d, mem_total: %zu, mem_free: %zu\n",i, mem_total, mem_free);
                exit(-1);
            }
            memset((void *)vertices.successors[i], 0, (size_t)(vertices.n_successors[i]*sizeof(unsigned int *)));
        }
        else
            vertices.successors[i] = NULL;
    }

    // parse input file to set up the successor pointers
    fseek(fp, 0L, SEEK_SET);
    while (fscanf(fp, "%d %d", &vertex_from, &vertex_to) != EOF) {
        for (i=0; i<vertices.n_successors[vertex_from]; i++) {
            if (*(vertices.successors[vertex_from]+i) == NULL) {
                *(vertices.successors[vertex_from]+i) = vertices.vertex_id[vertex_to];
                break;
            }
            else if (i==vertices.n_successors[vertex_from]-1) {
                printf("Setting up the successor pointers of virtex %u failed",vertex_from);
                return -1;
            }
        }
    }

    fclose(fp);

    // Get time for building data structure
    cycles_to_build = clock() - start;
    int build_msec = cycles_to_build * 1000 / CLOCKS_PER_SEC;
    
    //Reset time 
    start = clock();
   

  /*************************************************************************/
    // compute the pagerank on the GPU
    int n_iterations = 30;
    int numOfBlocks = 1;         // default value for 1000 vertex graph
    int threadsPerBlock = 1000;  // default value for 1000 vertex graph
    int converge_blocks = (n_vertices + 2048 - 1)/2048;
    if(converge_blocks == 0) {
        converge_blocks =1;
    }           


    if(n_vertices <= 1024) {
        threadsPerBlock = n_vertices;
        numOfBlocks = 1;
    } else {
        threadsPerBlock = 1024;
        numOfBlocks = (n_vertices + 1023)/1024;   // The "+ 1023" ensures we round up
    }   


    float dangling_value_h = 0;
    float * dangling_value_d;
    float * reduced_sums_d;
    float epsilon = 0.000001;
    float * d_diff;
    float h_diff = epsilon + 1;

    err = cudaMalloc((void **)&d_diff, sizeof(float));
    err = cudaMalloc((void **)&reduced_sums_d, 1024 * sizeof(float));

    err = cudaMalloc((void **)&dangling_value_d, sizeof(float));
    err = cudaMemcpy(dangling_value_d, &dangling_value_h, sizeof(float), cudaMemcpyHostToDevice);

    // Initialize pagerank and pagerank_next values
    initializePageranks<<<numOfBlocks,threadsPerBlock>>>(vertices, n_vertices);
    cudaDeviceSynchronize(); 
 
    int iteration = 0;
    while(epsilon < h_diff  && iteration < n_iterations) {
        // set the dangling value to 0 
        dangling_value_h = 0;
        err = cudaMemcpy(dangling_value_d, &dangling_value_h, sizeof(float), cudaMemcpyHostToDevice);
        
        // initial parallel pagerank_next computation
        addToNextPagerank<<<numOfBlocks,threadsPerBlock>>>(vertices, dangling_value_d, n_vertices);

        // get the dangling value
        err = cudaMemcpy(&dangling_value_h, dangling_value_d, sizeof(float), cudaMemcpyDeviceToHost);
 
        // final parallel pagerank_next computation
        finalPagerankForIteration<<<numOfBlocks,threadsPerBlock>>>(vertices, n_vertices, dangling_value_h);

        convergence<<<converge_blocks, 1024>>>(vertices, reduced_sums_d, n_vertices);
        getConvergence<<<1,1024>>>(reduced_sums_d, d_diff);
        
        // Get difference to compare to epsilon
        cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost);
        
        // Set pagerank = pagerank_next; And then pagerank_next = 0;
        setPageranksFromNext<<<numOfBlocks,threadsPerBlock>>>(vertices, n_vertices);
        
        iteration++;
    }
    cudaDeviceSynchronize();
    
    // End CPU Timer
    cycles_to_calc = clock() - start;

    // Print CPU time
    int calc_msec = cycles_to_calc * 1000 / CLOCKS_PER_SEC;

    // print the pagerank values computed on the GPU
    for (i=0;i<n_vertices;i++) {
        printf("AFTER GPU | Vertex %u:\tpagerank = %.6f\n", i, vertices.pagerank[i]);
    }
    
	
    printf("Time to build: %d seconds, %d milliseconds\n", build_msec/1000, build_msec%1000);
    printf("Time to calc: %d seconds, %d milliseconds\n", calc_msec/1000, calc_msec%1000);
    printf("Iteration: %d\n", iteration);


    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Done\n");
    return 0;
}

