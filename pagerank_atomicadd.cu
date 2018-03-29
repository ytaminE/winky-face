#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

__global__ void initializePagerankArray(float * pagerank_d, int n_vertices) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n_vertices) {
        pagerank_d[i] = 1.0/(float)n_vertices;
    }
}

__global__ void setPagerankNextArray(float * pagerank_next_d, int n_vertices) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n_vertices) {
        pagerank_next_d[i] = 0.0;
    }
}


__global__ void addToNextPagerankArray(float * pagerank_d, float * pagerank_next_d, int * n_successors_d, int * successors_d, int * successor_offset_d, float * dangling_value2, int n_vertices) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j;
    int n_suc;
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
}       

__global__ void finalPagerankArrayForIteration(float * pagerank_next_d, int n_vertices, float dangling_value2) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(i < n_vertices) {
        pagerank_next_d[i] += (dangling_value2 + (1-0.85))/((float)n_vertices);
    }
}

__global__ void setPagerankArrayFromNext(float * pagerank_d, float * pagerank_next_d, int n_vertices, float *diff) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	float temp;
    if(i < n_vertices) {
		temp=pagerank_d[i];
        pagerank_d[i] = pagerank_next_d[i];
        pagerank_next_d[i] = 0.0;
		atomicAdd(diff,((temp - pagerank_d[i])>=0)?(temp- pagerank_d[i]):(pagerank_d[i]-temp) );
    }
}      

int main(int argc, char ** args) {
    if (argc != 2) {
	fprintf(stderr,"Wrong number of args. Provide input graph file.\n");
        exit(-1);
    } 
    cudaFree(0);   
    cudaError_t err = cudaSuccess;
    cudaProfilerStart();
    
    // Start CPU timer
    clock_t cycles_to_build, cycles_to_calc;


    // build up the graph
    int i;
    unsigned int n_vertices = 0;
    unsigned int n_edges = 0;
    unsigned int vertex_from = 0, vertex_to = 0, vertex_prev = 0;
    
    // Vertex
    float * pagerank_h, *pagerank_d;
    float *pagerank_next_d;
    int * n_successors_h, *n_successors_d;
    int * successors_h, *successors_d;             
    int * successor_offset_h;
	int *successor_offset_d;

    FILE * fp;
    if ((fp = fopen(args[1], "r")) == NULL) {
        fprintf(stderr,"ERROR: Could not open input file.\n");
        exit(-1);
    }

    // parse input file to count the number of vertices
    while (fscanf(fp, "%u %u", &vertex_from, &vertex_to) != EOF) {
        if (vertex_from > n_vertices) {
            n_vertices = vertex_from;
	}
        else if (vertex_to > n_vertices) {
            n_vertices = vertex_to;
	}
	n_edges++;
    }
    n_vertices++;
     clock_t start = clock();   
    // Allocate flattened data structure host and device memory
    pagerank_h = (float *) malloc(n_vertices * sizeof(*pagerank_h));
    err = cudaMalloc((void **)&pagerank_d, n_vertices*sizeof(float));
    err = cudaMalloc((void **)&pagerank_next_d, n_vertices*sizeof(float));
    n_successors_h = (int *) calloc(n_vertices, sizeof(*n_successors_h));
    err = cudaMalloc((void **)&n_successors_d, n_vertices*sizeof(int));
    successor_offset_h = (int *) malloc(n_vertices * sizeof(*successor_offset_h));
    err = cudaMalloc((void **)&successor_offset_d, n_vertices*sizeof(int));

    // Allocate memory for contiguous successors_d data
    successors_h = (int *) malloc(n_edges * sizeof(*successors_h));
    err = cudaMalloc((void **)&successors_d, n_edges*sizeof(int));

    // allocate memory for successor pointers
    int offset = 0, edges = 0;      

    // parse input file to count the number of successors of each vertex
    fseek(fp, 0L, SEEK_SET);
    i = 0;
 
    while (fscanf(fp, "%u %u", &vertex_from, &vertex_to) != EOF) {
        n_successors_h[vertex_from] += 1;
	
	// Fill successor_offset_h array
        successor_offset_h[i] = offset;
	if(edges != 0 && vertex_prev != vertex_from) {
	    i = vertex_from;
	    offset = edges;
	    successor_offset_h[i] = offset;	   
	    vertex_prev = vertex_from;
	}

	// Fill successor array
	successors_h[edges] = vertex_to;
	
	edges++;
    }
    successor_offset_h[i] = edges - 1;    

    fclose(fp);

    // Get build time and reset start
    cycles_to_build = clock() - start;
    start = clock();

    // Transfer data structure to the GPU
    err = cudaMemcpy(n_successors_d, n_successors_h, n_vertices*sizeof(int), cudaMemcpyHostToDevice);
    err = cudaMemcpy(successors_d, successors_h, n_edges*sizeof(int), cudaMemcpyHostToDevice);
    err = cudaMemcpy(successor_offset_d, successor_offset_h, n_vertices*sizeof(int), cudaMemcpyHostToDevice);

    // Compute the pagerank
    int n_iterations = 30;
    int iteration = 0;
    int numOfBlocks = 1;                 
    int threadsPerBlock = 1000;                 

    if(n_vertices <= 1024) {
        threadsPerBlock = n_vertices;
        numOfBlocks = 1;
    } else {
        threadsPerBlock = 1024;
        numOfBlocks = (n_vertices + 1023)/1024;   // The "+ 1023" ensures we round up
    }

    float dangling_value_h = 0;
    float dangling_value_h2 = 0;
    float *dangling_value2, *reduced_sums_d;
    int n_blocks = (n_vertices + 2048 - 1)/2048;
    if (n_blocks == 0){
        n_blocks = 1;
    }
    float epsilon = 0.000001;
    float * d_diff;
    float h_diff = epsilon + 1;

    err = cudaMalloc((void **)&d_diff, sizeof(float));
    err = cudaMalloc((void **)&reduced_sums_d, 1024 * sizeof(float));     
    err = cudaMalloc((void **)&dangling_value2, sizeof(float));
    err = cudaMemcpy(dangling_value2, &dangling_value_h, sizeof(float), cudaMemcpyHostToDevice);

    initializePagerankArray<<<numOfBlocks,threadsPerBlock>>>(pagerank_d, n_vertices);
	cudaDeviceSynchronize();
    setPagerankNextArray<<<numOfBlocks,threadsPerBlock>>>(pagerank_next_d, n_vertices);
    cudaDeviceSynchronize();
	
    while(epsilon < h_diff && iteration < n_iterations) {  //was 23
       // set the dangling value to 0 
        dangling_value_h = 0;
        err = cudaMemcpy(dangling_value2, &dangling_value_h, sizeof(float), cudaMemcpyHostToDevice);     
        // initial parallel pagerank_next computation
        addToNextPagerankArray<<<numOfBlocks,threadsPerBlock>>>(pagerank_d, pagerank_next_d, n_successors_d, successors_d, successor_offset_d, dangling_value2, n_vertices);
        cudaDeviceSynchronize();
        // get the dangling value
        err = cudaMemcpy(&dangling_value_h2, dangling_value2, sizeof(float), cudaMemcpyDeviceToHost); 
        // final parallel pagerank_next computation
        finalPagerankArrayForIteration<<<numOfBlocks,threadsPerBlock>>>(pagerank_next_d, n_vertices, dangling_value_h2);
        cudaDeviceSynchronize();       
        // Get difference to compare to epsilon
		cudaMemset(d_diff, 0, sizeof(float) );
        setPagerankArrayFromNext<<<numOfBlocks,threadsPerBlock>>>(pagerank_d, pagerank_next_d, n_vertices, d_diff);
		cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost);      
		printf("probe2:   %f\n", h_diff);
        cudaDeviceSynchronize();

        iteration++;
    }

    err = cudaMemcpy(pagerank_h, pagerank_d, n_vertices*sizeof(float), cudaMemcpyDeviceToHost);

    // CPU time
    cycles_to_calc = clock() - start;
    int build_milli = cycles_to_build * 1000 / CLOCKS_PER_SEC;
    int calc_milli = cycles_to_calc * 1000 / CLOCKS_PER_SEC;
 
	FILE *f_result;
	f_result=fopen("rg","w");
	for (i=0;i<n_vertices;i++) {
        fprintf(f_result,"Vertex %u:\tpagerank = %.18f\n", i, pagerank_h[i]);
	}  

    printf("Time to build: %d seconds, %d milliseconds\n",build_milli/1000, build_milli%1000);
    printf("Time to calc: %d seconds, %d milliseconds\n",calc_milli/1000, calc_milli%1000);
    printf("iter: %d\n", iteration);


    // Free device global memory
    err = cudaFree(pagerank_d);
    err = cudaFree(pagerank_next_d);

    // Free host memory
    free(pagerank_h);

    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Done\n");
    return 0;
}

