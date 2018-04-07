#include <stdio.h>
#include <iostream>
#include <cuda_profiler_api.h>
#include <cublas_v2.h>
// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>

float* gpu_blas_mmul(float *A, float *B, float *C, int m, int k, int n, int n_iterations, cublasHandle_t handle);
void printPageRank(float* pageRank, int n_vertices);
void printMatrix(float* matrix, int n_vertices);
float abs_float(float in);

int main(int argc, char ** args) {
    // Initialize the cuda context
    cudaFree(0);  
    cudaProfilerStart();

    // Start CPU timer
    clock_t cycles_to_calc;
    clock_t cycles_to_build;

    clock_t startBuildTime = clock();

    // Initialize the graph context
     unsigned int n_vertices = 0;                   // number of vertices
     unsigned int n_edges = 0;                      // number of edges
     unsigned int vertex_from = 0, vertex_to = 0;   // edge from vertex_from to vertex_to
     cudaError_t err = cudaSuccess;

    // Read the graph file
    if (argc != 3) {
        fprintf(stderr,"Please include the path to the graph file and the number of maximum iterations\n");
            exit(-1);
    } 

     FILE * fp;
     if ((fp = fopen(args[1], "r")) == NULL) {
         fprintf(stderr,"ERROR: Can not open input file.\n");
         exit(-1);
     }

     int n_iterations = atoi(args[2]);

     // Count the number of vertices O(n) time
     while (fscanf(fp, "%u %u", &vertex_from, &vertex_to) != EOF) {
        if (vertex_from > n_vertices) {
            n_vertices = vertex_from;
        } else if (vertex_to > n_vertices) {
            n_vertices = vertex_to;
        }
      	n_edges++;
    }
    n_vertices++;
    printf("Total number of vertices in the graph : %d \n",n_vertices);

    // Count the number of outlinks for each vertice
    unsigned int * outgoingLinks = (unsigned int *) calloc(n_vertices,sizeof(unsigned int));    
    fseek(fp,0L, SEEK_SET); // Sets the file position of the stream to 0 (beginging of the file)
    while(fscanf(fp,"%u %u", &vertex_from, &vertex_to) != EOF) {
        outgoingLinks[vertex_from] += 1;
    }

    // Construct the matrix
    float *matrix;
    matrix = (float *)calloc(n_vertices * n_vertices, sizeof(float)); 
    fseek(fp,0L, SEEK_SET); // Sets the file position of the stream to 0 (beginging of the file)
    while (fscanf(fp, "%u %u", &vertex_from, &vertex_to) != EOF) {
        int i = vertex_to;
        int j = vertex_from;
        // printf("Vertex %d : outgoing weights: %f \n", j, (float)1/outgoingLinks[j]);
        /*
        *   Note: Because BLAS uses internally column-major order storage, the matrix is transposed.
        *   Ref: https://solarianprogrammer.com/2012/05/31/matrix-multiplication-cuda-cublas-curand-thrust/
        *        https://www.wikiwand.com/en/Row-_and_column-major_order#/Column-major_order
        */
        // matrix[i*n_vertices + j] = (float)1/outgoingLinks[j]; 
        matrix[j*n_vertices + i] = (float)0.85/outgoingLinks[j];             
    }
    printf("Current matrix is : \n");
    printMatrix(matrix, n_vertices);

    // Initialize the pageRank vector and next pageRank vector
    float *pageRank = (float *)malloc(n_vertices * sizeof(float));
    float *nextPagerank = (float *)calloc(n_vertices, sizeof(float));
    float *addition = (float *)malloc(n_vertices * sizeof(float));
    float value = (float) 1 / n_vertices;
    float add = (float)(1-0.85)/n_vertices;
    for(int i=0; i<n_vertices; i++) {
        pageRank[i] = value;
        addition[i] = add;
    }
    printf("Current PageRank is : \n");    
    printPageRank(pageRank, n_vertices);
    // printf("Addition: \n");
    // printPageRank(addition, n_vertices);    

    // cudaDeviceSynchronize();
    // printf("\n Start allocating memory on device and recording the start time \n");
    // cudaEvent_t start;
    // cudaError_t error;
    // error = cudaEventCreate(&start);
    // if (error != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
    //     exit(EXIT_FAILURE);
    // }

    // cudaEvent_t stop;
    // error = cudaEventCreate(&stop);

    // if (error != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
    //     exit(EXIT_FAILURE);
    // }

    // // Record the start event
    // error = cudaEventRecord(start, NULL);

    // if (error != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
    //     exit(EXIT_FAILURE);
    // }

    clock_t startTime = clock();
    cycles_to_build = startTime - startBuildTime;

    // Allocat memory on GPU
    float *d_matrix, *d_pageRank, *d_nextPagerank, *d_addition;
    // thrust::device_vector<float> d_matrix(n_vertices * n_vertices), d_pageRank(n_vertices * n_vertices), d_nextPagerank(n_vertices * 1);
    err = cudaMalloc(&d_matrix, n_vertices * n_vertices * sizeof(float));
    err = cudaMalloc(&d_pageRank, n_vertices * sizeof(float));
    err = cudaMalloc(&d_nextPagerank, n_vertices * sizeof(float));

    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    for(int i=0; i<n_iterations; i++) {
        
        // Copy memory from CPU to GPU    
        err = cudaMemcpy(d_matrix, matrix, n_vertices*n_vertices*sizeof(float), cudaMemcpyHostToDevice);
        err = cudaMemcpy(d_pageRank, pageRank, n_vertices*sizeof(float), cudaMemcpyHostToDevice);
        
        // Matrix Multiplication        
        gpu_blas_mmul(d_matrix, d_pageRank, d_nextPagerank, n_vertices, n_vertices, n_vertices, n_iterations, handle);
        cudaDeviceSynchronize();

        // Add addition vector
        const float alphaI = 1;
        err = cudaMalloc(&d_addition, n_vertices * sizeof(float));
        err = cudaMemcpy(d_addition, addition, n_vertices*sizeof(float), cudaMemcpyHostToDevice);    
        cublasSaxpy(handle, n_vertices, &alphaI, d_addition, 1, d_nextPagerank, 1);
        cudaDeviceSynchronize();

        // Copy the result from GPU back to CPU
        err = cudaMemcpy(nextPagerank,d_nextPagerank, n_vertices * sizeof(float), cudaMemcpyDeviceToHost);
        float *temp = pageRank;
        pageRank = nextPagerank;
        nextPagerank = temp;
    }

    clock_t endTime = clock();


    // Destroy the handle
    cublasDestroy(handle);


    printf("The next PageRank is : \n");
    printPageRank(pageRank, n_vertices);
    // printf("Addition: \n");    
    // printPageRank(addition, n_vertices);    

    // // Record the stop event
    // error = cudaEventRecord(stop, NULL);
    // if (error != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
    //     exit(EXIT_FAILURE);
    // }

    // // Wait for the stop event to complete
    // error = cudaEventSynchronize(stop);

    // if (error != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
    //     exit(EXIT_FAILURE);
    // }

    // float msecTotal = 0.0f;
    // error = cudaEventElapsedTime(&msecTotal, start, stop);
    
    // if (error != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
    //     exit(EXIT_FAILURE);
    // }
    // printf("Time= %.3f msec\n",msecTotal);

    cycles_to_calc = endTime - startTime;
    int build_milli = cycles_to_build * 1000 / CLOCKS_PER_SEC;
    int calc_milli = cycles_to_calc * 1000 / CLOCKS_PER_SEC;
    printf("Time to build: %d seconds, %d milliseconds\n",build_milli/1000, build_milli%1000);
    printf("Time to calc: %d seconds, %d milliseconds\n",calc_milli/1000, calc_milli%1000);

    // long double calc_msec = cycles_to_calc;
    // long double build_msec = cycles_to_build;
    // printf("Time to build: %.32f milliseconds\n", build_msec);
    // printf("Time to calc: %.32f milliseconds\n", calc_msec);

    // Free GPU memory
    cudaFree(d_matrix);
    cudaFree(d_pageRank);
    cudaFree(d_nextPagerank);

    // Free CPU memeory
    free(matrix);
    free(pageRank);
    free(nextPagerank);

    return 0;
}

// CUDA BLAS matrixmultiplication 
// REF:https://solarianprogrammer.com/2012/05/31/matrix-multiplication-cuda-cublas-curand-thrust/
float* gpu_blas_mmul(float *A, float *B, float *C, int m, int k, int n, int n_iterations, cublasHandle_t handle) {
    //  gpu_blas_mmul(d_matrix, d_pageRank, d_nextPagerank, d_addition, n_vertices, n_vertices, n_vertices, n_iterations, handle);
    int lda=m,ldb=k,ldc=m;
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;

    // Do the actual multiplication
    // float addition = (1-dampling)/m;
    // for(int i=0; i<n_iterations; i++) {
        // Formula is   C = addition * I + A * B
        //   which is  next_pageRank = (1-d)/N * I + matrix * pageRank 
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        // cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, 2);
        // if (err != CUBLAS_STATUS_SUCCESS)
        // {
        //     printf("Error happened when doing matrix multiplication\n");
        //     exit(EXIT_FAILURE);
        // }
        
        // cudaDeviceSynchronize();

        // err = cublasSaxpy(handle, m, &alphaI, d_addition, 1, C, 1);


        // if (err != CUBLAS_STATUS_SUCCESS)
        // {
        //     printf("Error happened when adding vector to matrix\n");
        //     exit(EXIT_FAILURE);
        // }

        // cudaDeviceSynchronize();

        // float* temp = B;
        // B = C;
        // C = temp;
    // }

    return C;
}

// Print the pagerank
void printPageRank(float* pageRank, int n_vertices) {
    for(int i=0; i<n_vertices; i++) {
        // printf("Vertex: %d PageRank: %.26f \n", i, pageRank[i]);        
        if(pageRank[i] != 0) {
            printf("Vertex: %d PageRank: %.26f \n", i, pageRank[i]);
        }
    } 
    return;
}

// Print the matrix
void printMatrix(float* matrix, int n_vertices) {
    for(int i=0; i<n_vertices; i++) {
        printf("Vertex: %d ", i);
        for(int j=0; j<n_vertices; j++) {
            // Note: print the transposed matrix beacuse of Column-major order
            // printf("%.3f  ", matrix[i*n_vertices + j]);
            // printf("%.9f  ", matrix[j*n_vertices + i]);
            if(matrix[j*n_vertices + i] != 0) {
                printf("%.9f  ", matrix[j*n_vertices + i]);
            }
        }
        printf("\n");
    }
    return;
}

// Get the abs of a float number
float abs_float(float in) {
    if (in >= 0)
      return in;
    else
      return -in;
}