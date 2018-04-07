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
        matrix[i*n_vertices + j] = (float)1/outgoingLinks[j]; 
    }
    printf("Current matrix is : \n");
    printMatrix(matrix, n_vertices);

    // Initialize the pageRank vector and next pageRank vector
    float *pageRank = (float *)malloc(n_vertices * sizeof(float));
    float *nextPagerank = (float *)calloc(n_vertices, sizeof(float));
    for(int i=0; i<n_vertices; i++) {
        pageRank[i] = (float) 1 / n_vertices;
    }
    printf("Current PageRank is : \n");    
    printPageRank(pageRank, n_vertices);

    clock_t startTime = clock();
    cycles_to_build = startTime - startBuildTime;


    for(int i=0; i<n_iterations; i++) {
        // Matrix Multiplication
        for(int i=0; i<n_vertices; i++) {
            float sum = 0;
            for(int j=0; j<n_vertices; j++) {
                sum += 0.85 * matrix[i*n_vertices + j] * pageRank[j];
            }
            nextPagerank [i] = sum + (float)(1-0.85)/n_vertices;
        } 
        float *temp = pageRank;
        pageRank = nextPagerank;
        nextPagerank = temp;
    }

    clock_t endTime = clock();

    printf("The next PageRank is : \n");
    printPageRank(pageRank, n_vertices);

    cycles_to_calc = endTime - startTime;
    int build_milli = cycles_to_build * 1000 / CLOCKS_PER_SEC;
    int calc_milli = cycles_to_calc * 1000 / CLOCKS_PER_SEC;
    printf("Time to build: %d seconds, %d milliseconds\n",build_milli/1000, build_milli%1000);
    printf("Time to calc: %d seconds, %d milliseconds\n",calc_milli/1000, calc_milli%1000);

    // cycles_to_calc = endTime - startTime;
    // long double calc_msec = cycles_to_calc;
    // printf("Time to calc: %.32f milliseconds\n", calc_msec);

    // Free CPU memeory
    free(matrix);
    free(pageRank);
    free(nextPagerank);

    return 0;
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