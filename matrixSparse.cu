#include <stdio.h>
#include <iostream>
#include <cuda_profiler_api.h>
#include <cusp/multiply.h>
#include <cusp/array2d.h>
#include <cusp/print.h>

// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>

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


    // // Construct the matrix
    // float *matrix;
    // matrix = (float *)calloc(n_vertices * n_vertices, sizeof(float)); 
    // fseek(fp,0L, SEEK_SET); // Sets the file position of the stream to 0 (beginging of the file)
    // while (fscanf(fp, "%u %u", &vertex_from, &vertex_to) != EOF) {
    //     int i = vertex_to;
    //     int j = vertex_from;
    //     // printf("Vertex %d : outgoing weights: %f \n", j, (float)1/outgoingLinks[j]);
    //     /*
    //     *   Note: Because BLAS uses internally column-major order storage, the matrix is transposed.
    //     *   Ref: https://solarianprogrammer.com/2012/05/31/matrix-multiplication-cuda-cublas-curand-thrust/
    //     *        https://www.wikiwand.com/en/Row-_and_column-major_order#/Column-major_order
    //     */
    //     // matrix[i*n_vertices + j] = (float)1/outgoingLinks[j]; 
    //     matrix[j*n_vertices + i] = (float)0.85/outgoingLinks[j];             
    // }
    // printf("Current matrix is : \n");
    // printMatrix(matrix, n_vertices);


    // initialize matrix
    cusp::array2d<float, cusp::host_memory> matrix(n_vertices,n_vertices);
    fseek(fp,0L, SEEK_SET); // Sets the file position of the stream to 0 (beginging of the file)
    while (fscanf(fp, "%u %u", &vertex_from, &vertex_to) != EOF) {
        int i = vertex_to;
        int j = vertex_from;
        // printf("Vertex %d : outgoing weights: %f \n", j, (float)1/outgoingLinks[j]);
        matrix(i, j) = (float)1/outgoingLinks[j];
    }
    printf("Matrix is : \n");    
    cusp::print(matrix);



    // // Initialize the pageRank vector and next pageRank vector
    // float *pageRank = (float *)malloc(n_vertices * sizeof(float));
    // float *nextPagerank = (float *)calloc(n_vertices, sizeof(float));
    // float *addition = (float *)malloc(n_vertices * sizeof(float));
    // float value = (float) 1 / n_vertices;
    // float add = (float)(1-0.85)/n_vertices;
    // for(int i=0; i<n_vertices; i++) {
    //     pageRank[i] = value;
    //     addition[i] = add;
    // }
    // printf("Current PageRank is : \n");    
    // printPageRank(pageRank, n_vertices);


    // initialize input vector
    float value = (float) 1 / n_vertices;
    float add = (float)(1-0.85)/n_vertices;
    cusp::array1d<float, cusp::host_memory> pageRank(n_vertices);
    cusp::array1d<float, cusp::host_memory> addition(n_vertices);    
    for(int i=0; i<n_vertices; i++) {
      pageRank[i] = value;
      addition[i] = add;
    }
    printf("Current PageRank is : \n");    
    cusp::print(pageRank);



    // allocate output vector
    cusp::array1d<float, cusp::host_memory> nextPagerank(n_vertices);

    // compute y = A * x
    cusp::multiply(matrix, pageRank, nextPagerank);
    
    // print nextPagerank
    printf("Next PageRank is : \n");    
    cusp::print(nextPagerank);
    return 0;
    
}