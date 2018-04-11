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

    // clock_t startBuildTime = clock();

    // Initialize the graph context
     unsigned int n_vertices = 0;                   // number of vertices
     unsigned int n_edges = 0;                      // number of edges
     unsigned int vertex_from = 0, vertex_to = 0;   // edge from vertex_from to vertex_to

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

    clock_t startBuildTime = clock();

    // Count the number of outlinks for each vertice
    unsigned int * outgoingLinks = (unsigned int *) calloc(n_vertices,sizeof(unsigned int));    
    fseek(fp,0L, SEEK_SET); // Sets the file position of the stream to 0 (beginging of the file)
    while(fscanf(fp,"%u %u", &vertex_from, &vertex_to) != EOF) {
        outgoingLinks[vertex_from] += 1;
    }

    // initialize matrix
    // cusp::array2d<float, cusp::host_memory> matrixArray(n_vertices,n_vertices);
    cusp::coo_matrix<int,float, cusp::host_memory> matrix(n_vertices, n_vertices, n_edges);
    fseek(fp,0L, SEEK_SET); // Sets the file position of the stream to 0 (beginging of the file)
    int count = 0;
    while (fscanf(fp, "%u %u", &vertex_from, &vertex_to) != EOF) {
        int i = vertex_to;
        int j = vertex_from;
        // printf("Vertex %d : outgoing weights: %f \n", j, (float)1/outgoingLinks[j]);
        // matrixArray(i, j) = (float)1/outgoingLinks[j];
        matrix.row_indices[count] = i; matrix.column_indices[count] = j; matrix.values[count] = (float)1/outgoingLinks[j];
        count++;
    }
    // printf("Matrix is : \n");    
    // cusp::print(matrixArray);
    // cusp::print(matrix);


    // initialize input vector
    float value = (float) 1 / n_vertices;
    float add = (float)(1-0.85)/n_vertices;
    cusp::array1d<float, cusp::host_memory> pageRank(n_vertices);
    cusp::array1d<float, cusp::host_memory> addition(n_vertices);    
    for(int i=0; i<n_vertices; i++) {
      pageRank[i] = value;
      addition[i] = add;
    }
    // printf("Current PageRank is : \n");    
    // cusp::print(pageRank);


    clock_t startTime = clock();
    cycles_to_build = startTime - startBuildTime;

    // allocate output vector
    cusp::array1d<float, cusp::host_memory> nextPagerank(n_vertices);

    // Device
    cusp::coo_matrix<int,float, cusp::device_memory> d_matrix(matrix);
    // cusp::convert(matrix, d_matrix);
    // cusp::array2d<float, cusp::device_memory> d_matrix(matrix);
    cusp::array1d<float, cusp::device_memory> d_pageRank(pageRank);
    cusp::array1d<float, cusp::device_memory> d_addition(addition);
    cusp::array1d<float, cusp::device_memory> d_nextPagerank(nextPagerank);


    // GPU
    for(int i=0; i<n_iterations; i++) {
        // compute y = A * x
        cusp::multiply(d_matrix, d_pageRank, d_nextPagerank);
        // compute y += 1.5*x
        cusp::blas::axpy(d_addition, d_nextPagerank, 1.0);
        d_pageRank.swap(d_nextPagerank);
    }

    // // CPU
    // for(int i=0; i<n_iterations; i++) {
    //     // compute y = A * x
    //     cusp::multiply(matrixArray, pageRank, nextPagerank);
    //     // compute y += 1.5*x
    //     cusp::blas::axpy(addition, nextPagerank, 1.0);
    //     pageRank.swap(nextPagerank);
    // }


    cusp::array1d<float, cusp::host_memory> res = d_pageRank;

    clock_t endTime = clock();

    // print nextPagerank
    // printf("Final GPU PageRank is : \n");    
    // cusp::print(res);
    // printf("Final CPU PageRank is : \n");    
    // cusp::print(pageRank);

    cycles_to_calc = endTime - startTime;
    int build_milli = cycles_to_build * 1000 / CLOCKS_PER_SEC;
    int calc_milli = cycles_to_calc * 1000 / CLOCKS_PER_SEC;
    printf("Time to build: %d seconds, %d milliseconds\n",build_milli/1000, build_milli%1000);
    printf("Time to calc: %d seconds, %d milliseconds\n",calc_milli/1000, calc_milli%1000);
    printf("Count : %d \n", count);
    printf("Number of edges : %d \n", n_edges);
    return 0;
    
}