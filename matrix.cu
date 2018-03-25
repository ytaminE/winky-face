#include <stdio.h>
#include <iostream>
#include <cuda_profiler_api.h>
#include <cublas_v2.h>
// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>

float* gpu_blas_mmul(float *A, float *B, float *C, const float *I, int m, int k, int n, int n_iterations, cublasHandle_t handle);
void printPageRank(float* pageRank, int n_vertices);
void printMatrix(float* matrix, int n_vertices);
float abs_float(float in);

typedef struct vertex vertex;
 
struct vertex {
    unsigned int vertex_id;
    float pagerank;             // current pagerank
    float pagerank_next;        // used to store the next pagerank
    unsigned int n_successors;  // number of outlinks
    vertex ** successors;       // a list of successors
};

int main(int argc, char ** args) {
    // Initialize the cuda context
    cudaFree(0);  
    cudaProfilerStart();

    // Start CPU timer
    clock_t cycles_to_build, cycles_to_calc;
    clock_t start = clock();

    // Initialize the graph context
     unsigned int n_vertices = 0;                   // number of vertices
     unsigned int n_edges = 0;                      // number of edges
     unsigned int vertex_from = 0, vertex_to = 0;   // edge from vertex_from to vertex_to

    // Read the graph file
    if (argc != 2) {
        fprintf(stderr,"Please include the path to the graph file.\n");
            exit(-1);
    } 

     FILE * fp;
     if ((fp = fopen(args[1], "r")) == NULL) {
         fprintf(stderr,"ERROR: Can not open input file.\n");
         exit(-1);
     }

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
    // printf("Total number of vertices in the graph : %d \n",n_vertices);

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
    for(int i=0; i<n_vertices; i++) {
        pageRank[i] = value;
        addition[i] = 1;
    } 
    printf("Current PageRank is : \n");    
    printPageRank(pageRank, n_vertices);


    // Allocat memory on GPU
    float *d_matrix, *d_pageRank, *d_nextPagerank, *d_addition;
    // thrust::device_vector<float> d_matrix(n_vertices * n_vertices), d_pageRank(n_vertices * n_vertices), d_nextPagerank(n_vertices * 1);
    cudaMalloc(&d_matrix, n_vertices * n_vertices * sizeof(float));
    cudaMalloc(&d_pageRank, n_vertices * sizeof(float));
    cudaMalloc(&d_nextPagerank, n_vertices * sizeof(float));
    cudaMalloc(&d_addition, n_vertices * sizeof(float));

    // Copy memory from CPU to GPU
    cudaMemcpy(d_matrix, matrix, n_vertices*n_vertices*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pageRank, pageRank, n_vertices*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_addition, addition, n_vertices*sizeof(float), cudaMemcpyHostToDevice);

    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Matrix Multiplication
    // gpu_blas_mmul(thrust::raw_pointer_cast(&d_A[0]), thrust::raw_pointer_cast(&d_B[0]), thrust::raw_pointer_cast(&d_C[0]), nr_rows_A, nr_cols_A, nr_cols_B);
    int n_iterations = 4;
    d_nextPagerank = gpu_blas_mmul(d_matrix, d_pageRank, d_nextPagerank, d_addition, n_vertices, n_vertices, n_vertices, n_iterations, handle);

    // Destroy the handle
    cublasDestroy(handle);

    // Copy the result from GPU back to CPU
    cudaMemcpy(nextPagerank,d_nextPagerank, n_vertices * sizeof(float), cudaMemcpyDeviceToHost);
    printf("The next PageRank is : \n");
    printPageRank(nextPagerank, n_vertices);

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
float* gpu_blas_mmul(float *A, float *B, float *C, const float* I, int m, int k, int n, int n_iterations, cublasHandle_t handle) {
    // gpu_blas_mmul(d_matrix, d_pageRank, d_nextPagerank, n_vertices, n_vertices, n_vertices, handle);
    int lda=m,ldb=k,ldc=m;
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;

    // Do the actual multiplication
    float dampling = 0.85;
    float addition = (1-dampling)/m;
    // cublasSscal(handle, m*m, &dampling, A, 1); 
    for(int i=0; i<n_iterations; i++) {
        // Formula is   C = addition * I + A * B
        //   which is  next_pageRank = (1-d)/N * I + matrix * pageRank 
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        cublasSaxpy(handle, m, &addition, I, 1, C, 1);
        float* temp = B;
        B = C;
        C = temp;
    }

    return C;
}

// Print the pagerank
void printPageRank(float* pageRank, int n_vertices) {
    for(int i=0; i<n_vertices; i++) {
        printf("Vertex: %d PageRank: %.3f \n", i, pageRank[i]);
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
            printf("%.3f  ", matrix[j*n_vertices + i]);
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