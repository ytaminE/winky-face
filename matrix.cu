#include <stdio.h>
#include <cuda_profiler_api.h>

void printMatrix(float* matrix, int n_vertices);

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
        matrix[i*n_vertices + j] = (float)1/outgoingLinks[j];
    }
    // printMatrix(matrix, n_vertices);

}


// Print the matrix
void printMatrix(float* matrix, int n_vertices) {
    for(int i=0; i<n_vertices; i++) {
        printf("Vertex: %d ", i);
        for(int j=0; j<n_vertices; j++) {
            printf("%.2f  ", matrix[i*n_vertices + j]);
        }
        printf("\n");
    }
    return;
}
