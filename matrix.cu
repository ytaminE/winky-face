#include <stdio.h>
#include <cuda_profiler_api.h>

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
     int i,j;
     unsigned int n_vertices = 0;                   // number of vertices
     unsigned int n_edges = 0;                      // number of edges
     unsigned int vertex_from = 0, vertex_to = 0;   // edge from vertex_from to vertex_to
     vertex * vertices;                             // a list of vertices in the graph
 
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
        if (vertex_from > n_vertices)
            n_vertices = vertex_from;
        else if (vertex_to > n_vertices)
            n_vertices = vertex_to;
    }
    n_vertices++;
    printf("Total number of vertices in the graph : %d, ",n_vertices);

}