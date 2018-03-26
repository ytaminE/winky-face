#include <stdlib.h>
#include <stdio.h>
#include <time.h>

int main(int argc, char **argv){
  if(argc != 4) {
      printf("Wrong num of args. Need: [output filename] [# of vertices] [max # of edge per vertex]\n");
      return EXIT_FAILURE;
  }

  clock_t start = clock();

  FILE* fp = fopen(argv[1], "w");
  if(fp == NULL) {
     perror("File opening failed");
     return EXIT_FAILURE;
  }

  int v_to;
  int n_vertices = atoi(argv[2]);
  int max_edges = atoi(argv[3]);

  srand(time(0));
  for(int i = 0; i < n_vertices; i++) {
    int has_edge = rand() % 4;
    if(has_edge > 0 || i == 0 || i == n_vertices -1) {
      int num_of_edges = (rand() % max_edges) + 1;
      for(int j = 0; j < num_of_edges;j++) {
        v_to = rand() % n_vertices;
        if (v_to != i) {
          fprintf(fp, "%d %d\n", i, v_to);
        }
      }

    }
  }
  fclose(fp);

  clock_t end = clock();
  double graph_generating_time = ((double)(end-start))/CLOCKS_PER_SEC;
  printf("Time to generate graph: %.2f seconds, %.f milliseconds\n", graph_generating_time, graph_generating_time*1000);

  return EXIT_SUCCESS;
}
