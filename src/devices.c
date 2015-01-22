/*******************************************************************************
 ******************************** BLUEFISH-1.0 *********************************
 *******************************************************************************
 *
 *  Copyright 2012 - 2014 Adam Sierakowski, The Johns Hopkins University
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  Please contact the Johns Hopkins University to use Bluefish for
 *  commercial and/or for-profit applications.
 ******************************************************************************/

#include "devices.h"

int nnodes;
node_struct *nodes;

void devs_read_input(void)
{
  int i, j;     // iterators
  int devcount = 0;

  int fret = 0;
  fret = fret;  // prevent compiler warning

  // open configuration file for reading
  char fname[FILE_NAME_SIZE];
  sprintf(fname, "%s/input/devs.config", ROOT_DIR);
  FILE *infile = fopen(fname, "r");
  if(infile == NULL) {
    fprintf(stderr, "Could not open file %s\n", fname);
    exit(EXIT_FAILURE);
  }

  // read node list
  fret = fscanf(infile, "n_nodes %d\n", &nnodes);
  // allocate node list
  nodes = (node_struct*) malloc(nnodes * sizeof(node_struct));

  // read nnodes nodes
  for(i = 0; i < nnodes; i++) {
    nodes[i].devstart = devcount;
    fret = fscanf(infile, "\n");
    fret = fscanf(infile, "n_devs %d\n", &nodes[i].ndevs);
    fret = fscanf(infile, "name %s\n", nodes[i].name);
    for(j = 0; j < nodes[i].ndevs; j++) {
      fret = fscanf(infile, "%d ", &nodes[i].devs[j]);
      devcount++;
    }
    fret = fscanf(infile, "\n");
  }
}

void devs_show_config(void)
{
  int i, j;     // iterators

  printf("Device configuration:\n");
  for(i = 0; i < nnodes; i++) {
    printf("  Node %d:\n", i);
    printf("    Name: %s\n", nodes[i].name);
    printf("    dev_nums (%d):", nodes[i].ndevs);
    for(j = 0; j < nodes[i].ndevs; j++)
      printf(" %d", nodes[i].devs[j]);
    printf("\n");
    printf("    devstart = %d\n", nodes[i].devstart);
  }
  printf("\n");
}

void devs_clean(void) {
  free(nodes);
}
