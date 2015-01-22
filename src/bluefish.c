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

#include <mpi.h>
#include "bluefish.h"

// define global variables that were declared in header file
dom_struct *dom;
dom_struct **_dom;
dom_struct *Dom;
dom_struct DOM;
real *p0;
real *p;
real **_p0;
real **_p;
real **_rhs_p;
int *flag_u;
int **_flag_u;
int *flag_v;
int **_flag_v;
int *flag_w;
int **_flag_w;
int pp_max_iter;
real pp_residual;
BC bc;

int main(int argc, char *argv[]) {

  printf("N%d >> Running Bluefish_0.1...\n", rank);

  // read device configuration file
  printf("N%d >> Reading the device configuration file...\n", rank);
  fflush(stdout);
  devs_read_input();
  //devs_show_config();
 
  // start mpi
  printf("N%d >> Starting MPI...\n", rank);
  mpi_startup(argc, argv);
  MPI_Barrier(MPI_COMM_WORLD);
 
  // read simulation input configuration file
  printf("N%d >> Reading the domain input file...\n", rank);
  domain_read_input();
  MPI_Barrier(MPI_COMM_WORLD);

  // initialize the subdomain map for this node
  printf("N%d >> Initializing subdomain map...\n", rank);
  domain_map();
  MPI_Barrier(MPI_COMM_WORLD);
#ifdef DEBUG
  printf("N%d >> Writing subdomain map debug file...\n", rank);
  domain_map_write_config();
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  // initialize VTK writer
  printf("N%d >> Initializing VTK writer...\n", rank);
  init_VTK();
  MPI_Barrier(MPI_COMM_WORLD);

  // allocate node memory
  printf("N%d >> Allocating MPI subdomain memory...\n", rank);
  mpi_dom_malloc();
  MPI_Barrier(MPI_COMM_WORLD);

  // initialize node memory
  printf("N%d >> Initializing MPI subdomain memory...\n", rank);
  mpi_dom_init();
  MPI_Barrier(MPI_COMM_WORLD);

  // allocate GPU memory
  printf("N%d >> Allocating GPU subdomain memory...\n", rank);
  fflush(stdout);
  cuda_dom_malloc();
  MPI_Barrier(MPI_COMM_WORLD);

  // push memory from node to GPUs
  printf("N%d >> Pushing memory from node to GPUs...\n", rank);
  fflush(stdout);
  copy_node_to_devs();
  MPI_Barrier(MPI_COMM_WORLD);

  // pull memory from GPUs to node
  printf("N%d >> Pulling memory from GPUs to node...\n", rank);
  fflush(stdout);
  copy_devs_to_node();
  MPI_Barrier(MPI_COMM_WORLD);

  // write VTK file
  printf("N%d >> Writing VTK output files...\n", rank);
  out_VTK();
  MPI_Barrier(MPI_COMM_WORLD);

  // clean up GPUs
  printf("N%d >> Freeing GPU subdomain memory...\n", rank);
  fflush(stdout);
  cuda_dom_free();
  MPI_Barrier(MPI_COMM_WORLD);

  // clean up host
  printf("N%d >> Freeing MPI subdomain memory...\n", rank);
  mpi_dom_free();
  MPI_Barrier(MPI_COMM_WORLD);
  printf("N%d >> Cleaning up domain...\n", rank);
  domain_map_clean();
  MPI_Barrier(MPI_COMM_WORLD);
  devs_clean();
  MPI_Barrier(MPI_COMM_WORLD);

  // end mpi
  mpi_cleanup();

  printf("N%d >> Bluefish_0.1 complete.\n", rank);
}
