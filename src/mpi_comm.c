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

#include "mpi_comm.h"

int nproc;
int rank;

void mpi_startup(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if(nproc != nnodes) {
    printf("MPI configuration does not match device configuration.\n");
    exit(EXIT_FAILURE);
  }
}

void mpi_dom_malloc(void)
{
  p0 = (real*) malloc(Dom[rank].Gcc.s3b * sizeof(real));
  p = (real*) malloc(Dom[rank].Gcc.s3b * sizeof(real));
  flag_u = (int*) malloc(Dom[rank].Gfx.s3b * sizeof(real));
  flag_v = (int*) malloc(Dom[rank].Gfy.s3b * sizeof(real));
  flag_w = (int*) malloc(Dom[rank].Gfz.s3b * sizeof(real));
}

void mpi_dom_init(void)
{
  int i, j, k;    // iterators

  // p
  for(k = Dom[rank].Gcc._ksb; k <= Dom[rank].Gcc._keb; k++) {
    for(j = Dom[rank].Gcc._jsb; j <= Dom[rank].Gcc._jeb; j++) {
      for(i = Dom[rank].Gcc._isb; i <= Dom[rank].Gcc._ieb; i++) {
        p[i + j*Dom[rank].Gcc.s1b + k*Dom[rank].Gcc.s2b] = rank;
        p0[i + j*Dom[rank].Gcc.s1b + k*Dom[rank].Gcc.s2b] = 0.;
      }
    }
  }

  // u
  for(k = Dom[rank].Gfx._ksb; k <= Dom[rank].Gfx._keb; k++) {
    for(j = Dom[rank].Gfx._jsb; j <= Dom[rank].Gfx._jeb; j++) {
      for(i = Dom[rank].Gfx._isb; i <= Dom[rank].Gfx._ieb; i++) {
        flag_u[i + j*Dom[rank].Gfx.s1b + k*Dom[rank].Gfx.s2b] = 0.;
      }
    }
  }

  // v
  for(k = Dom[rank].Gfy._ksb; k <= Dom[rank].Gfy._keb; k++) {
    for(j = Dom[rank].Gfy._jsb; j <= Dom[rank].Gfy._jeb; j++) {
      for(i = Dom[rank].Gfy._isb; i <= Dom[rank].Gfy._ieb; i++) {
        flag_v[i + j*Dom[rank].Gfy.s1b + k*Dom[rank].Gfy.s2b] = 0.;
      }
    }
  }

  // w
  for(k = Dom[rank].Gfz._ksb; k <= Dom[rank].Gfz._keb; k++) {
    for(j = Dom[rank].Gfz._jsb; j <= Dom[rank].Gfz._jeb; j++) {
      for(i = Dom[rank].Gfz._isb; i <= Dom[rank].Gfz._ieb; i++) {
        flag_w[i + j*Dom[rank].Gfz.s1b + k*Dom[rank].Gfz.s2b] = 0.;
      }
    }
  }
}

void mpi_dom_free(void)
{
  free(p0);
  free(p);
  free(flag_u);
  free(flag_v);
  free(flag_w);
}

void mpi_cleanup(void)
{
  MPI_Finalize();
}
