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

/****h* Bluefish/bluefish
 * NAME
 *  bluefish
 * FUNCTION
 *  Bluefish main execution code and global variable declarations.
 ******
 */

#ifndef _BLUEFISH_H
#define _BLUEFISH_H

#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef DOUBLE
  typedef double real;
#else
  typedef float real;
#endif

// devices.h included at end of this file (it needs definitions there)
#include "domain.h"
#include "mpi_comm.h"
#include "recorder.h"
#include "vtk.h"

/****d* bluefish/PI
 * NAME
 *  PI
 * TYPE
 */
#define PI 3.1415926535897932385
/*
 * PURPOSE
 *  Define the constant pi.
 ******
 */

/****d* blueblottle/DIV_ST
 * NAME
 *  DIV_ST
 * TYPE
 */
#define DIV_ST 1e-7
/*
 * PURPOSE
 *  Define a value to use for fudging the value of theta when a division by
 *  sin(theta) occurs.
 ******
 */

/****d* bluefish/FILE_NAME_SIZE
 * NAME
 *  FILE_NAME_SIZE
 * TYPE
 */
#define FILE_NAME_SIZE 256
/*
 * PURPOSE
 *  Define the maximum length of a file name.
 ******
 */

/****d* bluefish/CHAR_BUF_SIZE
 * NAME
 *  CHAR_BUF_SIZE
 * TYPE
 */
#define CHAR_BUF_SIZE 256
/*
 * PURPOSE
 *  Define the maximum length of a character buffer read.
 ******
 */

/****d* bluefish/ROOT_DIR
 * NAME
 *  ROOT_DIR
 * TYPE
 */
#define ROOT_DIR "."
/*
 * PURPOSE
 *  Define the root directory for the project.
 ******
 */

/****d* bluefish/OUTPUT_DIR
 * NAME
 *  OUTPUT_DIR
 * TYPE
 */
#define OUTPUT_DIR "./output/"
/*
 * PURPOSE
 *  Define the output directory for the project.
 ******
 */

/****d* bluefish/INPUT_DIR
 * NAME
 *  INPUT_DIR 
 * TYPE
 */
#define INPUT_DIR "./input/"
/*
 * PURPOSE
 *  Define the input directory for the project.
 ******
 */

/****d* bluefish/DOM_BUF
 * NAME
 *  DOM_BUF
 * TYPE
 */
#define DOM_BUF 1
/*
 * PURPOSE
 *  Define the size of the domain boundary condition ghost cell buffer (the
 *  number of ghost cells on one side of a give domain direction).
 ******
 */

/****d* bluefish/PERIODIC
 * NAME
 *  PERIODIC
 * TYPE
 */
#define PERIODIC 0
/*
 * PURPOSE
 *  Define the periodic boundary condition type.
 ******
 */

/****d* bluefish/DIRICHLET
 * NAME
 *  DIRICHLET
 * TYPE
 */
#define DIRICHLET 1
/*
 * PURPOSE
 *  Define the Dirichlet boundary condition type.
 ******
 */

/****d* bluefish/NEUMANN
 * NAME
 *  NEUMANN
 * TYPE
 */
#define NEUMANN 2
/*
 * PURPOSE
 *  Define the Neumann boundary condition type.
 ******
 */

/****d* bluefish/WEST
 * NAME
 *  WEST
 * TYPE
 */
#define WEST 0
/*
 * PURPOSE
 *  Define the West boundary.
 ******
 */

/****d* bluefish/EAST
 * NAME
 *  EAST
 * TYPE
 */
#define EAST 1
/*
 * PURPOSE
 *  Define the East boundary.
 ******
 */

/****d* bluefish/SOUTH
 * NAME
 *  SOUTH
 * TYPE
 */
#define SOUTH 2
/*
 * PURPOSE
 *  Define the South boundary.
 ******
 */

/****d* bluefish/NORTH
 * NAME
 *  NORTH
 * TYPE
 */
#define NORTH 3
/*
 * PURPOSE
 *  Define the North boundary.
 ******
 */

/****d* bluefish/BOTTOM
 * NAME
 *  BOTTOM
 * TYPE
 */
#define BOTTOM 4
/*
 * PURPOSE
 *  Define the Bottom boundary.
 ******
 */

/****d* bluefish/TOP
 * NAME
 *  TOP
 * TYPE
 */
#define TOP 5
/*
 * PURPOSE
 *  Define the Top boundary.
 ******
 */

/****d* bluefish/MAX_THREADS_1D
 * NAME
 *  MAX_THREADS_1D
 * TYPE
 */
#define MAX_THREADS_1D 128
/*
 * PURPOSE
 *  Define the maximum number of threads per block on a CUDA device.  Must be
 *  hardcoded, but does depend on the particular device being used.
 ******
 */

/****d* bluefish/MAX_THREADS_DIM
 * NAME
 *  MAX_THREADS_DIM
 * TYPE
 */
#define MAX_THREADS_DIM 16
/*
 * PURPOSE
 *  Define the maximum number of threads per dimension per block on a CUDA
 *  device.  Must be hardcoded, but does depend on the particular device being
 *  used.
 ******
 */

/****v* bluefish/dom
 * NAME
 *  dom
 * TYPE
 */
extern dom_struct *dom;
/*
 * PURPOSE
 *  Carry GPU domain decomposition subdomain information.  Contains an array
 *  of dom_struct structs that each represent one subdomain.
 ******
 */

/****v* bluefish/_dom
 * NAME
 *  _dom
 * TYPE
 */
extern dom_struct **_dom;
/*
 * PURPOSE
 *  CUDA device analog for dom.  It contains pointers to arrays containing
 *  the subdomain fields on which each device operates.
 ******
 */

/****v* bluefish/Dom
 * NAME
 *  Dom
 * TYPE
 */
extern dom_struct *Dom;
/*
 * PURPOSE
 *  Carry MPI node domain decomposition subdomain information. Contains an array
 *  of dom_struct structs that each represent one subdomain.
 ******
 */

/****v* bluefish/DOM
 * NAME
 *  DOM
 * TYPE
 */
extern dom_struct DOM;
/*
 * PURPOSE
 *  Carry global domain information.
 ******
 */

/****v* bluefish/p0
 * NAME
 *  p0
 * TYPE
 */
extern real *p0;
/*
 * PURPOSE
 *  Pressure field vector (grid type Gcc; x-component varies first, then
 *  y-component, then z-component). This is the previous stored time step.
 ******
 */

/****v* bluefish/p
 * NAME
 *  p
 * TYPE
 */
extern real *p;
/*
 * PURPOSE
 *  Pressure field vector (grid type Gcc; x-component varies first, then
 *  y-component, then z-component).
 ******
 */

/****v* bluefish/_p0
 * NAME
 *  _p0
 * TYPE
 */
extern real **_p0;
/*
 * PURPOSE
 *  CUDA device analog for p0.  It contains pointers to arrays containing
 *  the subdomain fields on which each device operates.
 ******
 */

/****v* bluefish/_p
 * NAME
 *  _p
 * TYPE
 */
extern real **_p;
/*
 * PURPOSE
 *  CUDA device analog for p.  It contains pointers to arrays containing
 *  the subdomain fields on which each device operates.
 ******
 */

/****v* bluefish/_rhs_p
 * NAME
 *  _rhs_p
 * TYPE
 */
extern real **_rhs_p;
/*
 * PURPOSE
 *  CUDA device array for storing the right-hand side of the pressure-Poisson
 *  problem.  It contains pointers to arrays containing the subdomain fields
 *  on which each device operates.
 ******
 */

/****v* particle/flag_u
 * NAME
 *  flag_u
 * TYPE
 */
extern int *flag_u;
/* 
 * PURPOSE
 *  Flag x-direction components of velocity field that are set as boundaries.
 ******
 */

/****v* particle/_flag_u
 * NAME
 *  _flag_u
 * TYPE
 */
extern int **_flag_u;
/*
 * PURPOSE
 *  CUDA device analog for flag_u.  It contains pointers to arrays containing
 *  the subdomain fields on which each device operates.
 ******
 */

/****v* particle/flag_v
 * NAME
 *  flag_v
 * TYPE
 */
extern int *flag_v;
/* 
 * PURPOSE
 *  Flag y-direction components of velocity field that are set as boundaries.
 ******
 */

/****v* particle/_flag_v
 * NAME
 *  _flag_v
 * TYPE
 */
extern int **_flag_v;
/*
 * PURPOSE
 *  CUDA device analog for flag_v.  It contains pointers to arrays containing
 *  the subdomain fields on which each device operates.
 ******
 */

/****v* particle/flag_w
 * NAME
 *  flag_w
 * TYPE
 */
extern int *flag_w;
/* 
 * PURPOSE
 *  Flag z-direction components of velocity field that are set as boundaries.
 ******
 */

/****v* particle/_flag_w
 * NAME
 *  _flag_w
 * TYPE
 */
extern int **_flag_w;
/*
 * PURPOSE
 *  CUDA device analog for flag_w.  It contains pointers to arrays containing
 *  the subdomain fields on which each device operates.
 ******
 */

/****v* bluefish/pp_max_iter
 * NAME
 *  pp_max_iter
 * TYPE
 */
extern int pp_max_iter;
/*
 * PURPOSE
 *  The maximum number of iterations for the pressure-Poisson problem solver.
 ******
 */

/****v* bluefish/pp_residual
 * NAME
 *  pp_residual
 * TYPE
 */
extern real pp_residual;
/*
 * PURPOSE
 *  The maximum desired residual for the pressure-Poisson problem solver.
 ******
 */

/****s* bluefish/BC
 * NAME
 *  BC
 * TYPE
 */
typedef struct BC {
  int pW;
  int pE;
  int pS;
  int pN;
  int pB;
  int pT;
} BC;
/*
 * PURPOSE
 *  Carry the type of boundary condition on each side of the domain.  Possible
 *  types include:
 *  * PERIODIC
 *  * DIRICHLET
 *  * NEUMANN
 *  * PRECURSOR
 *  If the boundary type is DIRICHLET or PRECURSOR, the value of the field
 *  variable on the boundary must be defined.
 * MEMBERS
 *  * pW -- the boundary condition type
 *  * pE -- the boundary condition type
 *  * pS -- the boundary condition type
 *  * pN -- the boundary condition type
 *  * pB -- the boundary condition type
 *  * pT -- the boundary condition type
 ******
 */

/****v* bluefish/bc
 * NAME
 *  bc
 * TYPE
 */
extern BC bc;
/*
 * PURPOSE
 *  Create an instance of the struct BC to carry boundary condition types.
 ******
 */

/****f* bluefish/cuda_dom_malloc()
 * NAME
 *  cuda_dom_malloc()
 * USAGE
 */
void cuda_dom_malloc(void);
/*
 * FUNCTION
 *  Allocate device memory reference pointers on host and device memory on
 *  device for the flow domain.
 ******
 */

/****f* bluefish/copy_node_to_devs()
 * NAME
 *  copy_node_to_devs()
 * USAGE
 */
void copy_node_to_devs(void);
/*
 * FUNCTION
 *  Copies the data on all nodes to the appropriate GPU (sub)domains.
 ******
 */

/****f* bluefish/copy_devs_to_node()
 * NAME
 *  copy_devs_to_node()
 * USAGE
 */
void copy_devs_to_node(void);
/*
 * FUNCTION
 *  Copies the data on all GPUs to the appropriate node (sub)domains.
 ******
 */

/****f* bluefish/cuda_dom_free()
 * NAME
 *  cuda_dom_free()
 * USAGE
 */
void cuda_dom_free(void);
/*
 * FUNCTION
 *  Free device memory for the domain on device and device memory reference
 *  pointers on host.
 ******
 */

#include "devices.h"    // this must be located at the end

#endif
