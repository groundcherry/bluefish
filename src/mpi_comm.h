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

/****h* Bluefish/mpi_comm
 * NAME
 *  mpi_comm
 * FUNCTION
 *  Handle MPI communications.
 ******
 */

#ifndef _MPI_COMM_H
#define _MPI_COMM_H

#include "bluefish.h"

/****v* mpi_comm/nproc
 * NAME
 *  nproc
 * TYPE
 */
extern int nproc;
/*
 * PURPOSE 
 *  The number of MPI processes launched by mpiexec.
 ****** 
 */

/****v* mpi_comm/rank
 * NAME
 *  rank
 * TYPE
 */
extern int rank;
/*
 * PURPOSE
 *  The MPI process rank number.
 ******
 */

/****f* mpi_comm/mpi_startup()
 * NAME
 *  mpi_startup()
 * USAGE
 */
void mpi_startup(int argc, char *argv[]);
/*
 * FUNCTION
 *  Call the MPI startup functions.
 * ARGUMENTS
 *  * argc -- command line argc
 *  * argv -- command line argv
 ******
 */

/****f* mpi_comm/mpi_dom_malloc()
 * NAME
 *  mpi_dom_malloc()
 * USAGE
 */
void mpi_dom_malloc(void);
/*
 * FUNCTION
 *  Allocate node memory.
 ******
 */

/****f* mpi_comm/mpi_dom_init()
 * NAME
 *  mpi_dom_init()
 * USAGE
 */
void mpi_dom_init(void);
/*
 * FUNCTION
 *  Initialize node memory.
 ******
 */

/****f* mpi_comm/mpi_dom_free()
 * NAME
 *  mpi_dom_free()
 * USAGE
 */
void mpi_dom_free(void);
/*
 * FUNCTION
 *  Free node memory.
 ******
 */

/****f* mpi_comm/mpi_cleanup()
 * NAME
 *  mpi_cleanup()
 * USAGE
 */
void mpi_cleanup(void);
/*
 * FUNCTION
 *  Call the MPI finalize function.
 ******
 */

#endif
