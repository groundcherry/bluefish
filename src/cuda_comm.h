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

/****h* Bluefish/cuda_comm
 * NAME
 *  cuda_comm
 * FUNCTION
 *  Low-level GPU subdomain control routines.
 ******
 */

#ifndef _CUDA_COMM_H
#define _CUDA_COMM_H

extern "C"
{
#include "bluefish.h"
}

/****f*
 * NAME
 *  touchp
 * TYPE
 */
__global__ void touchp(real *p, dom_struct *dom, int dev);
/*
 * FUNCTION
 *  Provide an test GPU memory manipulation routine. This currently
 *  sets all cells of p equal to dev.
 * ARGUMENTS
 *  * p -- device pressure
 *  * dom -- device domain description
 *  * dev -- either device rank (drank) or node rank (rank)
 ******
 */

#endif
