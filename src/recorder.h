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

/****h* Bluefish/recorder
 * NAME
 *  recorder
 * FUNCTION
 *  A utility for recording simulation metrics.
 ******
 */

#ifndef _RECORDER_H
#define _RECORDER_H

#include "bluefish.h"
#include <cgnslib.h>

/****f* recorder/cgns_grid()
 * NAME
 *  cgns_grid()
 * TYPE
 */
void cgns_grid(void);
/*
 * FUNCTION
 *  Write the CGNS grid output file.
 ******
 */

/****f* recorder/cgns_flow_field()
 * NAME
 *  cgns_flow_field()
 * TYPE
 */
void cgns_flow_field(real dtout);
/*
 * FUNCTION
 *  Write the CGNS flow_field output file.
 * ARGUMENTS
 *  * dtout -- the output timestep size
 ******
 */

#endif
