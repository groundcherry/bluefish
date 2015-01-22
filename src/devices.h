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

/****h* Bluefish/devices
 * NAME
 *  devices
 * FUNCTION
 *  Handle hardware configuration details such as MPI nodes and GPU devices.
 ******
 */

#ifndef _DEVICES_H
#define _DEVICES_H

#include "bluefish.h"

/****d* devices/MAX_DEVS_PER_NODE
 * NAME
 *  MAX_DEVS_PER_NODE
 * TYPE
 */
#define MAX_DEVS_PER_NODE 4
/*
 * PURPOSE
 *  Define the maximum possible number of GPUs per node.
 ******
 */

/****v* devices/nnodes
 * NAME
 *  nnodes
 * TYPE
 */
extern int nnodes;
/*
 * PURPOSE
 *  The number of MPI nodes.
 ******
 */

/****s* devices/node_struct
 * NAME
 *  node_struct
 * TYPE
 */
typedef struct node_struct {
  int ndevs;
  char name[CHAR_BUF_SIZE];
  int devs[MAX_DEVS_PER_NODE];
  int devstart;
} node_struct;
/*
 * PURPOSE
 *  Carry the hardware configuration of a particular node.
 * MEMBERS
 *  * ndevs -- the number of devices
 *  * name -- node name
 *  * devs -- node device access numbers
 *  * devstart -- the global device numbering starting point for this node.
 *    i.e., if this is the third node, and the two prior contained two GPUs
 *    each, devstart for this node is equal to 4.
 ******
 */

/****v* devices/nodes
 * NAME
 *  nodes
 * TYPE
 */
extern node_struct *nodes;
/*
 * PURPOSE
 *  An array containing all nnodes nodes.
 ******
 */

/****f* devices/devs_read_input()
 * NAME
 *  devs_read_input()
 * USAGE
 */
void devs_read_input(void);
/*
 * FUNCTION
 *  Read the hardware input specification file devs.config.
 ******
 */

/****f* devices/devs_show_config()
 * NAME
 *  devs_show_config()
 * USAGE
 */
void devs_show_config(void);
/*
 * FUNCTION
 *  Write the hardware configuration to stdout.
 ******
 */

/****f* devices/devs_clean()
 * NAME
 *  devs_clean()
 * USAGE
 */
void devs_clean(void);
/*
 * FUNCTION
 *  Free any allocated variables associated with the device configuration
 *  specification.
 ******
 */

#endif
