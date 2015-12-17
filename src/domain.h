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

/****h* Bluefish/domain
 * NAME
 *  domain
 * FUNCTION
 *  Low-level domain functions.
 ******
 */

#ifndef _DOMAIN_H
#define _DOMAIN_H

/****s* domain/grid_info
 * NAME
 *  grid_info
 * TYPE
 */
typedef struct grid_info {
  int is;
  int ie;
  int in;
  int isb;
  int ieb;
  int inb;
  int js;
  int je;
  int jn;
  int jsb;
  int jeb;
  int jnb;
  int ks;
  int ke;
  int kn;
  int ksb;
  int keb;
  int knb;
  int _is;
  int _ie;
  int _in;
  int _isb;
  int _ieb;
  int _inb;
  int _js;
  int _je;
  int _jn;
  int _jsb;
  int _jeb;
  int _jnb;
  int _ks;
  int _ke;
  int _kn;
  int _ksb;
  int _keb;
  int _knb;
  int s1;
  int s1b;
  int s2;
  int s2b;
  int s3;
  int s3b;
} grid_info;
/*
 * PURPOSE
 *  Carry information related to the different discretization grids.
 * MEMBERS
 *  * is -- the domain start index in the x-direction (global indexing)
 *  * ie -- the domain end index in the x-direction (global indexing)
 *  * in -- the number of elements in the domain in the x-direction
 *  * isb -- the domain start index in the x-direction plus boundary ghost
 *    elements (global indexing)
 *  * ieb -- the domain end index in the x-direction plus boundary ghost
 *    elements (global indexing)
 *  * inb -- the number of elements in the domain in the x-direction plus
 *    the boundary ghost elements
 *  * js -- the domain start index in the y-direction (global indexing)
 *  * je -- the domain end index in the y-direction (global indexing)
 *  * jn -- the number of elements in the domain in the y-direction
 *  * jsb -- the domain start index in the y-direction plus boundary ghost
 *    elements (global indexing)
 *  * jeb -- the domain end index in the y-direction plus boundary ghost
 *    elements (global indexing)
 *  * jnb -- the number of elements in the domain in the y-direction plus
 *    the boundary ghost elements
 *  * ks -- the domain start index in the z-direction (global indexing)
 *  * ke -- the domain end index in the z-direction (global indexing)
 *  * kn -- the number of elements in the domain in the z-direction
 *  * ksb -- the domain start index in the z-direction plus boundary ghost
 *    elements (global indexing)
 *  * keb -- the domain end index in the z-direction plus boundary ghost
 *    elements (global indexing)
 *  * knb -- the number of elements in the domain in the z-direction plus
 *    the boundary ghost elements
 *  * _is -- the domain start index in the x-direction (local indexing)
 *  * _ie -- the domain end index in the x-direction (local indexing)
 *  * _in -- the number of elements in the x-direction (local indexing)
 *  * _isb -- the domain start index in the x-direction plus boundary ghost
 *    elements (local indexing)
 *  * _ieb -- the domain end index in the x-direction plus boundary ghost
 *    elements (local indexing)
 *  * _inb -- the number of elements in the x-direction (local indexing) plus
 *    boundary ghost elements
 *  * _js -- the domain start index in the y-direction (local indexing)
 *  * _je -- the domain end index in the y-direction (local indexing)
 *  * _jn -- the number of elements in the y-direction (local indexing)
 *  * _jsb -- the domain start index in the y-direction plus boundary ghost
 *    elements (local indexing)
 *  * _jeb -- the domain end index in the y-direction plus boundary ghost
 *    elements (local indexing)
 *  * _jnb -- the number of elements in the y-direction (local indexing) plus
 *    boundary ghost elements
 *  * _ks -- the domain start index in the z-direction (local indexing)
 *  * _ke -- the domain end index in the z-direction (local indexing)
 *  * _kn -- the number of elements in the z-direction (local indexing)
 *  * _ksb -- the domain start index in the z-direction plus boundary ghost
 *    elements (local indexing)
 *  * _keb -- the domain end index in the z-direction plus boundary ghost
 *  * _knb -- the number of elements in the z-direction (local indexing) plus
 *    boundary ghost elements
 *    elements (local indexing)
 *  * s1 -- the looping stride length for the fastest-changing variable (x)
 *  * s1b -- the looping stride length for the fastest-changing variable (x)
 *    plus the boundary ghost elements
 *  * s2 -- the looping stride length for the second-changing variable (y)
 *  * s2b -- the looping stride length for the second-changing variable (y)
 *    plus the boundary ghost elements
 *  * s3 -- the looping stride length for the slowest-changing variable (z)
 *  * s3b -- the looping stride length for the slowest-changing variable (z)
 *    plus the boundary ghost elements
 ******
 */

/****v* domain/nMPIdom
 * NAME
 *  nMPIdom
 * TYPE
 */
extern int nMPIdom;
/*
 * PURPOSE
 *  Number of subdomains into which the domain should be decomposed for MPI
 *  subdivision of the domain.
 ******
 */

/****v* domain/nGPUdom
 * NAME
 *  nGPUdom
 * TYPE
 */
extern int nGPUdom;
/*
 * PURPOSE
 *  Number of subdomains into which the domain should be decomposed for GPU
 *  subdivision of the domain.
 ******
 */

/****s* domain/dom_struct
 * NAME
 *  dom_struct
 * TYPE
 */
typedef struct dom_struct {
  grid_info Gcc;
  grid_info Gfx;
  grid_info Gfy;
  grid_info Gfz;
  real xs;
  real xe;
  real xl;
  int xn;
  real dx;
  real ys;
  real ye;
  real yl;
  int yn;
  real dy;
  real zs;
  real ze;
  real zl;
  int zn;
  real dz;
  int E;
  int W;
  int N;
  int S;
  int T;
  int B;
  int e;
  int w;
  int n;
  int s;
  int t;
  int b;
} dom_struct;
/*
 * PURPOSE
 *  Carry information related to a subdomain.
 * MEMBERS
 *  * Gcc -- cell-centered grid information
 *  * xs -- physical start position in the x-direction
 *  * xe -- physical end position in the x-direction
 *  * xl -- physical length of the subdomain in the x-direction
 *  * xn -- number of discrete cells in the x-direction
 *  * dx -- cell size in the x-direction
 *  * ys -- physical start position in the y-direction
 *  * ye -- physical end position in the y-direction
 *  * yl -- physical length of the subdomain in the y-direction
 *  * yn -- number of discrete cells in the y-direction
 *  * dy -- cell size in the y-direction
 *  * zs -- physical start position in the z-direction
 *  * ze -- physical end position in the z-direction
 *  * zl -- physical length of the subdomain in the z-direction
 *  * zn -- number of discrete cells in the z-direction
 *  * dz -- cell size in the z-direction
 *  * E -- the subdomain adjacent to the east face of the cell
 *    (E = -1 if the face is a domain boundary) (global indexing)
 *  * W -- the subdomain adjacent to the west face of the cell
 *    (W = -1 if the face is a domain boundary) (global indexing)
 *  * N -- the subdomain adjacent to the north face of the cell
 *    (N = -1 if the face is a domain boundary) (global indexing)
 *  * S -- the subdomain adjacent to the south face of the cell
 *    (S = -1 if the face is a domain boundary) (global indexing)
 *  * T -- the subdomain adjacent to the top face of the cell
 *    (T = -1 if the face is a domain boundary) (global indexing)
 *  * B -- the subdomain adjacent to the bottom face of the cell
 *    (B = -1 if the face is a domain boundary) (global indexing)
 *  * e -- the subdomain adjacent to the east face of the cell
 *    (e = -1 if the face is a domain boundary) (local indexing)
 *  * w -- the subdomain adjacent to the west face of the cell
 *    (w = -1 if the face is a domain boundary) (local indexing)
 *  * n -- the subdomain adjacent to the north face of the cell
 *    (n = -1 if the face is a domain boundary) (local indexing)
 *  * s -- the subdomain adjacent to the south face of the cell
 *    (s = -1 if the face is a domain boundary) (local indexing)
 *  * t -- the subdomain adjacent to the top face of the cell
 *    (t = -1 if the face is a domain boundary) (local indexing)
 *  * b -- the subdomain adjacent to the bottom face of the cell
 *    (b = -1 if the face is a domain boundary) (local indexing)
 ******
 */

/****f* domain/domain_read_input()
 * NAME
 *  domain_read_input()
 * USAGE
 */
void domain_read_input(void);
/*
 * FUNCTION
 *  Read domain specifications and simulation parameters from flow.config.
 ******
 */

/****f* domain/domain_write_config()
 * NAME
 *  domain_write_config()
 * USAGE
 */
void domain_map_write_config(void);
/*
 * FUNCTION
 *  Write domain specifications and simulation parameters to file.
 ******
 */

/****f* domain/domain_map()
 * NAME
 *  domain_map()
 * USAGE
 */
void domain_map(void);
/*
 * FUNCTION
 *  Initialize the domain on the host.
 ******
 */

/****f* domain/domain_clean()
 * NAME
 *  domain_clean()
 * USAGE
 */
void domain_map_clean(void);
/*
 * FUNCTION
 *   Clean up. Free any allocated host memory.
 ******
 */

#endif
