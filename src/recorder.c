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

#include "recorder.h"

void cgns_grid(void)
{
  // create the file
  char fname[FILE_NAME_SIZE];
  sprintf(fname, "%s/output/%s", ROOT_DIR, "grid.cgns");
  int fn;
  int bn;
  int zn;
  int gn;
  int cn;
  cg_open(fname, CG_MODE_WRITE, &fn);
  cg_base_write(fn, "Base", 3, 3, &bn);
  cgsize_t size[9];
  size[0] = DOM.xn+1; // cells -> vertices
  size[1] = DOM.yn+1;
  size[2] = DOM.zn+1;
  size[3] = DOM.xn;
  size[4] = DOM.yn;
  size[5] = DOM.zn;
  size[6] = 0;
  size[7] = 0;
  size[8] = 0;
  cg_zone_write(fn, bn, "Zone0", size, Structured, &zn);
  cg_grid_write(fn, bn, zn, "GridCoordinates", &gn);

  real *x = malloc((DOM.xn+1)*(DOM.yn+1)*(DOM.zn+1) * sizeof(real));
  // cpumem = (DOM.xn+1)*(DOM.yn+1)*(DOM.zn+1) * sizeof(real);
  real *y = malloc((DOM.xn+1)*(DOM.yn+1)*(DOM.zn+1) * sizeof(real));
  // cpumem = (DOM.xn+1)*(DOM.yn+1)*(DOM.zn+1) * sizeof(real);
  real *z = malloc((DOM.xn+1)*(DOM.yn+1)*(DOM.zn+1) * sizeof(real));
  // cpumem = (DOM.xn+1)*(DOM.yn+1)*(DOM.zn+1) * sizeof(real);
  for(int k = DOM.Gcc.ks; k < DOM.Gcc.ke+1; k++) {
    for(int j = DOM.Gcc.js; j < DOM.Gcc.je+1; j++) {
      for(int i = DOM.Gcc.is; i < DOM.Gcc.ie+1; i++) {
        int C = (i-1) + (j-1)*(DOM.xn+1) + (k-1)*(DOM.xn+1)*(DOM.yn+1);
        x[C] = DOM.xs + (i-1)*DOM.dx;
        y[C] = DOM.ys + (j-1)*DOM.dy;
        z[C] = DOM.zs + (k-1)*DOM.dz;
      }
    }
  }

  cg_coord_write(fn, bn, zn, RealDouble, "CoordinateX", x, &cn);
  cg_coord_write(fn, bn, zn, RealDouble, "CoordinateY", y, &cn);
  cg_coord_write(fn, bn, zn, RealDouble, "CoordinateZ", z, &cn);

  free(x);
  free(y);
  free(z);

  cg_close(fn);
}

void cgns_flow_field(real dtout)
{
  // create the solution file
  char fname2[FILE_NAME_SIZE];
  char fnameall[FILE_NAME_SIZE];
  char fnameall2[FILE_NAME_SIZE];
  char gname[FILE_NAME_SIZE];
  char gnameall[FILE_NAME_SIZE];
  char format[CHAR_BUF_SIZE];
  char snodename[CHAR_BUF_SIZE];
  char snodenameall[CHAR_BUF_SIZE];
  int sigfigs = ceil(log10(1. / dtout));
  if(sigfigs < 1) sigfigs = 1;
  sprintf(format, "%%.%df", sigfigs);
  sprintf(fname2, "flow-%s.cgns", format);
  sprintf(fnameall2, "%s/output/flow-%s.cgns", ROOT_DIR, format);
  sprintf(snodename, "Solution-");
  sprintf(snodenameall, "/Base/Zone0/Solution-");
  sprintf(snodename, "%s%s", snodename, format);
  sprintf(snodenameall, "%s%s", snodenameall, format);
  sprintf(gname, "grid.cgns");
  sprintf(gnameall, "%s/output/%s", ROOT_DIR, "grid.cgns");
  int fn;
  int bn;
  int zn;
  int sn;
  int fnpress;
  int fnu;
  int fnv;
  int fnw;
  cg_open(fnameall, CG_MODE_WRITE, &fn);
  cg_base_write(fn, "Base", 3, 3, &bn);
  cgsize_t size[9];
  size[0] = DOM.xn+1; // cells -> vertices
  size[1] = DOM.yn+1;
  size[2] = DOM.zn+1;
  size[3] = DOM.xn;
  size[4] = DOM.yn;
  size[5] = DOM.zn;
  size[6] = 0;
  size[7] = 0;
  size[8] = 0;
  cg_zone_write(fn, bn, "Zone0", size, Structured, &zn);
  cg_goto(fn, bn, "Zone_t", zn, "end");
  // check that grid.cgns exists
  /*int fng;
  if(cg_open(gnameall, CG_MODE_READ, &fng) != 0) {
    fprintf(stderr, "CGNS flow field write failure: no grid.cgns\n");
    exit(EXIT_FAILURE);
  } else {
    cg_close(fng);
  }
    cg_close(fng);
*/
  
  cg_link_write("GridCoordinates", gname, "Base/Zone0/GridCoordinates");

  cg_sol_write(fn, bn, zn, "Solution", CellCenter, &sn);
  real *pout = malloc(DOM.Gcc.s3 * sizeof(real));
  // cpumem += DOM.Gcc.s3 * sizeof(real);
  for(int k = DOM.Gcc.ks; k < DOM.Gcc.ke; k++) {
    for(int j = DOM.Gcc.js; j < DOM.Gcc.je; j++) {
      for(int i = DOM.Gcc.is; i < DOM.Gcc.ie; i++) {
        int C = (i-DOM_BUF) + (j-DOM_BUF)*DOM.Gcc.s1 + (k-DOM_BUF)*DOM.Gcc.s2;
        int CC = i + j*DOM.Gcc.s1b + k*DOM.Gcc.s2b;
        pout[C] = p[CC];
      }
    }
  }
  cg_field_write(fn, bn, zn, sn, RealDouble, "Pressure", pout, &fnpress);

  real *uflagout = malloc(DOM.Gfx.s3 * sizeof(int));
  for(int k = DOM.Gfx.ks; k < DOM.Gfx.ke; k++) {
    for(int j = DOM.Gfx.js; j < DOM.Gfx.je; j++) {
      for(int i = DOM.Gfx.is; i < DOM.Gfx.ie-1; i++) {
        int C = (i-DOM_BUF) + (j-DOM_BUF)*DOM.Gfx.s1 + (k-DOM_BUF)*DOM.Gfx.s2;
        int CC0 = i + j*DOM.Gfx.s1b + k*DOM.Gfx.s2b;
        int CC1 = (i+1) + j*DOM.Gfx.s1b + k*DOM.Gfx.s2b;
        uflagout[C] = 0.5 * (flag_u[CC0] + flag_u[CC1]);
      }
    }
  }
  cg_field_write(fn, bn, zn, sn, Integer, "Flag_U", uflagout, &fnu);

  real *vflagout = malloc(DOM.Gfy.s3 * sizeof(int));
  for(int k = DOM.Gfy.ks; k < DOM.Gfy.ke; k++) {
    for(int j = DOM.Gfy.js; j < DOM.Gfy.je-1; j++) {
      for(int i = DOM.Gfy.is; i < DOM.Gfy.ie; i++) {
        int C = (i-DOM_BUF) + (j-DOM_BUF)*DOM.Gfy.s1 + (k-DOM_BUF)*DOM.Gfy.s2;
        int CC0 = i + j*DOM.Gfy.s1b + k*DOM.Gfy.s2b;
        int CC1 = i + (j+1)*DOM.Gfy.s1b + k*DOM.Gfy.s2b;
        vflagout[C] = 0.5 * (flag_v[CC0] + flag_v[CC1]);
      }
    }
  }
  cg_field_write(fn, bn, zn, sn, Integer, "Flag_V", vflagout, &fnv);

  real *wflagout = malloc(DOM.Gfz.s3 * sizeof(int));
  for(int k = DOM.Gfz.ks; k < DOM.Gfz.ke-1; k++) {
    for(int j = DOM.Gfz.js; j < DOM.Gfz.je; j++) {
      for(int i = DOM.Gfz.is; i < DOM.Gfz.ie; i++) {
        int C = (i-DOM_BUF) + (j-DOM_BUF)*DOM.Gfz.s1 + (k-DOM_BUF)*DOM.Gfz.s2;
        int CC0 = i + j*DOM.Gfz.s1b + k*DOM.Gfz.s2b;
        int CC1 = i + j*DOM.Gfx.s1b + (k+1)*DOM.Gfx.s2b;
        wflagout[C] = 0.5 * (flag_w[CC0] + flag_w[CC1]);
      }
    }
  }
  cg_field_write(fn, bn, zn, sn, Integer, "Flag_W", wflagout, &fnw);

  cg_close(fn);
  free(pout);
}
