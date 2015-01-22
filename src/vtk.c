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

#include "bluefish.h"

void init_VTK(void)
{
  if(rank == 0) {
    char fname[FILE_NAME_SIZE];

    // open PVD file for writing
    sprintf(fname, "%sout.pvd", OUTPUT_DIR);
    FILE *outfile = fopen(fname, "w");
    if(outfile == NULL) {
      fprintf(stderr, "Could not open file %s\n", fname);
      exit(EXIT_FAILURE);
    }

    // write PVD file header and footer
    fprintf(outfile, "<VTKFile type=\"Collection\">\n");
    fprintf(outfile, "<Collection>\n");
    fprintf(outfile, "</Collection>\n");
    fprintf(outfile, "</VTKFile>");

    // close the file
    fclose(outfile);
  }
}

void out_VTK(void)
{
  if(rank == 0) {
    char fname_pvd[FILE_NAME_SIZE]; // pvd filename
    char fname_pvtr[FILE_NAME_SIZE]; // pvtr filename

    sprintf(fname_pvd, "%sout.pvd", OUTPUT_DIR);
    sprintf(fname_pvtr, "out_%d.pvtr", 0);

    FILE *pvdfile= fopen(fname_pvd, "r+");
    // moves back 2 lines from the end of the file (above the footer)
    fseek(pvdfile, -24, SEEK_END);

    fprintf(pvdfile, "<DataSet timestep=\"%e\" part=\"0\" file=\"%s\"/>\n",
      0., fname_pvtr);
    fprintf(pvdfile, "</Collection>\n");
    fprintf(pvdfile, "</VTKFile>");
    fclose(pvdfile);
  }

  dom_out_VTK();
}

void dom_out_VTK(void)
{
  int i, j, k, l; // iterators
  char fname[FILE_NAME_SIZE]; // output filename
  char fname_dom[FILE_NAME_SIZE]; // subdomain filename
  int C;  // cell center index
  int Cx;  // cell center index for interpolation
  int Cy;  // cell center index for interpolation
  int Cz;  // cell center index for interpolation

  // only work on pvtr file once
  if(rank == 0) {
    sprintf(fname, "%sout_%d.pvtr", OUTPUT_DIR, 0);
    FILE *outfile = fopen(fname, "w");
    if(outfile == NULL) {
      fprintf(stderr, "Could not open file %s\n", fname);
      exit(EXIT_FAILURE);
    }

    // write Paraview pvtr file
    fprintf(outfile, "<VTKFile type=\"PRectilinearGrid\">\n");
    fprintf(outfile, "<PRectilinearGrid WholeExtent=");
    fprintf(outfile, "\"0 %d 0 %d 0 %d\" GhostLevel=\"0\">\n",
      DOM.xn, DOM.yn, DOM.zn);
    fprintf(outfile, "<PCellData Scalars=\"p p0\">\n");
    fprintf(outfile, "<PDataArray type=\"Float32\" Name=\"p\"/>\n");
    fprintf(outfile, "<PDataArray type=\"Float32\" Name=\"p0\"/>\n");
    fprintf(outfile, "</PCellData>\n");
    fprintf(outfile, "<PCoordinates>\n");
    fprintf(outfile, "<PDataArray type=\"Float32\" Name=\"x\"/>\n");
    fprintf(outfile, "<PDataArray type=\"Float32\" Name=\"y\"/>\n");
    fprintf(outfile, "<PDataArray type=\"Float32\" Name=\"z\"/>\n");
    fprintf(outfile, "</PCoordinates>\n");
    for(l = 0; l < nnodes; l++) {
      fprintf(outfile, "<Piece Extent=\"");
      fprintf(outfile, "%d %d ", Dom[l].Gcc.is-DOM_BUF, Dom[l].Gcc.ie);
      fprintf(outfile, "%d %d ", Dom[l].Gcc.js-DOM_BUF, Dom[l].Gcc.je);
      fprintf(outfile, "%d %d\" ", Dom[l].Gcc.ks-DOM_BUF, Dom[l].Gcc.ke);
      sprintf(fname_dom, "out_%d_%d.vtr", 0, l);
      fprintf(outfile, "Source=\"%s\"/>\n", fname_dom);
    }
    fprintf(outfile, "</PRectilinearGrid>\n");
    fprintf(outfile, "</VTKFile>\n");
    fclose(outfile);
  }

  // interpolate velocities to cell centers
  // cell-center working arrays
  real *flag_uu = (real*) malloc(Dom[rank].Gcc.s3b * sizeof(real));
  real *flag_vv = (real*) malloc(Dom[rank].Gcc.s3b * sizeof(real));
  real *flag_ww = (real*) malloc(Dom[rank].Gcc.s3b * sizeof(real));
  for(k = Dom[rank].Gcc.ks; k < Dom[rank].Gcc.ke; k++) {
    for(j = Dom[rank].Gcc.js; j < Dom[rank].Gcc.je; j++) {
      for(i = Dom[rank].Gcc.is; i < Dom[rank].Gcc.ie; i++) {
        // interpolate velocity
        C = i + j*Dom[rank].Gcc.s1b + k*Dom[rank].Gcc.s2b;
        Cx = i + j*Dom[rank].Gfx.s1b + k*Dom[rank].Gfx.s2b;
        Cy = i + j*Dom[rank].Gfy.s1b + k*Dom[rank].Gfy.s2b;
        Cz = i + j*Dom[rank].Gfz.s1b + k*Dom[rank].Gfz.s2b;
        // interpolate flags
        flag_uu[C] = 0.5*(flag_u[Cx] + flag_u[Cx+1]);
        flag_vv[C] = 0.5*(flag_v[Cy] + flag_v[Cy+Dom[rank].Gfy.s1b]);
        flag_ww[C] = 0.5*(flag_w[Cz] + flag_w[Cz+Dom[rank].Gfz.s2b]);
      }
    }
  }

  // write subdomain file
  // number of cells in the subdomain

  // open file for writing
  sprintf(fname, "%s/out_%d_%d.vtr", OUTPUT_DIR, 0, rank);
  FILE *outfile = fopen(fname, "w");
  if(outfile == NULL) {
    fprintf(stderr, "Could not open file %s\n", fname);
    exit(EXIT_FAILURE);
  }

  fprintf(outfile, "<VTKFile type=\"RectilinearGrid\">\n");
  fprintf(outfile, "<RectilinearGrid WholeExtent=\"");
  fprintf(outfile, "%d %d ", Dom[rank].Gcc.is-DOM_BUF,
    Dom[rank].Gcc.ie);
  fprintf(outfile, "%d %d ", Dom[rank].Gcc.js-DOM_BUF,
    Dom[rank].Gcc.je);
  fprintf(outfile, "%d %d\" GhostLevel=\"0\">\n", Dom[rank].Gcc.ks-DOM_BUF,
    Dom[rank].Gcc.ke);

  fprintf(outfile, "<Piece Extent=\"");
  fprintf(outfile, "%d %d ", Dom[rank].Gcc.is-DOM_BUF,
    Dom[rank].Gcc.ie);
  fprintf(outfile, "%d %d ", Dom[rank].Gcc.js-DOM_BUF,
    Dom[rank].Gcc.je);
  fprintf(outfile, "%d %d\">\n", Dom[rank].Gcc.ks-DOM_BUF,
    Dom[rank].Gcc.ke);
  fprintf(outfile, "<CellData Scalars=\"p p0\">\n");

  // write pressure
  fprintf(outfile, "<DataArray type=\"Float32\" Name=\"p\">\n");
  for(k = Dom[rank].Gcc._ks; k <= Dom[rank].Gcc._ke; k++) {
    for(j = Dom[rank].Gcc._js; j <= Dom[rank].Gcc._je; j++) {
      for(i = Dom[rank].Gcc._is; i <= Dom[rank].Gcc._ie; i++) {
        C = i + j*Dom[rank].Gcc.s1b + k*Dom[rank].Gcc.s2b;
        fprintf(outfile, "%lf ", p[C]);
      }
    }
  }
  fprintf(outfile, "\n");
  fprintf(outfile, "</DataArray>\n");

  // write pressure
  fprintf(outfile, "<DataArray type=\"Float32\" Name=\"p0\">\n");
  for(k = Dom[rank].Gcc._ks; k <= Dom[rank].Gcc._ke; k++) {
    for(j = Dom[rank].Gcc._js; j <= Dom[rank].Gcc._je; j++) {
      for(i = Dom[rank].Gcc._is; i <= Dom[rank].Gcc._ie; i++) {
        C = i + j*Dom[rank].Gcc.s1b + k*Dom[rank].Gcc.s2b;
        fprintf(outfile, "%lf ", p0[C]);
      }
    }
  }
  fprintf(outfile, "\n");
  fprintf(outfile, "</DataArray>\n");

  fprintf(outfile, "</CellData>\n");

  fprintf(outfile, "<Coordinates>\n");
  fprintf(outfile, "<DataArray type=\"Float32\" Name=\"x\">\n");
  for(i = Dom[rank].Gcc._is-DOM_BUF; i <= Dom[rank].Gcc._ie; i++) {
    fprintf(outfile, "%lf ", i * Dom[rank].dx + Dom[rank].xs);
  }
  fprintf(outfile, "\n");
  fprintf(outfile, "</DataArray>\n");
  fprintf(outfile, "<DataArray type=\"Float32\" Name=\"y\">\n");
  for(j = Dom[rank].Gcc._js-DOM_BUF; j <= Dom[rank].Gcc._je; j++) {
    fprintf(outfile, "%lf ", j * Dom[rank].dy + Dom[rank].ys);
  }
  fprintf(outfile, "\n");
  fprintf(outfile, "</DataArray>\n");
  fprintf(outfile, "<DataArray type=\"Float32\" Name=\"z\">\n");
  for(k = Dom[rank].Gcc._ks-DOM_BUF; k <= Dom[rank].Gcc._ke; k++) {
    fprintf(outfile, "%lf ", k * Dom[rank].dz + Dom[rank].zs);
  }
  fprintf(outfile, "\n");
  fprintf(outfile, "</DataArray>\n");
  fprintf(outfile, "</Coordinates>\n");
  fprintf(outfile, "</Piece>\n");
  fprintf(outfile, "</RectilinearGrid>\n");
  fprintf(outfile, "</VTKFile>\n");
  fclose(outfile);

  // clean up interpolated fields
  free(flag_uu);
  free(flag_vv);
  free(flag_ww);
}
