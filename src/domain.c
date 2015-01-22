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

#include <time.h>

#include "bluefish.h"
#include "domain.h"

int nGPUdom;
int nMPIdom;

void domain_read_input(void)
{
  int i;  // iterator

  int fret = 0;
  fret = fret; // prevent compiler warning

  // open configuration file for reading
  char fname[FILE_NAME_SIZE];
  sprintf(fname, "%s/input/flow.config", ROOT_DIR);
  FILE *infile = fopen(fname, "r");
  if(infile == NULL) {
    fprintf(stderr, "Could not open file %s\n", fname);
    exit(EXIT_FAILURE);
  }

  char buf[CHAR_BUF_SIZE];  // character read buffer

  // read domain
  fret = fscanf(infile, "DOMAIN\n");
#ifdef DOUBLE
  fret = fscanf(infile, "(Xs, Xe, Xn) %lf %lf %d\n", &DOM.xs, &DOM.xe, &DOM.xn);
  fret = fscanf(infile, "(Ys, Ye, Yn) %lf %lf %d\n", &DOM.ys, &DOM.ye, &DOM.yn);
  fret = fscanf(infile, "(Zs, Ze, Zn) %lf %lf %d\n", &DOM.zs, &DOM.ze, &DOM.zn);
  fret = fscanf(infile, "\n"); // \n
#else // single precision
  fret = fscanf(infile, "(Xs, Xe, Xn) %f %f %d\n", &DOM.xs, &DOM.xe, &DOM.xn);
  fret = fscanf(infile, "(Ys, Ye, Yn) %f %f %d\n", &DOM.ys, &DOM.ye, &DOM.yn);
  fret = fscanf(infile, "(Zs, Ze, Zn) %f %f %d\n", &DOM.zs, &DOM.ze, &DOM.zn);
  fret = fscanf(infile, "\n"); // \n
#endif
  /**** READ MPI DECOMPOSITION ****/
  fret = fscanf(infile, "MPI DOMAIN DECOMPOSITION\n");
  fret = fscanf(infile, "n %d\n", &nMPIdom);
  // allocate subdomain data structure
  Dom = (dom_struct*) malloc(nMPIdom * sizeof(dom_struct));
  for(i = 0; i < nMPIdom; i++) {  // read subDomains
    int tmp;
    fret = fscanf(infile, "%d\n", &tmp);
#ifdef DOUBLE
    fret = fscanf(infile, "(Xs, Xe, Xn) %lf %lf %d\n", &Dom[i].xs, &Dom[i].xe,
      &Dom[i].xn);
    fret = fscanf(infile, "(Ys, Ye, Yn) %lf %lf %d\n", &Dom[i].ys, &Dom[i].ye,
      &Dom[i].yn);
    fret = fscanf(infile, "(Zs, Ze, Zn) %lf %lf %d\n", &Dom[i].zs, &Dom[i].ze,
      &Dom[i].zn);
#else // single
    fret = fscanf(infile, "(Xs, Xe, Xn) %f %f %d\n", &Dom[i].xs, &Dom[i].xe,
      &Dom[i].xn);
    fret = fscanf(infile, "(Ys, Ye, Yn) %f %f %d\n", &Dom[i].ys, &Dom[i].ye,
      &Dom[i].yn);
    fret = fscanf(infile, "(Zs, Ze, Zn) %f %f %d\n", &Dom[i].zs, &Dom[i].ze,
      &Dom[i].zn);
#endif
    fret = fscanf(infile, "W %d E %d S %d N %d B %d T %d\n", &Dom[i].W, &Dom[i].E,
      &Dom[i].S, &Dom[i].N, &Dom[i].B, &Dom[i].T);
    Dom[i].w = Dom[i].W;
    Dom[i].e = Dom[i].E;
    Dom[i].s = Dom[i].S;
    Dom[i].n = Dom[i].N;
    Dom[i].b = Dom[i].B;
    Dom[i].t = Dom[i].T;
    fret = fscanf(infile, "\n");
  }
  fret = fscanf(infile, "\n");

  /**** READ GPU DECOMPOSITION ****/
  fret = fscanf(infile, "GPU DOMAIN DECOMPOSITION\n");
  fret = fscanf(infile, "n %d\n", &nGPUdom);
  // allocate subdomain data structure
  dom = (dom_struct*) malloc(nGPUdom * sizeof(dom_struct));
  for(i = 0; i < nGPUdom; i++) {  // read subdomains
    int tmp;
    fret = fscanf(infile, "%d\n", &tmp);
#ifdef DOUBLE
    fret = fscanf(infile, "(Xs, Xe, Xn) %lf %lf %d\n", &dom[i].xs, &dom[i].xe,
      &dom[i].xn);
    fret = fscanf(infile, "(Ys, Ye, Yn) %lf %lf %d\n", &dom[i].ys, &dom[i].ye,
      &dom[i].yn);
    fret = fscanf(infile, "(Zs, Ze, Zn) %lf %lf %d\n", &dom[i].zs, &dom[i].ze,
      &dom[i].zn);
#else // single
    fret = fscanf(infile, "(Xs, Xe, Xn) %f %f %d\n", &dom[i].xs, &dom[i].xe,
      &dom[i].xn);
    fret = fscanf(infile, "(Ys, Ye, Yn) %f %f %d\n", &dom[i].ys, &dom[i].ye,
      &dom[i].yn);
    fret = fscanf(infile, "(Zs, Ze, Zn) %f %f %d\n", &dom[i].zs, &dom[i].ze,
      &dom[i].zn);
#endif
    fret = fscanf(infile, "W %d E %d S %d N %d B %d T %d\n", &dom[i].W, &dom[i].E,
      &dom[i].S, &dom[i].N, &dom[i].B, &dom[i].T);
    fret = fscanf(infile, "w %d e %d s %d n %d b %d t %d\n", &dom[i].w, &dom[i].e,
      &dom[i].s, &dom[i].n, &dom[i].b, &dom[i].t);
    fret = fscanf(infile, "\n");
  }
  fret = fscanf(infile, "\n");

  fret = fscanf(infile, "SIMULATION PARAMETERS\n");
#ifdef DOUBLE
  fret = fscanf(infile, "pp_max_iter %d\n", &pp_max_iter);
  fret = fscanf(infile, "pp_residual %lf\n", &pp_residual);
#else
  fret = fscanf(infile, "pp_max_iter %d\n", &pp_max_iter);
  fret = fscanf(infile, "pp_residual %f\n", &pp_residual);
#endif
  fret = fscanf(infile, "\n");

  fret = fscanf(infile, "BOUNDARY CONDITIONS\n");
  fret = fscanf(infile, "PRESSURE\n");
  fret = fscanf(infile, "bc.pW %s", buf);
  if(strcmp(buf, "PERIODIC") == 0)
    bc.pW = PERIODIC;
  else if(strcmp(buf, "NEUMANN") == 0)
    bc.pW = NEUMANN;
  else {
    fprintf(stderr, "flow.config read error.\n");
    exit(EXIT_FAILURE);
  }
  fret = fscanf(infile, "\n");
  fret = fscanf(infile, "bc.pE %s", buf);
  if(strcmp(buf, "PERIODIC") == 0)
    bc.pE = PERIODIC;
  else if(strcmp(buf, "NEUMANN") == 0)
    bc.pE = NEUMANN;
  else {
    fprintf(stderr, "flow.config read error.\n");
    exit(EXIT_FAILURE);
  }
  fret = fscanf(infile, "\n");
  fret = fscanf(infile, "bc.pS %s", buf);
  if(strcmp(buf, "PERIODIC") == 0)
    bc.pS = PERIODIC;
  else if(strcmp(buf, "NEUMANN") == 0)
    bc.pS = NEUMANN;
  else {
    fprintf(stderr, "flow.config read error.\n");
    exit(EXIT_FAILURE);
  }
  fret = fscanf(infile, "\n");
  fret = fscanf(infile, "bc.pN %s", buf);
  if(strcmp(buf, "PERIODIC") == 0)
    bc.pN = PERIODIC;
  else if(strcmp(buf, "NEUMANN") == 0)
    bc.pN = NEUMANN;
  else {
    fprintf(stderr, "flow.config read error.\n");
    exit(EXIT_FAILURE);
  }
  fret = fscanf(infile, "\n");
  fret = fscanf(infile, "bc.pB %s", buf);
  if(strcmp(buf, "PERIODIC") == 0)
    bc.pB = PERIODIC;
  else if(strcmp(buf, "NEUMANN") == 0)
    bc.pB = NEUMANN;
  else {
    fprintf(stderr, "flow.config read error.\n");
    exit(EXIT_FAILURE);
  }
  fret = fscanf(infile, "\n");
  fret = fscanf(infile, "bc.pT %s", buf);
  if(strcmp(buf, "PERIODIC") == 0)
    bc.pT = PERIODIC;
  else if(strcmp(buf, "NEUMANN") == 0)
    bc.pT = NEUMANN;
  else {
    fprintf(stderr, "flow.config read error.\n");
    exit(EXIT_FAILURE);
  }
  fret = fscanf(infile, "\n");

  fclose(infile);
}

void domain_map_write_config(void)
{
  int i, j;   // iterator

  // write a file for each node
  char fname[FILE_NAME_SIZE];
  sprintf(fname, "node-%d-map.debug", rank);
  FILE *outfile = fopen(fname, "w");

  fprintf(outfile, "Domain:\n");
  fprintf(outfile, "  X: (%f, %f), dX = %f\n", DOM.xs, DOM.xe, DOM.dx);
  fprintf(outfile, "  Y: (%f, %f), dY = %f\n", DOM.ys, DOM.ye, DOM.dy);
  fprintf(outfile, "  Z: (%f, %f), dZ = %f\n", DOM.zs, DOM.ze, DOM.dz);
  fprintf(outfile, "  Xn = %d, Yn = %d, Zn = %d\n", DOM.xn, DOM.yn, DOM.zn);
  fprintf(outfile, "Domain Grids:\n");
  fprintf(outfile, "  DOM.Gcc:\n");
  fprintf(outfile, "    is = %d, ie = %d, in = %d\n", DOM.Gcc.is, DOM.Gcc.ie, DOM.Gcc.in);
  fprintf(outfile, "    isb = %d, ieb = %d, inb = %d\n", DOM.Gcc.isb, DOM.Gcc.ieb,
    DOM.Gcc.inb);
  fprintf(outfile, "    js = %d, je = %d, jn = %d\n", DOM.Gcc.js, DOM.Gcc.je, DOM.Gcc.jn);
  fprintf(outfile, "    jsb = %d, jeb = %d, jnb = %d\n", DOM.Gcc.jsb, DOM.Gcc.jeb,
    DOM.Gcc.jnb);
  fprintf(outfile, "    ks = %d, ke = %d, kn = %d\n", DOM.Gcc.ks, DOM.Gcc.ke, DOM.Gcc.kn);
  fprintf(outfile, "    ksb = %d, keb = %d, knb = %d\n", DOM.Gcc.ksb, DOM.Gcc.keb,
    DOM.Gcc.knb);
  fprintf(outfile, "    _is = %d, _ie = %d, in = %d\n", DOM.Gcc._is, DOM.Gcc._ie,
    DOM.Gcc.in);
  fprintf(outfile, "    _isb = %d, _ieb = %d, inb = %d\n", DOM.Gcc._isb, DOM.Gcc._ieb,
    DOM.Gcc.inb);
  fprintf(outfile, "    _js = %d, _je = %d, jn = %d\n", DOM.Gcc._js, DOM.Gcc._je,
    DOM.Gcc.jn);
  fprintf(outfile, "    _jsb = %d, _jeb = %d, jnb = %d\n", DOM.Gcc._jsb, DOM.Gcc._jeb,
    DOM.Gcc.jnb);
  fprintf(outfile, "    _ks = %d, _ke = %d, kn = %d\n", DOM.Gcc._ks, DOM.Gcc._ke,
    DOM.Gcc.kn);
  fprintf(outfile, "    _ksb = %d, _keb = %d, knb = %d\n", DOM.Gcc._ksb, DOM.Gcc._keb,
    DOM.Gcc.knb);
  fprintf(outfile, "    s1 = %d, s2 = %d, s3 = %d\n", DOM.Gcc.s1, DOM.Gcc.s2,
    DOM.Gcc.s3);
  fprintf(outfile, "    s1b = %d, s2b = %d, s3b = %d\n", DOM.Gcc.s1b, DOM.Gcc.s2b,
    DOM.Gcc.s3b);
  fprintf(outfile, "  DOM.Gfx:\n");
  fprintf(outfile, "    is = %d, ie = %d, in = %d\n", DOM.Gfx.is, DOM.Gfx.ie, DOM.Gfx.in);
  fprintf(outfile, "    isb = %d, ieb = %d, inb = %d\n", DOM.Gfx.isb, DOM.Gfx.ieb,
    DOM.Gfx.inb);
  fprintf(outfile, "    js = %d, je = %d, jn = %d\n", DOM.Gfx.js, DOM.Gfx.je, DOM.Gfx.jn);
  fprintf(outfile, "    jsb = %d, jeb = %d, jnb = %d\n", DOM.Gfx.jsb, DOM.Gfx.jeb,
    DOM.Gfx.jnb);
  fprintf(outfile, "    ks = %d, ke = %d, kn = %d\n", DOM.Gfx.ks, DOM.Gfx.ke, DOM.Gfx.kn);
  fprintf(outfile, "    ksb = %d, keb = %d, knb = %d\n", DOM.Gfx.ksb, DOM.Gfx.keb,
    DOM.Gfx.knb);
  fprintf(outfile, "    _is = %d, _ie = %d, in = %d\n", DOM.Gfx._is, DOM.Gfx._ie,
    DOM.Gfx.in);
  fprintf(outfile, "    _isb = %d, _ieb = %d, inb = %d\n", DOM.Gfx._isb, DOM.Gfx._ieb,
    DOM.Gfx.inb);
  fprintf(outfile, "    _js = %d, _je = %d, jn = %d\n", DOM.Gfx._js, DOM.Gfx._je,
    DOM.Gfx.jn);
  fprintf(outfile, "    _jsb = %d, _jeb = %d, jnb = %d\n", DOM.Gfx._jsb, DOM.Gfx._jeb,
    DOM.Gfx.jnb);
  fprintf(outfile, "    _ks = %d, _ke = %d, kn = %d\n", DOM.Gfx._ks, DOM.Gfx._ke,
    DOM.Gfx.kn);
  fprintf(outfile, "    _ksb = %d, _keb = %d, knb = %d\n", DOM.Gfx._ksb, DOM.Gfx._keb,
    DOM.Gfx.knb);
  fprintf(outfile, "    s1 = %d, s2 = %d, s3 = %d\n", DOM.Gfx.s1, DOM.Gfx.s2,
    DOM.Gfx.s3);
  fprintf(outfile, "    s1b = %d, s2b = %d, s3b = %d\n", DOM.Gfx.s1b, DOM.Gfx.s2b,
    DOM.Gfx.s3b);
  fprintf(outfile, "  DOM.Gfy:\n");
  fprintf(outfile, "    is = %d, ie = %d, in = %d\n", DOM.Gfy.is, DOM.Gfy.ie, DOM.Gfy.in);
  fprintf(outfile, "    isb = %d, ieb = %d, inb = %d\n", DOM.Gfy.isb, DOM.Gfy.ieb,
    DOM.Gfy.inb);
  fprintf(outfile, "    js = %d, je = %d, jn = %d\n", DOM.Gfy.js, DOM.Gfy.je, DOM.Gfy.jn);
  fprintf(outfile, "    jsb = %d, jeb = %d, jnb = %d\n", DOM.Gfy.jsb, DOM.Gfy.jeb,
    DOM.Gfy.jnb);
  fprintf(outfile, "    ks = %d, ke = %d, kn = %d\n", DOM.Gfy.ks, DOM.Gfy.ke, DOM.Gfy.kn);
  fprintf(outfile, "    ksb = %d, keb = %d, knb = %d\n", DOM.Gfy.ksb, DOM.Gfy.keb,
    DOM.Gfy.knb);
  fprintf(outfile, "    _is = %d, _ie = %d, in = %d\n", DOM.Gfy._is, DOM.Gfy._ie,
    DOM.Gfy.in);
  fprintf(outfile, "    _isb = %d, _ieb = %d, inb = %d\n", DOM.Gfy._isb, DOM.Gfy._ieb,
    DOM.Gfy.inb);
  fprintf(outfile, "    _js = %d, _je = %d, jn = %d\n", DOM.Gfy._js, DOM.Gfy._je,
    DOM.Gfy.jn);
  fprintf(outfile, "    _jsb = %d, _jeb = %d, jnb = %d\n", DOM.Gfy._jsb, DOM.Gfy._jeb,
    DOM.Gfy.jnb);
  fprintf(outfile, "    _ks = %d, _ke = %d, kn = %d\n", DOM.Gfy._ks, DOM.Gfy._ke,
    DOM.Gfy.kn);
  fprintf(outfile, "    _ksb = %d, _keb = %d, knb = %d\n", DOM.Gfy._ksb, DOM.Gfy._keb,
    DOM.Gfy.knb);
  fprintf(outfile, "    s1 = %d, s2 = %d, s3 = %d\n", DOM.Gfy.s1, DOM.Gfy.s2,
    DOM.Gfy.s3);
  fprintf(outfile, "    s1b = %d, s2b = %d, s3b = %d\n", DOM.Gfy.s1b, DOM.Gfy.s2b,
    DOM.Gfy.s3b);
  fprintf(outfile, "  DOM.Gfz:\n");
  fprintf(outfile, "    is = %d, ie = %d, in = %d\n", DOM.Gfz.is, DOM.Gfz.ie, DOM.Gfz.in);
  fprintf(outfile, "    isb = %d, ieb = %d, inb = %d\n", DOM.Gfz.isb, DOM.Gfz.ieb,
    DOM.Gfz.inb);
  fprintf(outfile, "    js = %d, je = %d, jn = %d\n", DOM.Gfz.js, DOM.Gfz.je, DOM.Gfz.jn);
  fprintf(outfile, "    jsb = %d, jeb = %d, jnb = %d\n", DOM.Gfz.jsb, DOM.Gfz.jeb,
    DOM.Gfz.jnb);
  fprintf(outfile, "    ks = %d, ke = %d, kn = %d\n", DOM.Gfz.ks, DOM.Gfz.ke, DOM.Gfz.kn);
  fprintf(outfile, "    ksb = %d, keb = %d, knb = %d\n", DOM.Gfz.ksb, DOM.Gfz.keb,
    DOM.Gfz.knb);
  fprintf(outfile, "    _is = %d, _ie = %d, in = %d\n", DOM.Gfz._is, DOM.Gfz._ie,
    DOM.Gfz.in);
  fprintf(outfile, "    _isb = %d, _ieb = %d, inb = %d\n", DOM.Gfz._isb, DOM.Gfz._ieb,
    DOM.Gfz.inb);
  fprintf(outfile, "    _js = %d, _je = %d, jn = %d\n", DOM.Gfz._js, DOM.Gfz._je,
    DOM.Gfz.jn);
  fprintf(outfile, "    _jsb = %d, _jeb = %d, jnb = %d\n", DOM.Gfz._jsb, DOM.Gfz._jeb,
    DOM.Gfz.jnb);
  fprintf(outfile, "    _ks = %d, _ke = %d, kn = %d\n", DOM.Gfz._ks, DOM.Gfz._ke,
    DOM.Gfz.kn);
  fprintf(outfile, "    _ksb = %d, _keb = %d, knb = %d\n", DOM.Gfz._ksb, DOM.Gfz._keb,
    DOM.Gfz.knb);
  fprintf(outfile, "    s1 = %d, s2 = %d, s3 = %d\n", DOM.Gfz.s1, DOM.Gfz.s2,
    DOM.Gfz.s3);
  fprintf(outfile, "    s1b = %d, s2b = %d, s3b = %d\n", DOM.Gfz.s1b, DOM.Gfz.s2b,
    DOM.Gfz.s3b);

  fprintf(outfile, "MPI Domain Decomposition:\n");
  fprintf(outfile, "  nMPIdom = %d\n", nMPIdom);
  i = rank;
  fprintf(outfile, "MPI Subdomain %d:\n", i);
  fprintf(outfile, "  X: (%lf, %lf), dX = %f\n", Dom[i].xs, Dom[i].xe, Dom[i].dx);
  fprintf(outfile, "  Y: (%lf, %lf), dY = %f\n", Dom[i].ys, Dom[i].ye, Dom[i].dy);
  fprintf(outfile, "  Z: (%lf, %lf), dZ = %f\n", Dom[i].zs, Dom[i].ze, Dom[i].dz);
  fprintf(outfile, "  Xn = %d, Yn = %d, Zn = %d\n", Dom[i].xn, Dom[i].yn, Dom[i].zn);
  fprintf(outfile, "Connectivity:\n");
    fprintf(outfile, "  W: %d, E: %d, S: %d, N: %d, B: %d, T: %d\n", dom[i].W, dom[i].E,
      dom[i].S, dom[i].N, dom[i].B, dom[i].T);
  fprintf(outfile, "Grids:\n");
  fprintf(outfile, "  Dom[%d].Gcc:\n", i);
  fprintf(outfile, "    is = %d, ie = %d, in = %d\n", Dom[i].Gcc.is, Dom[i].Gcc.ie,
    Dom[i].Gcc.in);
  fprintf(outfile, "    isb = %d, ieb = %d, inb = %d\n", Dom[i].Gcc.isb, Dom[i].Gcc.ieb,
    Dom[i].Gcc.inb);
  fprintf(outfile, "    js = %d, je = %d, jn = %d\n", Dom[i].Gcc.js, Dom[i].Gcc.je,
    Dom[i].Gcc.jn);
  fprintf(outfile, "    jsb = %d, jeb = %d, jnb = %d\n", Dom[i].Gcc.jsb, Dom[i].Gcc.jeb,
    Dom[i].Gcc.jnb);
  fprintf(outfile, "    ks = %d, ke = %d, kn = %d\n", Dom[i].Gcc.ks, Dom[i].Gcc.ke,
    Dom[i].Gcc.kn);
  fprintf(outfile, "    ksb = %d, keb = %d, knb = %d\n", Dom[i].Gcc.ksb, Dom[i].Gcc.keb,
    Dom[i].Gcc.knb);
  fprintf(outfile, "    _is = %d, _ie = %d, in = %d\n", Dom[i].Gcc._is, Dom[i].Gcc._ie,
    Dom[i].Gcc.in);
  fprintf(outfile, "    _isb = %d, _ieb = %d, inb = %d\n", Dom[i].Gcc._isb,
    Dom[i].Gcc._ieb, Dom[i].Gcc.inb);
  fprintf(outfile, "    _js = %d, _je = %d, jn = %d\n", Dom[i].Gcc._js, Dom[i].Gcc._je,
    Dom[i].Gcc.jn);
  fprintf(outfile, "    _jsb = %d, _jeb = %d, jnb = %d\n", Dom[i].Gcc._jsb,
    Dom[i].Gcc._jeb, Dom[i].Gcc.jnb);
  fprintf(outfile, "    _ks = %d, _ke = %d, kn = %d\n", Dom[i].Gcc._ks, Dom[i].Gcc._ke,
    Dom[i].Gcc.kn);
  fprintf(outfile, "    _ksb = %d, _keb = %d, knb = %d\n", Dom[i].Gcc._ksb,
    Dom[i].Gcc._keb, Dom[i].Gcc.knb);
  fprintf(outfile, "    s1 = %d, s2 = %d, s3 = %d\n", Dom[i].Gcc.s1, Dom[i].Gcc.s2,
    Dom[i].Gcc.s3);
  fprintf(outfile, "    s1b = %d, s2b = %d, s3b = %d\n", Dom[i].Gcc.s1b, Dom[i].Gcc.s2b,
    Dom[i].Gcc.s3b);
  fprintf(outfile, "  Dom[%d].Gfx:\n", i);
  fprintf(outfile, "    is = %d, ie = %d, in = %d\n", Dom[i].Gfx.is, Dom[i].Gfx.ie,
    Dom[i].Gfx.in);
  fprintf(outfile, "    isb = %d, ieb = %d, inb = %d\n", Dom[i].Gfx.isb, Dom[i].Gfx.ieb,
    Dom[i].Gfx.inb);
  fprintf(outfile, "    js = %d, je = %d, jn = %d\n", Dom[i].Gfx.js, Dom[i].Gfx.je,
    Dom[i].Gfx.jn);
  fprintf(outfile, "    jsb = %d, jeb = %d, jnb = %d\n", Dom[i].Gfx.jsb, Dom[i].Gfx.jeb,
    Dom[i].Gfx.jnb);
  fprintf(outfile, "    ks = %d, ke = %d, kn = %d\n", Dom[i].Gfx.ks, Dom[i].Gfx.ke,
    Dom[i].Gfx.kn);
  fprintf(outfile, "    ksb = %d, keb = %d, knb = %d\n", Dom[i].Gfx.ksb, Dom[i].Gfx.keb,
    Dom[i].Gfx.knb);
  fprintf(outfile, "    _is = %d, _ie = %d, in = %d\n", Dom[i].Gfx._is, Dom[i].Gfx._ie,
    Dom[i].Gfx.in);
  fprintf(outfile, "    _isb = %d, _ieb = %d, inb = %d\n", Dom[i].Gfx._isb,
    Dom[i].Gfx._ieb, Dom[i].Gfx.inb);
  fprintf(outfile, "    _js = %d, _je = %d, jn = %d\n", Dom[i].Gfx._js, Dom[i].Gfx._je,
    Dom[i].Gfx.jn);
  fprintf(outfile, "    _jsb = %d, _jeb = %d, jnb = %d\n", Dom[i].Gfx._jsb,
    Dom[i].Gfx._jeb, Dom[i].Gfx.jnb);
  fprintf(outfile, "    _ks = %d, _ke = %d, kn = %d\n", Dom[i].Gfx._ks, Dom[i].Gfx._ke,
    Dom[i].Gfx.kn);
  fprintf(outfile, "    _ksb = %d, _keb = %d, knb = %d\n", Dom[i].Gfx._ksb,
    Dom[i].Gfx._keb, Dom[i].Gfx.knb);
  fprintf(outfile, "    s1 = %d, s2 = %d, s3 = %d\n", Dom[i].Gfx.s1, Dom[i].Gfx.s2,
    Dom[i].Gfx.s3);
  fprintf(outfile, "    s1b = %d, s2b = %d, s3b = %d\n", Dom[i].Gfx.s1b, Dom[i].Gfx.s2b,
    Dom[i].Gfx.s3b);
  fprintf(outfile, "  Dom[%d].Gfy:\n", i);
  fprintf(outfile, "    is = %d, ie = %d, in = %d\n", Dom[i].Gfy.is, Dom[i].Gfy.ie,
    Dom[i].Gfy.in);
  fprintf(outfile, "    isb = %d, ieb = %d, inb = %d\n", Dom[i].Gfy.isb, Dom[i].Gfy.ieb,
    Dom[i].Gfy.inb);
  fprintf(outfile, "    js = %d, je = %d, jn = %d\n", Dom[i].Gfy.js, Dom[i].Gfy.je,
    Dom[i].Gfy.jn);
  fprintf(outfile, "    jsb = %d, jeb = %d, jnb = %d\n", Dom[i].Gfy.jsb, Dom[i].Gfy.jeb,
    Dom[i].Gfy.jnb);
  fprintf(outfile, "    ks = %d, ke = %d, kn = %d\n", Dom[i].Gfy.ks, Dom[i].Gfy.ke,
    Dom[i].Gfy.kn);
  fprintf(outfile, "    ksb = %d, keb = %d, knb = %d\n", Dom[i].Gfy.ksb, Dom[i].Gfy.keb,
    Dom[i].Gfy.knb);
  fprintf(outfile, "    _is = %d, _ie = %d, in = %d\n", Dom[i].Gfy._is, Dom[i].Gfy._ie,
    Dom[i].Gfy.in);
  fprintf(outfile, "    _isb = %d, _ieb = %d, inb = %d\n", Dom[i].Gfy._isb,
    Dom[i].Gfy._ieb, Dom[i].Gfy.inb);
  fprintf(outfile, "    _js = %d, _je = %d, jn = %d\n", Dom[i].Gfy._js, Dom[i].Gfy._je,
    Dom[i].Gfy.jn);
  fprintf(outfile, "    _jsb = %d, _jeb = %d, jnb = %d\n", Dom[i].Gfy._jsb,
    Dom[i].Gfy._jeb, Dom[i].Gfy.jnb);
  fprintf(outfile, "    _ks = %d, _ke = %d, kn = %d\n", Dom[i].Gfy._ks, Dom[i].Gfy._ke,
    Dom[i].Gfy.kn);
  fprintf(outfile, "    _ksb = %d, _keb = %d, knb = %d\n", Dom[i].Gfy._ksb,
    Dom[i].Gfy._keb, Dom[i].Gfy.knb);
  fprintf(outfile, "    s1 = %d, s2 = %d, s3 = %d\n", Dom[i].Gfy.s1, Dom[i].Gfy.s2,
    Dom[i].Gfy.s3);
  fprintf(outfile, "    s1b = %d, s2b = %d, s3b = %d\n", Dom[i].Gfy.s1b, Dom[i].Gfy.s2b,
    Dom[i].Gfy.s3b);
  fprintf(outfile, "  Dom[%d].Gfz:\n", i);
  fprintf(outfile, "    is = %d, ie = %d, in = %d\n", Dom[i].Gfz.is, Dom[i].Gfz.ie,
    Dom[i].Gfz.in);
  fprintf(outfile, "    isb = %d, ieb = %d, inb = %d\n", Dom[i].Gfz.isb, Dom[i].Gfz.ieb,
    Dom[i].Gfz.inb);
  fprintf(outfile, "    js = %d, je = %d, jn = %d\n", Dom[i].Gfz.js, Dom[i].Gfz.je,
    Dom[i].Gfz.jn);
  fprintf(outfile, "    jsb = %d, jeb = %d, jnb = %d\n", Dom[i].Gfz.jsb, Dom[i].Gfz.jeb,
    Dom[i].Gfz.jnb);
  fprintf(outfile, "    ks = %d, ke = %d, kn = %d\n", Dom[i].Gfz.ks, Dom[i].Gfz.ke,
    Dom[i].Gfz.kn);
  fprintf(outfile, "    ksb = %d, keb = %d, knb = %d\n", Dom[i].Gfz.ksb, Dom[i].Gfz.keb,
    Dom[i].Gfz.knb);
  fprintf(outfile, "    _is = %d, _ie = %d, in = %d\n", Dom[i].Gfz._is, Dom[i].Gfz._ie,
    Dom[i].Gfz.in);
  fprintf(outfile, "    _isb = %d, _ieb = %d, inb = %d\n", Dom[i].Gfz._isb,
    Dom[i].Gfz._ieb, Dom[i].Gfz.inb);
  fprintf(outfile, "    _js = %d, _je = %d, jn = %d\n", Dom[i].Gfz._js, Dom[i].Gfz._je,
    Dom[i].Gfz.jn);
  fprintf(outfile, "    _jsb = %d, _jeb = %d, jnb = %d\n", Dom[i].Gfz._jsb,
    Dom[i].Gfz._jeb, Dom[i].Gfz.jnb);
  fprintf(outfile, "    _ks = %d, _ke = %d, kn = %d\n", Dom[i].Gfz._ks, Dom[i].Gfz._ke,
    Dom[i].Gfz.kn);
  fprintf(outfile, "    _ksb = %d, _keb = %d, knb = %d\n", Dom[i].Gfz._ksb,
    Dom[i].Gfz._keb, Dom[i].Gfz.knb);
  fprintf(outfile, "    s1 = %d, s2 = %d, s3 = %d\n", Dom[i].Gfz.s1, Dom[i].Gfz.s2,
    Dom[i].Gfz.s3);
  fprintf(outfile, "    s1b = %d, s2b = %d, s3b = %d\n", Dom[i].Gfz.s1b, Dom[i].Gfz.s2b,
    Dom[i].Gfz.s3b);

  fprintf(outfile, "GPU Domain Decomposition:\n");
  fprintf(outfile, "  nGPUdom = %d\n", nGPUdom);
  for(j = 0; j < nodes[rank].ndevs; j++) {
    i = nodes[rank].devstart + j;
    fprintf(outfile, "GPU Subdomain %d:\n", i);
    fprintf(outfile, "  X: (%lf, %lf), dX = %f\n", dom[i].xs, dom[i].xe, dom[i].dx);
    fprintf(outfile, "  Y: (%lf, %lf), dY = %f\n", dom[i].ys, dom[i].ye, dom[i].dy);
    fprintf(outfile, "  Z: (%lf, %lf), dZ = %f\n", dom[i].zs, dom[i].ze, dom[i].dz);
    fprintf(outfile, "  Xn = %d, Yn = %d, Zn = %d\n", dom[i].xn, dom[i].yn, dom[i].zn);
    fprintf(outfile, "Connectivity:\n");
    fprintf(outfile, "  W: %d, E: %d, S: %d, N: %d, B: %d, T: %d\n", dom[i].W, dom[i].E,
      dom[i].S, dom[i].N, dom[i].B, dom[i].T);
    fprintf(outfile, "  w: %d, e: %d, s: %d, n: %d, b: %d, t: %d\n", dom[i].w, dom[i].e,
      dom[i].s, dom[i].n, dom[i].b, dom[i].w);
    fprintf(outfile, "Grids:\n");
    fprintf(outfile, "  dom[%d].Gcc:\n", i);
    fprintf(outfile, "    is = %d, ie = %d, in = %d\n", dom[i].Gcc.is, dom[i].Gcc.ie,
      dom[i].Gcc.in);
    fprintf(outfile, "    isb = %d, ieb = %d, inb = %d\n", dom[i].Gcc.isb, dom[i].Gcc.ieb,
      dom[i].Gcc.inb);
    fprintf(outfile, "    js = %d, je = %d, jn = %d\n", dom[i].Gcc.js, dom[i].Gcc.je,
      dom[i].Gcc.jn);
    fprintf(outfile, "    jsb = %d, jeb = %d, jnb = %d\n", dom[i].Gcc.jsb, dom[i].Gcc.jeb,
      dom[i].Gcc.jnb);
    fprintf(outfile, "    ks = %d, ke = %d, kn = %d\n", dom[i].Gcc.ks, dom[i].Gcc.ke,
      dom[i].Gcc.kn);
    fprintf(outfile, "    ksb = %d, keb = %d, knb = %d\n", dom[i].Gcc.ksb, dom[i].Gcc.keb,
      dom[i].Gcc.knb);
    fprintf(outfile, "    _is = %d, _ie = %d, in = %d\n", dom[i].Gcc._is, dom[i].Gcc._ie,
      dom[i].Gcc.in);
    fprintf(outfile, "    _isb = %d, _ieb = %d, inb = %d\n", dom[i].Gcc._isb,
      dom[i].Gcc._ieb, dom[i].Gcc.inb);
    fprintf(outfile, "    _js = %d, _je = %d, jn = %d\n", dom[i].Gcc._js, dom[i].Gcc._je,
      dom[i].Gcc.jn);
    fprintf(outfile, "    _jsb = %d, _jeb = %d, jnb = %d\n", dom[i].Gcc._jsb,
      dom[i].Gcc._jeb, dom[i].Gcc.jnb);
    fprintf(outfile, "    _ks = %d, _ke = %d, kn = %d\n", dom[i].Gcc._ks, dom[i].Gcc._ke,
      dom[i].Gcc.kn);
    fprintf(outfile, "    _ksb = %d, _keb = %d, knb = %d\n", dom[i].Gcc._ksb,
      dom[i].Gcc._keb, dom[i].Gcc.knb);
    fprintf(outfile, "    s1 = %d, s2 = %d, s3 = %d\n", dom[i].Gcc.s1, dom[i].Gcc.s2,
      dom[i].Gcc.s3);
    fprintf(outfile, "    s1b = %d, s2b = %d, s3b = %d\n", dom[i].Gcc.s1b, dom[i].Gcc.s2b,
      dom[i].Gcc.s3b);
    fprintf(outfile, "  dom[%d].Gfx:\n", i);
    fprintf(outfile, "    is = %d, ie = %d, in = %d\n", dom[i].Gfx.is, dom[i].Gfx.ie,
      dom[i].Gfx.in);
    fprintf(outfile, "    isb = %d, ieb = %d, inb = %d\n", dom[i].Gfx.isb, dom[i].Gfx.ieb,
      dom[i].Gfx.inb);
    fprintf(outfile, "    js = %d, je = %d, jn = %d\n", dom[i].Gfx.js, dom[i].Gfx.je,
      dom[i].Gfx.jn);
    fprintf(outfile, "    jsb = %d, jeb = %d, jnb = %d\n", dom[i].Gfx.jsb, dom[i].Gfx.jeb,
      dom[i].Gfx.jnb);
    fprintf(outfile, "    ks = %d, ke = %d, kn = %d\n", dom[i].Gfx.ks, dom[i].Gfx.ke,
      dom[i].Gfx.kn);
    fprintf(outfile, "    ksb = %d, keb = %d, knb = %d\n", dom[i].Gfx.ksb, dom[i].Gfx.keb,
      dom[i].Gfx.knb);
    fprintf(outfile, "    _is = %d, _ie = %d, in = %d\n", dom[i].Gfx._is, dom[i].Gfx._ie,
      dom[i].Gfx.in);
    fprintf(outfile, "    _isb = %d, _ieb = %d, inb = %d\n", dom[i].Gfx._isb,
      dom[i].Gfx._ieb, dom[i].Gfx.inb);
    fprintf(outfile, "    _js = %d, _je = %d, jn = %d\n", dom[i].Gfx._js, dom[i].Gfx._je,
      dom[i].Gfx.jn);
    fprintf(outfile, "    _jsb = %d, _jeb = %d, jnb = %d\n", dom[i].Gfx._jsb,
      dom[i].Gfx._jeb, dom[i].Gfx.jnb);
    fprintf(outfile, "    _ks = %d, _ke = %d, kn = %d\n", dom[i].Gfx._ks, dom[i].Gfx._ke,
      dom[i].Gfx.kn);
    fprintf(outfile, "    _ksb = %d, _keb = %d, knb = %d\n", dom[i].Gfx._ksb,
      dom[i].Gfx._keb, dom[i].Gfx.knb);
    fprintf(outfile, "    s1 = %d, s2 = %d, s3 = %d\n", dom[i].Gfx.s1, dom[i].Gfx.s2,
      dom[i].Gfx.s3);
    fprintf(outfile, "    s1b = %d, s2b = %d, s3b = %d\n", dom[i].Gfx.s1b, dom[i].Gfx.s2b,
      dom[i].Gfx.s3b);
    fprintf(outfile, "  dom[%d].Gfy:\n", i);
    fprintf(outfile, "    is = %d, ie = %d, in = %d\n", dom[i].Gfy.is, dom[i].Gfy.ie,
      dom[i].Gfy.in);
    fprintf(outfile, "    isb = %d, ieb = %d, inb = %d\n", dom[i].Gfy.isb, dom[i].Gfy.ieb,
      dom[i].Gfy.inb);
    fprintf(outfile, "    js = %d, je = %d, jn = %d\n", dom[i].Gfy.js, dom[i].Gfy.je,
      dom[i].Gfy.jn);
    fprintf(outfile, "    jsb = %d, jeb = %d, jnb = %d\n", dom[i].Gfy.jsb, dom[i].Gfy.jeb,
      dom[i].Gfy.jnb);
    fprintf(outfile, "    ks = %d, ke = %d, kn = %d\n", dom[i].Gfy.ks, dom[i].Gfy.ke,
      dom[i].Gfy.kn);
    fprintf(outfile, "    ksb = %d, keb = %d, knb = %d\n", dom[i].Gfy.ksb, dom[i].Gfy.keb,
      dom[i].Gfy.knb);
    fprintf(outfile, "    _is = %d, _ie = %d, in = %d\n", dom[i].Gfy._is, dom[i].Gfy._ie,
      dom[i].Gfy.in);
    fprintf(outfile, "    _isb = %d, _ieb = %d, inb = %d\n", dom[i].Gfy._isb,
      dom[i].Gfy._ieb, dom[i].Gfy.inb);
    fprintf(outfile, "    _js = %d, _je = %d, jn = %d\n", dom[i].Gfy._js, dom[i].Gfy._je,
      dom[i].Gfy.jn);
    fprintf(outfile, "    _jsb = %d, _jeb = %d, jnb = %d\n", dom[i].Gfy._jsb,
      dom[i].Gfy._jeb, dom[i].Gfy.jnb);
    fprintf(outfile, "    _ks = %d, _ke = %d, kn = %d\n", dom[i].Gfy._ks, dom[i].Gfy._ke,
      dom[i].Gfy.kn);
    fprintf(outfile, "    _ksb = %d, _keb = %d, knb = %d\n", dom[i].Gfy._ksb,
      dom[i].Gfy._keb, dom[i].Gfy.knb);
    fprintf(outfile, "    s1 = %d, s2 = %d, s3 = %d\n", dom[i].Gfy.s1, dom[i].Gfy.s2,
      dom[i].Gfy.s3);
    fprintf(outfile, "    s1b = %d, s2b = %d, s3b = %d\n", dom[i].Gfy.s1b, dom[i].Gfy.s2b,
      dom[i].Gfy.s3b);
    fprintf(outfile, "  dom[%d].Gfz:\n", i);
    fprintf(outfile, "    is = %d, ie = %d, in = %d\n", dom[i].Gfz.is, dom[i].Gfz.ie,
      dom[i].Gfz.in);
    fprintf(outfile, "    isb = %d, ieb = %d, inb = %d\n", dom[i].Gfz.isb, dom[i].Gfz.ieb,
      dom[i].Gfz.inb);
    fprintf(outfile, "    js = %d, je = %d, jn = %d\n", dom[i].Gfz.js, dom[i].Gfz.je,
      dom[i].Gfz.jn);
    fprintf(outfile, "    jsb = %d, jeb = %d, jnb = %d\n", dom[i].Gfz.jsb, dom[i].Gfz.jeb,
      dom[i].Gfz.jnb);
    fprintf(outfile, "    ks = %d, ke = %d, kn = %d\n", dom[i].Gfz.ks, dom[i].Gfz.ke,
      dom[i].Gfz.kn);
    fprintf(outfile, "    ksb = %d, keb = %d, knb = %d\n", dom[i].Gfz.ksb, dom[i].Gfz.keb,
      dom[i].Gfz.knb);
    fprintf(outfile, "    _is = %d, _ie = %d, in = %d\n", dom[i].Gfz._is, dom[i].Gfz._ie,
      dom[i].Gfz.in);
    fprintf(outfile, "    _isb = %d, _ieb = %d, inb = %d\n", dom[i].Gfz._isb,
      dom[i].Gfz._ieb, dom[i].Gfz.inb);
    fprintf(outfile, "    _js = %d, _je = %d, jn = %d\n", dom[i].Gfz._js, dom[i].Gfz._je,
      dom[i].Gfz.jn);
    fprintf(outfile, "    _jsb = %d, _jeb = %d, jnb = %d\n", dom[i].Gfz._jsb,
      dom[i].Gfz._jeb, dom[i].Gfz.jnb);
    fprintf(outfile, "    _ks = %d, _ke = %d, kn = %d\n", dom[i].Gfz._ks, dom[i].Gfz._ke,
      dom[i].Gfz.kn);
    fprintf(outfile, "    _ksb = %d, _keb = %d, knb = %d\n", dom[i].Gfz._ksb,
      dom[i].Gfz._keb, dom[i].Gfz.knb);
    fprintf(outfile, "    s1 = %d, s2 = %d, s3 = %d\n", dom[i].Gfz.s1, dom[i].Gfz.s2,
      dom[i].Gfz.s3);
    fprintf(outfile, "    s1b = %d, s2b = %d, s3b = %d\n", dom[i].Gfz.s1b, dom[i].Gfz.s2b,
      dom[i].Gfz.s3b);
  }
  fprintf(outfile, "Simulation Parameters:\n");
  fprintf(outfile, "  pp_max_iter = %d\n", pp_max_iter);
  fprintf(outfile, "  pp_residual = %e\n", pp_residual);
  fprintf(outfile, "Boundary Conditions: (0 = PERIODIC, 1 = DIRICHLET, 2 = NEUMANN)\n");
  fprintf(outfile, "  bc.pW = %d", bc.pW);
  fprintf(outfile, ", bc.pE = %d", bc.pE);
  fprintf(outfile, "\n");
  fprintf(outfile, "  bc.pS = %d", bc.pS);
  fprintf(outfile, ", bc.pN = %d", bc.pN);
  fprintf(outfile, "\n");
  fprintf(outfile, "  bc.pB = %d", bc.pB);
  fprintf(outfile, ", bc.pT = %d", bc.pT);
  fprintf(outfile, "\n");

  fclose(outfile);
}

void domain_map(void)
{
  int i;  // iterators

  // calculate domain sizes
  DOM.xl = DOM.xe - DOM.xs;
  DOM.yl = DOM.ye - DOM.ys;
  DOM.zl = DOM.ze - DOM.zs;

  // calculate cell sizes
  DOM.dx = DOM.xl / DOM.xn;
  DOM.dy = DOM.yl / DOM.yn;
  DOM.dz = DOM.zl / DOM.zn;

  // set up grids
  // Gcc
  DOM.Gcc.is = DOM_BUF;
  DOM.Gcc.isb = DOM.Gcc.is - DOM_BUF;
  DOM.Gcc.in = DOM.xn;
  DOM.Gcc.inb = DOM.Gcc.in + 2 * DOM_BUF;
  DOM.Gcc.ie = DOM.Gcc.isb + DOM.Gcc.in;
  DOM.Gcc.ieb = DOM.Gcc.ie + DOM_BUF;

  DOM.Gcc.js = DOM_BUF;
  DOM.Gcc.jsb = DOM.Gcc.js - DOM_BUF;
  DOM.Gcc.jn = DOM.yn;
  DOM.Gcc.jnb = DOM.Gcc.jn + 2 * DOM_BUF;
  DOM.Gcc.je = DOM.Gcc.jsb + DOM.Gcc.jn;
  DOM.Gcc.jeb = DOM.Gcc.je + DOM_BUF;

  DOM.Gcc.ks = DOM_BUF;
  DOM.Gcc.ksb = DOM.Gcc.ks - DOM_BUF;
  DOM.Gcc.kn = DOM.zn;
  DOM.Gcc.knb = DOM.Gcc.kn + 2 * DOM_BUF;
  DOM.Gcc.ke = DOM.Gcc.ksb + DOM.Gcc.kn;
  DOM.Gcc.keb = DOM.Gcc.ke + DOM_BUF;

  DOM.Gcc._is = DOM.Gcc.is;
  DOM.Gcc._isb = DOM.Gcc.isb;
  DOM.Gcc.in = DOM.Gcc.in;
  DOM.Gcc.inb = DOM.Gcc.inb;
  DOM.Gcc._ie = DOM.Gcc.ie;
  DOM.Gcc._ieb = DOM.Gcc.ieb;

  DOM.Gcc._js = DOM.Gcc.js;
  DOM.Gcc._jsb = DOM.Gcc.jsb;
  DOM.Gcc.jn = DOM.Gcc.jn;
  DOM.Gcc.jnb = DOM.Gcc.jnb;
  DOM.Gcc._je = DOM.Gcc.je;
  DOM.Gcc._jeb = DOM.Gcc.jeb;

  DOM.Gcc._ks = DOM.Gcc.ks;
  DOM.Gcc._ksb = DOM.Gcc.ksb;
  DOM.Gcc.kn = DOM.Gcc.kn;
  DOM.Gcc.knb = DOM.Gcc.knb;
  DOM.Gcc._ke = DOM.Gcc.ke;
  DOM.Gcc._keb = DOM.Gcc.keb;

  DOM.Gcc.s1 = DOM.Gcc.in;
  DOM.Gcc.s2 = DOM.Gcc.s1 * DOM.Gcc.jn;
  DOM.Gcc.s3 = DOM.Gcc.s2 * DOM.Gcc.kn;
  DOM.Gcc.s1b = DOM.Gcc.inb;
  DOM.Gcc.s2b = DOM.Gcc.s1b * DOM.Gcc.jnb;
  DOM.Gcc.s3b = DOM.Gcc.s2b * DOM.Gcc.knb;

  // Gfx
  DOM.Gfx.is = DOM_BUF;
  DOM.Gfx.isb = DOM.Gfx.is - DOM_BUF;
  DOM.Gfx.in = DOM.xn + 1;
  DOM.Gfx.inb = DOM.Gfx.in + 2 * DOM_BUF;
  DOM.Gfx.ie = DOM.Gfx.isb + DOM.Gfx.in;
  DOM.Gfx.ieb = DOM.Gfx.ie + DOM_BUF;

  DOM.Gfx.js = DOM_BUF;
  DOM.Gfx.jsb = DOM.Gfx.js - DOM_BUF;
  DOM.Gfx.jn = DOM.yn;
  DOM.Gfx.jnb = DOM.Gfx.jn + 2 * DOM_BUF;
  DOM.Gfx.je = DOM.Gfx.jsb + DOM.Gfx.jn;
  DOM.Gfx.jeb = DOM.Gfx.je + DOM_BUF;

  DOM.Gfx.ks = DOM_BUF;
  DOM.Gfx.ksb = DOM.Gfx.ks - DOM_BUF;
  DOM.Gfx.kn = DOM.zn;
  DOM.Gfx.knb = DOM.Gfx.kn + 2 * DOM_BUF;
  DOM.Gfx.ke = DOM.Gfx.ksb + DOM.Gfx.kn;
  DOM.Gfx.keb = DOM.Gfx.ke + DOM_BUF;

  DOM.Gfx._is = DOM.Gfx.is;
  DOM.Gfx._isb = DOM.Gfx.isb;
  DOM.Gfx.in = DOM.Gfx.in;
  DOM.Gfx.inb = DOM.Gfx.inb;
  DOM.Gfx._ie = DOM.Gfx.ie;
  DOM.Gfx._ieb = DOM.Gfx.ieb;

  DOM.Gfx._js = DOM.Gfx.js;
  DOM.Gfx._jsb = DOM.Gfx.jsb;
  DOM.Gfx.jn = DOM.Gfx.jn;
  DOM.Gfx.jnb = DOM.Gfx.jnb;
  DOM.Gfx._je = DOM.Gfx.je;
  DOM.Gfx._jeb = DOM.Gfx.jeb;

  DOM.Gfx._ks = DOM.Gfx.ks;
  DOM.Gfx._ksb = DOM.Gfx.ksb;
  DOM.Gfx.kn = DOM.Gfx.kn;
  DOM.Gfx.knb = DOM.Gfx.knb;
  DOM.Gfx._ke = DOM.Gfx.ke;
  DOM.Gfx._keb = DOM.Gfx.keb;

  DOM.Gfx.s1 = DOM.Gfx.in;
  DOM.Gfx.s2 = DOM.Gfx.s1 * DOM.Gfx.jn;
  DOM.Gfx.s3 = DOM.Gfx.s2 * DOM.Gfx.kn;
  DOM.Gfx.s1b = DOM.Gfx.inb;
  DOM.Gfx.s2b = DOM.Gfx.s1b * DOM.Gfx.jnb;
  DOM.Gfx.s3b = DOM.Gfx.s2b * DOM.Gfx.knb;

  // Gfy
  DOM.Gfy.is = DOM_BUF;
  DOM.Gfy.isb = DOM.Gfy.is - DOM_BUF;
  DOM.Gfy.in = DOM.xn;
  DOM.Gfy.inb = DOM.Gfy.in + 2 * DOM_BUF;
  DOM.Gfy.ie = DOM.Gfy.isb + DOM.Gfy.in;
  DOM.Gfy.ieb = DOM.Gfy.ie + DOM_BUF;

  DOM.Gfy.js = DOM_BUF;
  DOM.Gfy.jsb = DOM.Gfy.js - DOM_BUF;
  DOM.Gfy.jn = DOM.yn + 1;
  DOM.Gfy.jnb = DOM.Gfy.jn + 2 * DOM_BUF;
  DOM.Gfy.je = DOM.Gfy.jsb + DOM.Gfy.jn;
  DOM.Gfy.jeb = DOM.Gfy.je + DOM_BUF;

  DOM.Gfy.ks = DOM_BUF;
  DOM.Gfy.ksb = DOM.Gfy.ks - DOM_BUF;
  DOM.Gfy.kn = DOM.zn;
  DOM.Gfy.knb = DOM.Gfy.kn + 2 * DOM_BUF;
  DOM.Gfy.ke = DOM.Gfy.ksb + DOM.Gfy.kn;
  DOM.Gfy.keb = DOM.Gfy.ke + DOM_BUF;

  DOM.Gfy._is = DOM.Gfy.is;
  DOM.Gfy._isb = DOM.Gfy.isb;
  DOM.Gfy.in = DOM.Gfy.in;
  DOM.Gfy.inb = DOM.Gfy.inb;
  DOM.Gfy._ie = DOM.Gfy.ie;
  DOM.Gfy._ieb = DOM.Gfy.ieb;

  DOM.Gfy._js = DOM.Gfy.js;
  DOM.Gfy._jsb = DOM.Gfy.jsb;
  DOM.Gfy.jn = DOM.Gfy.jn;
  DOM.Gfy.jnb = DOM.Gfy.jnb;
  DOM.Gfy._je = DOM.Gfy.je;
  DOM.Gfy._jeb = DOM.Gfy.jeb;

  DOM.Gfy._ks = DOM.Gfy.ks;
  DOM.Gfy._ksb = DOM.Gfy.ksb;
  DOM.Gfy.kn = DOM.Gfy.kn;
  DOM.Gfy.knb = DOM.Gfy.knb;
  DOM.Gfy._ke = DOM.Gfy.ke;
  DOM.Gfy._keb = DOM.Gfy.keb;

  DOM.Gfy.s1 = DOM.Gfy.in;
  DOM.Gfy.s2 = DOM.Gfy.s1 * DOM.Gfy.jn;
  DOM.Gfy.s3 = DOM.Gfy.s2 * DOM.Gfy.kn;
  DOM.Gfy.s1b = DOM.Gfy.inb;
  DOM.Gfy.s2b = DOM.Gfy.s1b * DOM.Gfy.jnb;
  DOM.Gfy.s3b = DOM.Gfy.s2b * DOM.Gfy.knb;

  // Gfz
  DOM.Gfz.is = DOM_BUF;
  DOM.Gfz.isb = DOM.Gfz.is - DOM_BUF;
  DOM.Gfz.in = DOM.xn;
  DOM.Gfz.inb = DOM.Gfz.in + 2 * DOM_BUF;
  DOM.Gfz.ie = DOM.Gfz.isb + DOM.Gfz.in;
  DOM.Gfz.ieb = DOM.Gfz.ie + DOM_BUF;

  DOM.Gfz.js = DOM_BUF;
  DOM.Gfz.jsb = DOM.Gfz.js - DOM_BUF;
  DOM.Gfz.jn = DOM.yn;
  DOM.Gfz.jnb = DOM.Gfz.jn + 2 * DOM_BUF;
  DOM.Gfz.je = DOM.Gfz.jsb + DOM.Gfz.jn;
  DOM.Gfz.jeb = DOM.Gfz.je + DOM_BUF;

  DOM.Gfz.ks = DOM_BUF;
  DOM.Gfz.ksb = DOM.Gfz.ks - DOM_BUF;
  DOM.Gfz.kn = DOM.zn + 1;
  DOM.Gfz.knb = DOM.Gfz.kn + 2 * DOM_BUF;
  DOM.Gfz.ke = DOM.Gfz.ksb + DOM.Gfz.kn;
  DOM.Gfz.keb = DOM.Gfz.ke + DOM_BUF;

  DOM.Gfz._is = DOM.Gfz.is;
  DOM.Gfz._isb = DOM.Gfz.isb;
  DOM.Gfz.in = DOM.Gfz.in;
  DOM.Gfz.inb = DOM.Gfz.inb;
  DOM.Gfz._ie = DOM.Gfz.ie;
  DOM.Gfz._ieb = DOM.Gfz.ieb;

  DOM.Gfz._js = DOM.Gfz.js;
  DOM.Gfz._jsb = DOM.Gfz.jsb;
  DOM.Gfz.jn = DOM.Gfz.jn;
  DOM.Gfz.jnb = DOM.Gfz.jnb;
  DOM.Gfz._je = DOM.Gfz.je;
  DOM.Gfz._jeb = DOM.Gfz.jeb;

  DOM.Gfz._ks = DOM.Gfz.ks;
  DOM.Gfz._ksb = DOM.Gfz.ksb;
  DOM.Gfz.kn = DOM.Gfz.kn;
  DOM.Gfz.knb = DOM.Gfz.knb;
  DOM.Gfz._ke = DOM.Gfz.ke;
  DOM.Gfz._keb = DOM.Gfz.keb;

  DOM.Gfz.s1 = DOM.Gfz.in;
  DOM.Gfz.s2 = DOM.Gfz.s1 * DOM.Gfz.jn;
  DOM.Gfz.s3 = DOM.Gfz.s2 * DOM.Gfz.kn;
  DOM.Gfz.s1b = DOM.Gfz.inb;
  DOM.Gfz.s2b = DOM.Gfz.s1b * DOM.Gfz.jnb;
  DOM.Gfz.s3b = DOM.Gfz.s2b * DOM.Gfz.knb;

  // configure MPI subdomains
  /* These subdomains must be provided in increasing order */
  for(i = 0; i < nMPIdom; i++) {
    Dom[i].xl = Dom[i].xe - Dom[i].xs;
    Dom[i].yl = Dom[i].ye - Dom[i].ys;
    Dom[i].zl = Dom[i].ze - Dom[i].zs;
    Dom[i].dx = Dom[i].xl / Dom[i].xn;
    Dom[i].dy = Dom[i].yl / Dom[i].yn;
    Dom[i].dz = Dom[i].zl / Dom[i].zn;

    // Gcc
    if(Dom[i].W > -1)
      Dom[i].Gcc.is = Dom[Dom[i].W].Gcc.ie + 1;
    else
      Dom[i].Gcc.is = DOM_BUF;
    Dom[i].Gcc.isb = Dom[i].Gcc.is - DOM_BUF;
    Dom[i].Gcc.in = Dom[i].xn;
    Dom[i].Gcc.inb = Dom[i].Gcc.in + 2 * DOM_BUF;
    Dom[i].Gcc.ie = Dom[i].Gcc.isb + Dom[i].Gcc.in;
    Dom[i].Gcc.ieb = Dom[i].Gcc.ie + DOM_BUF;

    if(Dom[i].S > -1)
      Dom[i].Gcc.js = Dom[Dom[i].S].Gcc.je + 1;
    else
      Dom[i].Gcc.js = DOM_BUF;
    Dom[i].Gcc.jsb = Dom[i].Gcc.js - DOM_BUF;
    Dom[i].Gcc.jn = Dom[i].yn;
    Dom[i].Gcc.jnb = Dom[i].Gcc.jn + 2 * DOM_BUF;
    Dom[i].Gcc.je = Dom[i].Gcc.jsb + Dom[i].Gcc.jn;
    Dom[i].Gcc.jeb = Dom[i].Gcc.je + DOM_BUF;

    if(Dom[i].B > -1)
      Dom[i].Gcc.ks = Dom[Dom[i].B].Gcc.ke + 1;
    else
      Dom[i].Gcc.ks = DOM_BUF;
    Dom[i].Gcc.ksb = Dom[i].Gcc.ks - DOM_BUF;
    Dom[i].Gcc.kn = Dom[i].zn;
    Dom[i].Gcc.knb = Dom[i].Gcc.kn + 2 * DOM_BUF;
    Dom[i].Gcc.ke = Dom[i].Gcc.ksb + Dom[i].Gcc.kn;
    Dom[i].Gcc.keb = Dom[i].Gcc.ke + DOM_BUF;

    Dom[i].Gcc._is = DOM_BUF;
    Dom[i].Gcc._isb = Dom[i].Gcc._is - DOM_BUF;
    Dom[i].Gcc.in = Dom[i].xn;
    Dom[i].Gcc.inb = Dom[i].Gcc.in + 2 * DOM_BUF;
    Dom[i].Gcc._ie = Dom[i].Gcc._isb + Dom[i].Gcc.in;
    Dom[i].Gcc._ieb = Dom[i].Gcc._ie + DOM_BUF;

    Dom[i].Gcc._js = DOM_BUF;
    Dom[i].Gcc._jsb = Dom[i].Gcc._js - DOM_BUF;
    Dom[i].Gcc.jn = Dom[i].yn;
    Dom[i].Gcc.jnb = Dom[i].Gcc.jn + 2 * DOM_BUF;
    Dom[i].Gcc._je = Dom[i].Gcc._jsb + Dom[i].Gcc.jn;
    Dom[i].Gcc._jeb = Dom[i].Gcc._je + DOM_BUF;

    Dom[i].Gcc._ks = DOM_BUF;
    Dom[i].Gcc._ksb = Dom[i].Gcc._ks - DOM_BUF;
    Dom[i].Gcc.kn = Dom[i].zn;
    Dom[i].Gcc.knb = Dom[i].Gcc.kn + 2 * DOM_BUF;
    Dom[i].Gcc._ke = Dom[i].Gcc._ksb + Dom[i].Gcc.kn;
    Dom[i].Gcc._keb = Dom[i].Gcc._ke + DOM_BUF;

    Dom[i].Gcc.s1 = Dom[i].Gcc.in;
    Dom[i].Gcc.s2 = Dom[i].Gcc.s1 * Dom[i].Gcc.jn;
    Dom[i].Gcc.s3 = Dom[i].Gcc.s2 * Dom[i].Gcc.kn;
    Dom[i].Gcc.s1b = Dom[i].Gcc.inb;
    Dom[i].Gcc.s2b = Dom[i].Gcc.s1b * Dom[i].Gcc.jnb;
    Dom[i].Gcc.s3b = Dom[i].Gcc.s2b * Dom[i].Gcc.knb;

    // Gfx
    if(Dom[i].W > -1)
      Dom[i].Gfx.is = Dom[Dom[i].W].Gfx.ie;
    else
      Dom[i].Gfx.is = DOM_BUF;
    Dom[i].Gfx.isb = Dom[i].Gfx.is - DOM_BUF;
    Dom[i].Gfx.in = Dom[i].xn + 1;
    Dom[i].Gfx.inb = Dom[i].Gfx.in + 2 * DOM_BUF;
    Dom[i].Gfx.ie = Dom[i].Gfx.isb + Dom[i].Gfx.in;
    Dom[i].Gfx.ieb = Dom[i].Gfx.ie + DOM_BUF;

    if(Dom[i].S > -1)
      Dom[i].Gfx.js = Dom[Dom[i].S].Gfx.je + 1;
    else
      Dom[i].Gfx.js = DOM_BUF;
    Dom[i].Gfx.jsb = Dom[i].Gfx.js - DOM_BUF;
    Dom[i].Gfx.jn = Dom[i].yn;
    Dom[i].Gfx.jnb = Dom[i].Gfx.jn + 2 * DOM_BUF;
    Dom[i].Gfx.je = Dom[i].Gfx.jsb + Dom[i].Gfx.jn;
    Dom[i].Gfx.jeb = Dom[i].Gfx.je + DOM_BUF;

    if(Dom[i].B > -1)
      Dom[i].Gfx.ks = Dom[Dom[i].B].Gfx.ke + 1;
    else
      Dom[i].Gfx.ks = DOM_BUF;
    Dom[i].Gfx.ksb = Dom[i].Gfx.ks - DOM_BUF;
    Dom[i].Gfx.kn = Dom[i].zn;
    Dom[i].Gfx.knb = Dom[i].Gfx.kn + 2 * DOM_BUF;
    Dom[i].Gfx.ke = Dom[i].Gfx.ksb + Dom[i].Gfx.kn;
    Dom[i].Gfx.keb = Dom[i].Gfx.ke + DOM_BUF;

    Dom[i].Gfx._is = DOM_BUF;
    Dom[i].Gfx._isb = Dom[i].Gfx._is - DOM_BUF;
    Dom[i].Gfx.in = Dom[i].xn;
    Dom[i].Gfx.inb = Dom[i].Gfx.in + 2 * DOM_BUF;
    Dom[i].Gfx._ie = Dom[i].Gfx._isb + Dom[i].Gfx.in;
    Dom[i].Gfx._ieb = Dom[i].Gfx._ie + DOM_BUF;

    Dom[i].Gfx._js = DOM_BUF;
    Dom[i].Gfx._jsb = Dom[i].Gfx._js - DOM_BUF;
    Dom[i].Gfx.jn = Dom[i].yn;
    Dom[i].Gfx.jnb = Dom[i].Gfx.jn + 2 * DOM_BUF;
    Dom[i].Gfx._je = Dom[i].Gfx._jsb + Dom[i].Gfx.jn;
    Dom[i].Gfx._jeb = Dom[i].Gfx._je + DOM_BUF;

    Dom[i].Gfx._ks = DOM_BUF;
    Dom[i].Gfx._ksb = Dom[i].Gfx._ks - DOM_BUF;
    Dom[i].Gfx.kn = Dom[i].zn;
    Dom[i].Gfx.knb = Dom[i].Gfx.kn + 2 * DOM_BUF;
    Dom[i].Gfx._ke = Dom[i].Gfx._ksb + Dom[i].Gfx.kn;
    Dom[i].Gfx._keb = Dom[i].Gfx._ke + DOM_BUF;

    Dom[i].Gfx.s1 = Dom[i].Gfx.in;
    Dom[i].Gfx.s2 = Dom[i].Gfx.s1 * Dom[i].Gfx.jn;
    Dom[i].Gfx.s3 = Dom[i].Gfx.s2 * Dom[i].Gfx.kn;
    Dom[i].Gfx.s1b = Dom[i].Gfx.inb;
    Dom[i].Gfx.s2b = Dom[i].Gfx.s1b * Dom[i].Gfx.jnb;
    Dom[i].Gfx.s3b = Dom[i].Gfx.s2b * Dom[i].Gfx.knb;

    // Gfy
    if(Dom[i].W > -1)
      Dom[i].Gfy.is = Dom[Dom[i].W].Gfy.ie + 1;
    else
      Dom[i].Gfy.is = DOM_BUF;
    Dom[i].Gfy.isb = Dom[i].Gfy.is - DOM_BUF;
    Dom[i].Gfy.in = Dom[i].xn;
    Dom[i].Gfy.inb = Dom[i].Gfy.in + 2 * DOM_BUF;
    Dom[i].Gfy.ie = Dom[i].Gfy.isb + Dom[i].Gfy.in;
    Dom[i].Gfy.ieb = Dom[i].Gfy.ie + DOM_BUF;

    if(Dom[i].S > -1)
      Dom[i].Gfy.js = Dom[Dom[i].S].Gfy.je;
    else
      Dom[i].Gfy.js = DOM_BUF;
    Dom[i].Gfy.jsb = Dom[i].Gfy.js - DOM_BUF;
    Dom[i].Gfy.jn = Dom[i].yn + 1;
    Dom[i].Gfy.jnb = Dom[i].Gfy.jn + 2 * DOM_BUF;
    Dom[i].Gfy.je = Dom[i].Gfy.jsb + Dom[i].Gfy.jn;
    Dom[i].Gfy.jeb = Dom[i].Gfy.je + DOM_BUF;

    if(Dom[i].B > -1)
      Dom[i].Gfy.ks = Dom[Dom[i].B].Gfy.ke + 1;
    else
      Dom[i].Gfy.ks = DOM_BUF;
    Dom[i].Gfy.ksb = Dom[i].Gfy.ks - DOM_BUF;
    Dom[i].Gfy.kn = Dom[i].zn;
    Dom[i].Gfy.knb = Dom[i].Gfy.kn + 2 * DOM_BUF;
    Dom[i].Gfy.ke = Dom[i].Gfy.ksb + Dom[i].Gfy.kn;
    Dom[i].Gfy.keb = Dom[i].Gfy.ke + DOM_BUF;

    Dom[i].Gfy._is = DOM_BUF;
    Dom[i].Gfy._isb = Dom[i].Gfy._is - DOM_BUF;
    Dom[i].Gfy.in = Dom[i].xn;
    Dom[i].Gfy.inb = Dom[i].Gfy.in + 2 * DOM_BUF;
    Dom[i].Gfy._ie = Dom[i].Gfy._isb + Dom[i].Gfy.in;
    Dom[i].Gfy._ieb = Dom[i].Gfy._ie + DOM_BUF;

    Dom[i].Gfy._js = DOM_BUF;
    Dom[i].Gfy._jsb = Dom[i].Gfy._js - DOM_BUF;
    Dom[i].Gfy.jn = Dom[i].yn;
    Dom[i].Gfy.jnb = Dom[i].Gfy.jn + 2 * DOM_BUF;
    Dom[i].Gfy._je = Dom[i].Gfy._jsb + Dom[i].Gfy.jn;
    Dom[i].Gfy._jeb = Dom[i].Gfy._je + DOM_BUF;

    Dom[i].Gfy._ks = DOM_BUF;
    Dom[i].Gfy._ksb = Dom[i].Gfy._ks - DOM_BUF;
    Dom[i].Gfy.kn = Dom[i].zn;
    Dom[i].Gfy.knb = Dom[i].Gfy.kn + 2 * DOM_BUF;
    Dom[i].Gfy._ke = Dom[i].Gfy._ksb + Dom[i].Gfy.kn;
    Dom[i].Gfy._keb = Dom[i].Gfy._ke + DOM_BUF;

    Dom[i].Gfy.s1 = Dom[i].Gfy.in;
    Dom[i].Gfy.s2 = Dom[i].Gfy.s1 * Dom[i].Gfy.jn;
    Dom[i].Gfy.s3 = Dom[i].Gfy.s2 * Dom[i].Gfy.kn;
    Dom[i].Gfy.s1b = Dom[i].Gfy.inb;
    Dom[i].Gfy.s2b = Dom[i].Gfy.s1b * Dom[i].Gfy.jnb;
    Dom[i].Gfy.s3b = Dom[i].Gfy.s2b * Dom[i].Gfy.knb;

    // Gfz
    if(Dom[i].W > -1)
      Dom[i].Gfz.is = Dom[Dom[i].W].Gfz.ie + 1;
    else
      Dom[i].Gfz.is = DOM_BUF;
    Dom[i].Gfz.isb = Dom[i].Gfz.is - DOM_BUF;
    Dom[i].Gfz.in = Dom[i].xn;
    Dom[i].Gfz.inb = Dom[i].Gfz.in + 2 * DOM_BUF;
    Dom[i].Gfz.ie = Dom[i].Gfz.isb + Dom[i].Gfz.in;
    Dom[i].Gfz.ieb = Dom[i].Gfz.ie + DOM_BUF;

    if(Dom[i].S > -1)
      Dom[i].Gfz.js = Dom[Dom[i].S].Gfz.je + 1;
    else
      Dom[i].Gfz.js = DOM_BUF;
    Dom[i].Gfz.jsb = Dom[i].Gfz.js - DOM_BUF;
    Dom[i].Gfz.jn = Dom[i].yn;
    Dom[i].Gfz.jnb = Dom[i].Gfz.jn + 2 * DOM_BUF;
    Dom[i].Gfz.je = Dom[i].Gfz.jsb + Dom[i].Gfz.jn;
    Dom[i].Gfz.jeb = Dom[i].Gfz.je + DOM_BUF;

    if(Dom[i].B > -1)
      Dom[i].Gfz.ks = Dom[Dom[i].B].Gfz.ke;
    else
      Dom[i].Gfz.ks = DOM_BUF;
    Dom[i].Gfz.ksb = Dom[i].Gfz.ks - DOM_BUF;
    Dom[i].Gfz.kn = Dom[i].zn + 1;
    Dom[i].Gfz.knb = Dom[i].Gfz.kn + 2 * DOM_BUF;
    Dom[i].Gfz.ke = Dom[i].Gfz.ksb + Dom[i].Gfz.kn;
    Dom[i].Gfz.keb = Dom[i].Gfz.ke + DOM_BUF;

    Dom[i].Gfz._is = DOM_BUF;
    Dom[i].Gfz._isb = Dom[i].Gfz._is - DOM_BUF;
    Dom[i].Gfz.in = Dom[i].xn;
    Dom[i].Gfz.inb = Dom[i].Gfz.in + 2 * DOM_BUF;
    Dom[i].Gfz._ie = Dom[i].Gfz._isb + Dom[i].Gfz.in;
    Dom[i].Gfz._ieb = Dom[i].Gfz._ie + DOM_BUF;

    Dom[i].Gfz._js = DOM_BUF;
    Dom[i].Gfz._jsb = Dom[i].Gfz._js - DOM_BUF;
    Dom[i].Gfz.jn = Dom[i].yn;
    Dom[i].Gfz.jnb = Dom[i].Gfz.jn + 2 * DOM_BUF;
    Dom[i].Gfz._je = Dom[i].Gfz._jsb + Dom[i].Gfz.jn;
    Dom[i].Gfz._jeb = Dom[i].Gfz._je + DOM_BUF;

    Dom[i].Gfz._ks = DOM_BUF;
    Dom[i].Gfz._ksb = Dom[i].Gfz._ks - DOM_BUF;
    Dom[i].Gfz.kn = Dom[i].zn;
    Dom[i].Gfz.knb = Dom[i].Gfz.kn + 2 * DOM_BUF;
    Dom[i].Gfz._ke = Dom[i].Gfz._ksb + Dom[i].Gfz.kn;
    Dom[i].Gfz._keb = Dom[i].Gfz._ke + DOM_BUF;

    Dom[i].Gfz.s1 = Dom[i].Gfz.in;
    Dom[i].Gfz.s2 = Dom[i].Gfz.s1 * Dom[i].Gfz.jn;
    Dom[i].Gfz.s3 = Dom[i].Gfz.s2 * Dom[i].Gfz.kn;
    Dom[i].Gfz.s1b = Dom[i].Gfz.inb;
    Dom[i].Gfz.s2b = Dom[i].Gfz.s1b * Dom[i].Gfz.jnb;
    Dom[i].Gfz.s3b = Dom[i].Gfz.s2b * Dom[i].Gfz.knb;
  }

  // configure GPU subdomains
  /* These subdomains must be provided in increasing order */
  for(i = 0; i < nGPUdom; i++) {
    dom[i].xl = dom[i].xe - dom[i].xs;
    dom[i].yl = dom[i].ye - dom[i].ys;
    dom[i].zl = dom[i].ze - dom[i].zs;
    dom[i].dx = dom[i].xl / dom[i].xn;
    dom[i].dy = dom[i].yl / dom[i].yn;
    dom[i].dz = dom[i].zl / dom[i].zn;

    // Gcc
    if(dom[i].w > -1)
      dom[i].Gcc.is = dom[dom[i].w].Gcc.ie + 1;
    else
      dom[i].Gcc.is = DOM_BUF;
    dom[i].Gcc.isb = dom[i].Gcc.is - DOM_BUF;
    dom[i].Gcc.in = dom[i].xn;
    dom[i].Gcc.inb = dom[i].Gcc.in + 2 * DOM_BUF;
    dom[i].Gcc.ie = dom[i].Gcc.isb + dom[i].Gcc.in;
    dom[i].Gcc.ieb = dom[i].Gcc.ie + DOM_BUF;

    if(dom[i].s > -1)
      dom[i].Gcc.js = dom[dom[i].s].Gcc.je + 1;
    else
      dom[i].Gcc.js = DOM_BUF;
    dom[i].Gcc.jsb = dom[i].Gcc.js - DOM_BUF;
    dom[i].Gcc.jn = dom[i].yn;
    dom[i].Gcc.jnb = dom[i].Gcc.jn + 2 * DOM_BUF;
    dom[i].Gcc.je = dom[i].Gcc.jsb + dom[i].Gcc.jn;
    dom[i].Gcc.jeb = dom[i].Gcc.je + DOM_BUF;

    if(dom[i].b > -1)
      dom[i].Gcc.ks = dom[dom[i].b].Gcc.ke + 1;
    else
      dom[i].Gcc.ks = DOM_BUF;
    dom[i].Gcc.ksb = dom[i].Gcc.ks - DOM_BUF;
    dom[i].Gcc.kn = dom[i].zn;
    dom[i].Gcc.knb = dom[i].Gcc.kn + 2 * DOM_BUF;
    dom[i].Gcc.ke = dom[i].Gcc.ksb + dom[i].Gcc.jn;
    dom[i].Gcc.keb = dom[i].Gcc.ke + DOM_BUF;

    dom[i].Gcc._is = DOM_BUF;
    dom[i].Gcc._isb = dom[i].Gcc._is - DOM_BUF;
    dom[i].Gcc.in = dom[i].xn;
    dom[i].Gcc.inb = dom[i].Gcc.in + 2 * DOM_BUF;
    dom[i].Gcc._ie = dom[i].Gcc._isb + dom[i].Gcc.in;
    dom[i].Gcc._ieb = dom[i].Gcc._ie + DOM_BUF;

    dom[i].Gcc._js = DOM_BUF;
    dom[i].Gcc._jsb = dom[i].Gcc._js - DOM_BUF;
    dom[i].Gcc.jn = dom[i].yn;
    dom[i].Gcc.jnb = dom[i].Gcc.jn + 2 * DOM_BUF;
    dom[i].Gcc._je = dom[i].Gcc._jsb + dom[i].Gcc.jn;
    dom[i].Gcc._jeb = dom[i].Gcc._je + DOM_BUF;

    dom[i].Gcc._ks = DOM_BUF;
    dom[i].Gcc._ksb = dom[i].Gcc._ks - DOM_BUF;
    dom[i].Gcc.kn = dom[i].zn;
    dom[i].Gcc.knb = dom[i].Gcc.kn + 2 * DOM_BUF;
    dom[i].Gcc._ke = dom[i].Gcc._ksb + dom[i].Gcc.kn;
    dom[i].Gcc._keb = dom[i].Gcc._ke + DOM_BUF;

    dom[i].Gcc.s1 = dom[i].Gcc.in;
    dom[i].Gcc.s2 = dom[i].Gcc.s1 * dom[i].Gcc.jn;
    dom[i].Gcc.s3 = dom[i].Gcc.s2 * dom[i].Gcc.kn;
    dom[i].Gcc.s1b = dom[i].Gcc.inb;
    dom[i].Gcc.s2b = dom[i].Gcc.s1b * dom[i].Gcc.jnb;
    dom[i].Gcc.s3b = dom[i].Gcc.s2b * dom[i].Gcc.knb;

    // Gfx
    if(dom[i].w > -1)
      dom[i].Gfx.is = dom[dom[i].w].Gfx.ie;
    else
      dom[i].Gfx.is = DOM_BUF;
    dom[i].Gfx.isb = dom[i].Gfx.is - DOM_BUF;
    dom[i].Gfx.in = dom[i].xn + 1;
    dom[i].Gfx.inb = dom[i].Gfx.in + 2 * DOM_BUF;
    dom[i].Gfx.ie = dom[i].Gfx.isb + dom[i].Gfx.in;
    dom[i].Gfx.ieb = dom[i].Gfx.ie + DOM_BUF;

    if(dom[i].s > -1)
      dom[i].Gfx.js = dom[dom[i].s].Gfx.je + 1;
    else
      dom[i].Gfx.js = DOM_BUF;
    dom[i].Gfx.jsb = dom[i].Gfx.js - DOM_BUF;
    dom[i].Gfx.jn = dom[i].yn;
    dom[i].Gfx.jnb = dom[i].Gfx.jn + 2 * DOM_BUF;
    dom[i].Gfx.je = dom[i].Gfx.jsb + dom[i].Gfx.jn;
    dom[i].Gfx.jeb = dom[i].Gfx.je + DOM_BUF;

    if(dom[i].b > -1)
      dom[i].Gfx.ks = dom[dom[i].b].Gfx.ke + 1;
    else
      dom[i].Gfx.ks = DOM_BUF;
    dom[i].Gfx.ksb = dom[i].Gfx.ks - DOM_BUF;
    dom[i].Gfx.kn = dom[i].zn;
    dom[i].Gfx.knb = dom[i].Gfx.kn + 2 * DOM_BUF;
    dom[i].Gfx.ke = dom[i].Gfx.ksb + dom[i].Gfx.kn;
    dom[i].Gfx.keb = dom[i].Gfx.ke + DOM_BUF;

    dom[i].Gfx._is = DOM_BUF;
    dom[i].Gfx._isb = dom[i].Gfx._is - DOM_BUF;
    dom[i].Gfx.in = dom[i].xn;
    dom[i].Gfx.inb = dom[i].Gfx.in + 2 * DOM_BUF;
    dom[i].Gfx._ie = dom[i].Gfx._isb + dom[i].Gfx.in;
    dom[i].Gfx._ieb = dom[i].Gfx._ie + DOM_BUF;

    dom[i].Gfx._js = DOM_BUF;
    dom[i].Gfx._jsb = dom[i].Gfx._js - DOM_BUF;
    dom[i].Gfx.jn = dom[i].yn;
    dom[i].Gfx.jnb = dom[i].Gfx.jn + 2 * DOM_BUF;
    dom[i].Gfx._je = dom[i].Gfx._jsb + dom[i].Gfx.jn;
    dom[i].Gfx._jeb = dom[i].Gfx._je + DOM_BUF;

    dom[i].Gfx._ks = DOM_BUF;
    dom[i].Gfx._ksb = dom[i].Gfx._ks - DOM_BUF;
    dom[i].Gfx.kn = dom[i].zn;
    dom[i].Gfx.knb = dom[i].Gfx.kn + 2 * DOM_BUF;
    dom[i].Gfx._ke = dom[i].Gfx._ksb + dom[i].Gfx.kn;
    dom[i].Gfx._keb = dom[i].Gfx._ke + DOM_BUF;

    dom[i].Gfx.s1 = dom[i].Gfx.in;
    dom[i].Gfx.s2 = dom[i].Gfx.s1 * dom[i].Gfx.jn;
    dom[i].Gfx.s3 = dom[i].Gfx.s2 * dom[i].Gfx.kn;
    dom[i].Gfx.s1b = dom[i].Gfx.inb;
    dom[i].Gfx.s2b = dom[i].Gfx.s1b * dom[i].Gfx.jnb;
    dom[i].Gfx.s3b = dom[i].Gfx.s2b * dom[i].Gfx.knb;

    // Gfy
    if(dom[i].w > -1)
      dom[i].Gfy.is = dom[dom[i].w].Gfy.ie + 1;
    else
      dom[i].Gfy.is = DOM_BUF;
    dom[i].Gfy.isb = dom[i].Gfy.is - DOM_BUF;
    dom[i].Gfy.in = dom[i].xn;
    dom[i].Gfy.inb = dom[i].Gfy.in + 2 * DOM_BUF;
    dom[i].Gfy.ie = dom[i].Gfy.isb + dom[i].Gfy.in;
    dom[i].Gfy.ieb = dom[i].Gfy.ie + DOM_BUF;

    if(dom[i].s > -1)
      dom[i].Gfy.js = dom[dom[i].s].Gfy.je;
    else
      dom[i].Gfy.js = DOM_BUF;
    dom[i].Gfy.jsb = dom[i].Gfy.js - DOM_BUF;
    dom[i].Gfy.jn = dom[i].yn + 1;
    dom[i].Gfy.jnb = dom[i].Gfy.jn + 2 * DOM_BUF;
    dom[i].Gfy.je = dom[i].Gfy.jsb + dom[i].Gfy.jn;
    dom[i].Gfy.jeb = dom[i].Gfy.je + DOM_BUF;

    if(dom[i].b > -1)
      dom[i].Gfy.ks = dom[dom[i].b].Gfy.ke + 1;
    else
      dom[i].Gfy.ks = DOM_BUF;
    dom[i].Gfy.ksb = dom[i].Gfy.ks - DOM_BUF;
    dom[i].Gfy.kn = dom[i].zn;
    dom[i].Gfy.knb = dom[i].Gfy.kn + 2 * DOM_BUF;
    dom[i].Gfy.ke = dom[i].Gfy.ksb + dom[i].Gfy.kn;
    dom[i].Gfy.keb = dom[i].Gfy.ke + DOM_BUF;

    dom[i].Gfy._is = DOM_BUF;
    dom[i].Gfy._isb = dom[i].Gfy._is - DOM_BUF;
    dom[i].Gfy.in = dom[i].xn;
    dom[i].Gfy.inb = dom[i].Gfy.in + 2 * DOM_BUF;
    dom[i].Gfy._ie = dom[i].Gfy._isb + dom[i].Gfy.in;
    dom[i].Gfy._ieb = dom[i].Gfy._ie + DOM_BUF;

    dom[i].Gfy._js = DOM_BUF;
    dom[i].Gfy._jsb = dom[i].Gfy._js - DOM_BUF;
    dom[i].Gfy.jn = dom[i].yn;
    dom[i].Gfy.jnb = dom[i].Gfy.jn + 2 * DOM_BUF;
    dom[i].Gfy._je = dom[i].Gfy._jsb + dom[i].Gfy.jn;
    dom[i].Gfy._jeb = dom[i].Gfy._je + DOM_BUF;

    dom[i].Gfy._ks = DOM_BUF;
    dom[i].Gfy._ksb = dom[i].Gfy._ks - DOM_BUF;
    dom[i].Gfy.kn = dom[i].zn;
    dom[i].Gfy.knb = dom[i].Gfy.kn + 2 * DOM_BUF;
    dom[i].Gfy._ke = dom[i].Gfy._ksb + dom[i].Gfy.kn;
    dom[i].Gfy._keb = dom[i].Gfy._ke + DOM_BUF;

    dom[i].Gfy.s1 = dom[i].Gfy.in;
    dom[i].Gfy.s2 = dom[i].Gfy.s1 * dom[i].Gfy.jn;
    dom[i].Gfy.s3 = dom[i].Gfy.s2 * dom[i].Gfy.kn;
    dom[i].Gfy.s1b = dom[i].Gfy.inb;
    dom[i].Gfy.s2b = dom[i].Gfy.s1b * dom[i].Gfy.jnb;
    dom[i].Gfy.s3b = dom[i].Gfy.s2b * dom[i].Gfy.knb;

    // Gfz
    if(dom[i].w > -1)
      dom[i].Gfz.is = dom[dom[i].w].Gfz.ie + 1;
    else
      dom[i].Gfz.is = DOM_BUF;
    dom[i].Gfz.isb = dom[i].Gfz.is - DOM_BUF;
    dom[i].Gfz.in = dom[i].xn;
    dom[i].Gfz.inb = dom[i].Gfz.in + 2 * DOM_BUF;
    dom[i].Gfz.ie = dom[i].Gfz.isb + dom[i].Gfz.in;
    dom[i].Gfz.ieb = dom[i].Gfz.ie + DOM_BUF;

    if(dom[i].s > -1)
      dom[i].Gfz.js = dom[dom[i].s].Gfz.je + 1;
    else
      dom[i].Gfz.js = DOM_BUF;
    dom[i].Gfz.jsb = dom[i].Gfz.js - DOM_BUF;
    dom[i].Gfz.jn = dom[i].yn;
    dom[i].Gfz.jnb = dom[i].Gfz.jn + 2 * DOM_BUF;
    dom[i].Gfz.je = dom[i].Gfz.jsb + dom[i].Gfz.jn;
    dom[i].Gfz.jeb = dom[i].Gfz.je + DOM_BUF;

    if(dom[i].b > -1)
      dom[i].Gfz.ks = dom[dom[i].b].Gfz.ke;
    else
      dom[i].Gfz.ks = DOM_BUF;
    dom[i].Gfz.ksb = dom[i].Gfz.ks - DOM_BUF;
    dom[i].Gfz.kn = dom[i].zn + 1;
    dom[i].Gfz.knb = dom[i].Gfz.kn + 2 * DOM_BUF;
    dom[i].Gfz.ke = dom[i].Gfz.ksb + dom[i].Gfz.kn;
    dom[i].Gfz.keb = dom[i].Gfz.ke + DOM_BUF;

    dom[i].Gfz._is = DOM_BUF;
    dom[i].Gfz._isb = dom[i].Gfz._is - DOM_BUF;
    dom[i].Gfz.in = dom[i].xn;
    dom[i].Gfz.inb = dom[i].Gfz.in + 2 * DOM_BUF;
    dom[i].Gfz._ie = dom[i].Gfz._isb + dom[i].Gfz.in;
    dom[i].Gfz._ieb = dom[i].Gfz._ie + DOM_BUF;

    dom[i].Gfz._js = DOM_BUF;
    dom[i].Gfz._jsb = dom[i].Gfz._js - DOM_BUF;
    dom[i].Gfz.jn = dom[i].yn;
    dom[i].Gfz.jnb = dom[i].Gfz.jn + 2 * DOM_BUF;
    dom[i].Gfz._je = dom[i].Gfz._jsb + dom[i].Gfz.jn;
    dom[i].Gfz._jeb = dom[i].Gfz._je + DOM_BUF;

    dom[i].Gfz._ks = DOM_BUF;
    dom[i].Gfz._ksb = dom[i].Gfz._ks - DOM_BUF;
    dom[i].Gfz.kn = dom[i].zn;
    dom[i].Gfz.knb = dom[i].Gfz.kn + 2 * DOM_BUF;
    dom[i].Gfz._ke = dom[i].Gfz._ksb + dom[i].Gfz.kn;
    dom[i].Gfz._keb = dom[i].Gfz._ke + DOM_BUF;

    dom[i].Gfz.s1 = dom[i].Gfz.in;
    dom[i].Gfz.s2 = dom[i].Gfz.s1 * dom[i].Gfz.jn;
    dom[i].Gfz.s3 = dom[i].Gfz.s2 * dom[i].Gfz.kn;
    dom[i].Gfz.s1b = dom[i].Gfz.inb;
    dom[i].Gfz.s2b = dom[i].Gfz.s1b * dom[i].Gfz.jnb;
    dom[i].Gfz.s3b = dom[i].Gfz.s2b * dom[i].Gfz.knb;
  }
}

void domain_map_clean(void)
{
  free(dom);
  free(Dom);
}
