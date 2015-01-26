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

#include "cuda_comm.h"

#include <cuda.h>
#include <helper_cuda.h>

extern "C"
void cuda_dom_malloc(void)
{
  // allocate device memory on host
  _dom = (dom_struct**) malloc(nodes[rank].ndevs * sizeof(dom_struct*));
  _p0 = (real**) malloc(nodes[rank].ndevs * sizeof(real*));
  _p = (real**) malloc(nodes[rank].ndevs * sizeof(real*));
  //_rhs_p = (real**) malloc(nodes[rank].ndevs * sizeof(real*));
  _flag_u = (int**) malloc(nodes[rank].ndevs * sizeof(int*));
  _flag_v = (int**) malloc(nodes[rank].ndevs * sizeof(int*));
  _flag_w = (int**) malloc(nodes[rank].ndevs * sizeof(int*));

  // allocate device memory on device
  #pragma omp parallel num_threads(nodes[rank].ndevs)
  {
    int drank = omp_get_thread_num();
    int dev = nodes[rank].devstart + drank;
    checkCudaErrors(cudaSetDevice(nodes[rank].devs[dev]));

    checkCudaErrors(cudaMalloc((void**) &(_dom[drank]),
      sizeof(dom_struct)));

    // copy domain map info
    checkCudaErrors(cudaMemcpy(_dom[drank], &dom[dev], sizeof(dom_struct),
      cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**) &(_p0[drank]),
      sizeof(real) * dom[dev].Gcc.s3b));

    checkCudaErrors(cudaMalloc((void**) &(_p[drank]),
      sizeof(real) * dom[dev].Gcc.s3b));

    //checkCudaErrors(cudaMalloc((void**) &(_rhs_p[drank]),
     // sizeof(real) * (dom[Dev].Gcc.s3)));

    checkCudaErrors(cudaMalloc((void**) &(_flag_u[drank]),
      sizeof(int) * (dom[dev].Gfx.s3b)));
    checkCudaErrors(cudaMalloc((void**) &(_flag_v[drank]),
      sizeof(int) * (dom[dev].Gfy.s3b)));
    checkCudaErrors(cudaMalloc((void**) &(_flag_w[drank]),
      sizeof(int) * (dom[dev].Gfz.s3b)));
  }
}

extern "C"
void copy_node_to_devs(void)
{
  #pragma omp parallel num_threads(nodes[rank].ndevs)
  {
    int drank = omp_get_thread_num();
    int dev = nodes[rank].devstart + drank;
    checkCudaErrors(cudaSetDevice(nodes[rank].devs[dev]));

    int i, I, j, J, k, K;    // iterators

    int c, C;

    // temorary device-level copy arrays
    real *_p_ = (real*) malloc(dom[dev].Gcc.s3b * sizeof(real));
    real *_p0_ = (real*) malloc(dom[dev].Gcc.s3b * sizeof(real));
    int *_flag_u_ = (int*) malloc(dom[dev].Gfx.s3b * sizeof(real));
    int *_flag_v_ = (int*) malloc(dom[dev].Gfy.s3b * sizeof(real));
    int *_flag_w_ = (int*) malloc(dom[dev].Gfz.s3b * sizeof(real));

    // p
    for(k = dom[dev].Gcc._ks; k <= dom[dev].Gcc._ke; k++) {
      for(j = dom[dev].Gcc._js; j <= dom[dev].Gcc._je; j++) {
        for(i = dom[dev].Gcc._is; i <= dom[dev].Gcc._ie; i++) {
          I = i + dom[dev].Gcc.isb;
          J = j + dom[dev].Gcc.jsb;
          K = k + dom[dev].Gcc.ksb;
          C = I + J*Dom[rank].Gcc.s1b + K*Dom[rank].Gcc.s2b;
          c = i + j*dom[dev].Gcc.s1b + k*dom[dev].Gcc.s2b;
          _p_[c] = p[C];
          _p0_[c] = p0[C];
        }
      }
    }

    // u
    for(k = dom[dev].Gfx._ks; k <= dom[dev].Gfx._ke; k++) {
      for(j = dom[dev].Gfx._js; j <= dom[dev].Gfx._je; j++) {
        for(i = dom[dev].Gfx._is; i <= dom[dev].Gfx._ie; i++) {
          I = i + dom[dev].Gfx.isb;
          J = j + dom[dev].Gfx.jsb;
          K = k + dom[dev].Gfx.ksb;
          C = I + J*Dom[rank].Gfx.s1b + K*Dom[rank].Gfx.s2b;
          c = i + j*dom[dev].Gfx.s1b + k*dom[dev].Gfx.s2b;
          _flag_u_[c] = flag_u[C];
        }
      }
    }

    // v
    for(k = dom[dev].Gfy._ks; k <= dom[dev].Gfy._ke; k++) {
      for(j = dom[dev].Gfy._js; j <= dom[dev].Gfy._je; j++) {
        for(i = dom[dev].Gfy._is; i <= dom[dev].Gfy._ie; i++) {
          I = i + dom[dev].Gfy.isb;
          J = j + dom[dev].Gfy.jsb;
          K = k + dom[dev].Gfy.ksb;
          C = I + J*Dom[rank].Gfy.s1b + K*Dom[rank].Gfy.s2b;
          c = i + j*dom[dev].Gfy.s1b + k*dom[dev].Gfy.s2b;
          _flag_v_[c] = flag_v[C];
        }
      }
    }

    // w
    for(k = dom[dev].Gfz._ks; k <= dom[dev].Gfz._ke; k++) {
      for(j = dom[dev].Gfz._js; j <= dom[dev].Gfz._je; j++) {
        for(i = dom[dev].Gfz._is; i <= dom[dev].Gfz._ie; i++) {
          I = i + dom[dev].Gfz.isb;
          J = j + dom[dev].Gfz.jsb;
          K = k + dom[dev].Gfz.ksb;
          C = I + J*Dom[rank].Gfz.s1b + K*Dom[rank].Gfz.s2b;
          c = i + j*dom[dev].Gfz.s1b + k*dom[dev].Gfz.s2b;
          _flag_w_[c] = flag_w[C];
        }
      }
    }

    // copy from host to device
    checkCudaErrors(cudaMemcpy(_p[drank], _p_,
      sizeof(real) * dom[dev].Gcc.s3b, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_p0[drank], _p0_,
      sizeof(real) * dom[dev].Gcc.s3b, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_flag_u[drank], _flag_u_,
      sizeof(int) * dom[dev].Gfx.s3b, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_flag_v[drank], _flag_v_,
      sizeof(int) * dom[dev].Gfy.s3b, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_flag_w[drank], _flag_w_,
      sizeof(int) * dom[dev].Gfz.s3b, cudaMemcpyHostToDevice));

    // clean up
    free(_p_);
    free(_p0_);
    free(_flag_u_);
    free(_flag_v_);
    free(_flag_w_);

    // touch p
    int threads = MAX_THREADS_DIM;
    int blocks_y = (int)ceil((real) dom[dev].Gcc.jnb / (real) threads);
    int blocks_z = (int)ceil((real) dom[dev].Gcc.knb / (real) threads);
    dim3 numBlocks(blocks_y, blocks_z);
    dim3 dimBlocks(threads, threads);
    touchp<<<numBlocks, dimBlocks>>>(_p[drank], _dom[drank], dev);
    touchp<<<numBlocks, dimBlocks>>>(_p0[drank], _dom[drank], rank);
  }
}

extern "C"
void copy_devs_to_node(void)
{
  // initialize device memory on host
  #pragma omp parallel num_threads(nodes[rank].ndevs)
  {
    int drank = omp_get_thread_num();
    int dev = nodes[rank].devstart + drank;
    checkCudaErrors(cudaSetDevice(nodes[rank].devs[drank]));

    int i, I, j, J, k, K;    // iterators

    int c, C;

    // temorary device-level copy arrays
    real *_p_ = (real*) malloc(dom[dev].Gcc.s3b * sizeof(real));
    real *_p0_ = (real*) malloc(dom[dev].Gcc.s3b * sizeof(real));
    int *_flag_u_ = (int*) malloc(dom[dev].Gfx.s3b * sizeof(real));
    int *_flag_v_ = (int*) malloc(dom[dev].Gfy.s3b * sizeof(real));
    int *_flag_w_ = (int*) malloc(dom[dev].Gfz.s3b * sizeof(real));

    // copy from device to host
    checkCudaErrors(cudaMemcpy(_p_, _p[drank],
      sizeof(real) * dom[dev].Gcc.s3b, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(_p0_, _p0[drank],
      sizeof(real) * dom[dev].Gcc.s3b, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(_flag_u_, _flag_u[drank],
      sizeof(int) * dom[dev].Gfx.s3b, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(_flag_v_, _flag_v[drank],
      sizeof(int) * dom[dev].Gfy.s3b, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(_flag_w_, _flag_w[drank],
      sizeof(int) * dom[dev].Gfz.s3b, cudaMemcpyDeviceToHost));

    // p
    for(k = dom[dev].Gcc._ks; k <= dom[dev].Gcc._ke; k++) {
      for(j = dom[dev].Gcc._js; j <= dom[dev].Gcc._je; j++) {
        for(i = dom[dev].Gcc._is; i <= dom[dev].Gcc._ie; i++) {
          I = i + dom[dev].Gcc.isb;
          J = j + dom[dev].Gcc.jsb;
          K = k + dom[dev].Gcc.ksb;
          C = I + J*Dom[rank].Gcc.s1b + K*Dom[rank].Gcc.s2b;
          c = i + j*dom[dev].Gcc.s1b + k*dom[dev].Gcc.s2b;
          p[C] = _p_[c];
          p0[C] = _p0_[c];
        }
      }
    }

    // u
    for(k = dom[dev].Gfx._ks; k <= dom[dev].Gfx._ke; k++) {
      for(j = dom[dev].Gfx._js; j <= dom[dev].Gfx._je; j++) {
        for(i = dom[dev].Gfx._is; i <= dom[dev].Gfx._ie; i++) {
          I = i + dom[dev].Gfx.isb;
          J = j + dom[dev].Gfx.jsb;
          K = k + dom[dev].Gfx.ksb;
          C = I + J*Dom[rank].Gfx.s1b + K*Dom[rank].Gfx.s2b;
          c = i + j*dom[dev].Gfx.s1b + k*dom[dev].Gfx.s2b;
          flag_u[C] = _flag_u_[c];
        }
      }
    }

    // v
    for(k = dom[dev].Gfy._ks; k <= dom[dev].Gfy._ke; k++) {
      for(j = dom[dev].Gfy._js; j <= dom[dev].Gfy._je; j++) {
        for(i = dom[dev].Gfy._is; i <= dom[dev].Gfy._ie; i++) {
          I = i + dom[dev].Gfy.isb;
          J = j + dom[dev].Gfy.jsb;
          K = k + dom[dev].Gfy.ksb;
          C = I + J*Dom[rank].Gfy.s1b + K*Dom[rank].Gfy.s2b;
          c = i + j*dom[dev].Gfy.s1b + k*dom[dev].Gfy.s2b;
          flag_v[C] = _flag_v_[c];
        }
      }
    }

    // w
    for(k = dom[dev].Gfz._ks; k <= dom[dev].Gfz._ke; k++) {
      for(j = dom[dev].Gfz._js; j <= dom[dev].Gfz._je; j++) {
        for(i = dom[dev].Gfz._is; i <= dom[dev].Gfz._ie; i++) {
          I = i + dom[dev].Gfz.isb;
          J = j + dom[dev].Gfz.jsb;
          K = k + dom[dev].Gfz.ksb;
          C = I + J*Dom[rank].Gfz.s1b + K*Dom[rank].Gfz.s2b;
          c = i + j*dom[dev].Gfz.s1b + k*dom[dev].Gfz.s2b;
          flag_w[C] = _flag_w_[c];
        }
      }
    }

    // clean up
    free(_p_);
    free(_p0_);
    free(_flag_u_);
    free(_flag_v_);
    free(_flag_w_);
  }
}

extern "C"
void cuda_dom_free(void)
{
  // free GPU memory pointers on node
  #pragma omp parallel num_threads(nodes[rank].ndevs)
  {
    int drank = omp_get_thread_num();
    checkCudaErrors(cudaSetDevice(nodes[rank].devs[drank]));

    checkCudaErrors(cudaFree(_dom[drank]));
    checkCudaErrors(cudaFree(_p0[drank]));
    checkCudaErrors(cudaFree(_p[drank]));
    checkCudaErrors(cudaFree(_flag_u[drank]));
    checkCudaErrors(cudaFree(_flag_v[drank]));
    checkCudaErrors(cudaFree(_flag_w[drank]));
    //checkCudaErrors(cudaFree(_rhs_p[drank]));
  }

  // free GPU memory pointer on node
  free(_dom);
  free(_p0);
  free(_p);
  free(_flag_u);
  free(_flag_v);
  free(_flag_w);
  //free(_rhs_p);
}
