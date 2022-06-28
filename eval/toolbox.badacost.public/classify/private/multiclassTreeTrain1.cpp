/*******************************************************************************
* Piotr's Image&Video Toolbox      Version 3.24
* Copyright 2013 Piotr Dollar.  [pdollar-at-caltech.edu]
* Please email me if you find bugs, or have suggestions or questions!
* Licensed under the Simplified BSD License [see external/bsd.txt]
*******************************************************************************/
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <mex.h>
#include <iostream>
#include <vector>

#ifdef USEOMP
#include <omp.h>
#endif

typedef unsigned int uint32;
#define gini(p) p*p

// [split_val,thrs,gains] = mexFunction(data,hs,ws,order,H,split);
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  int H, N, F, split; 
  float *data, *ws, thr;
  double gain; uint32 *hs, *order;
  double cfactor, *costs_factor;
  data = (float*) mxGetData(prhs[0]);
  hs = (uint32*) mxGetData(prhs[1]);
  ws = (float*) mxGetData(prhs[2]);
  order = (uint32*) mxGetData(prhs[3]);
  H = (int) mxGetScalar(prhs[4]);
  split = (int) mxGetScalar(prhs[5]);
  costs_factor = (double*) mxGetData(prhs[6]); // Correction for class probabilities with costs
  N = (int) mxGetM(prhs[0]);
  F = (int) mxGetN(prhs[0]);
  
  // create output structures
  plhs[0] = mxCreateNumericMatrix(1,F,mxDOUBLE_CLASS,mxREAL);
  plhs[1] = mxCreateNumericMatrix(1,F,mxDOUBLE_CLASS,mxREAL);
  plhs[2] = mxCreateNumericMatrix(1,F,mxDOUBLE_CLASS,mxREAL);
  double *split_val = (double*) mxGetData(plhs[0]);
  double *thrs = (double*) mxGetData(plhs[1]);
  double *gains = (double*) mxGetData(plhs[2]);
  
  double vInit, w, g;
  std::vector<double> W(H);
  
  // perform initialization
  vInit = 0; g = 0; w = 0; 
  for(int i=0; i<H; i++ ) W[i] = 0;
  for(int j=0; j<N; j++ ) { w+=ws[j]; W[hs[j]-1]+=ws[j]; }
  for(int i=0; i<H; i++ ) g+=gini(W[i]*costs_factor[i]); vInit=(1.-(g)/w/w); 
  
  // loop over features, then thresholds (data is sorted by feature value)
  // FIXME!: Take a look. Paralelise the search for the best of the F features.
  #ifdef USEOMP
  int nThreads = 4;
  nThreads = std::min(nThreads,omp_get_max_threads());
  #pragma omp parallel for num_threads(nThreads)
  #endif
  for(int i=0; i<F; i++ ) {
    std::vector<double> Wl(H), Wr(H);
    double wl, wr, gl, gr;
    float *data1 = (float*) data+i*size_t(N);
    uint32 *order1 = (uint32*) order+i*size_t(N); 
    for(int j=0; j<H; j++ ) { Wl[j]=0; Wr[j]=W[j]; } gl=wl=0; gr=g; wr=w;          
    
    double best = vInit;
    for(int j=0; j<N-1; j++) {
      double v;
      int j1, j2, h; 
      j1=order1[j]; j2=order1[j+1]; h=hs[j1]-1;      
     
      // gini = 1-\sum_h p_h^2; v = gini_l*pl + gini_r*pr
      gl-=gini(Wl[h]*costs_factor[h]); // remove datum j1-th datum previous gini vaule
      gr-=gini(Wr[h]*costs_factor[h]); // remove datum j1-th datum previous gini vaule
      // We move the threshold one datum to the right (we are now between) 
      // datum j1 and datum j2 in the ordered by order1. Therefore j1-th 
      // datum is now on the left of the threshold.
      wl+=ws[j1]; // increase sum weights of j1-th datum from the left sum of weights
      Wl[h]+=ws[j1]; // increase sum weights of j1-th to the h-th class sum at the left 
      gl+=gini(Wl[h]*costs_factor[h]); // update contribution of gini on the left
      
      wr-=ws[j1]; // remove weight of j1-th datum from the right node sum of weights
      Wr[h]-=ws[j1]; // remove weight of j1-th datum from the h-th class sum at the right
      gr+=gini(Wr[h]*costs_factor[h]);

      v = (wl-((gl/wl)))/w + (wr-(gr/wr))/w;

      if( v<best ) {
         best = v;
         split_val[i] = best;
         gains[i] = vInit - best;
         thrs[i] = 0.5f*(data1[j1]+data1[j2]);       
      }     
    }
  }
} 
