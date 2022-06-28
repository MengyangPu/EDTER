/*******************************************************************************
* Based on Piotr's Image&Video Toolbox      Version 3.24
* Modified by Jose M. Buenaposada for multiclass with costs.
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

typedef unsigned char uint8;
typedef unsigned int uint32;
#define gini(p) p*p

// construct cdf given data vector and wts
//   - Data is indexed between [0, 255]
//   - Class labels are ints.
//   - cdfs is a vector of H (number of classes) vectors
//     of 256 components each (possible values for each indexed data).
void constructCdfPerClass( uint8* data, uint32 *hs, float *wts, 
                           int nBins, int N, uint32 *dids, 
                           std::vector<std::vector<double> >& cdfs )
{
  for(int i=0; i<N; i++) {      
    cdfs[hs[dids[i]]-1][data[dids[i]]] += wts[dids[i]];
  }
  for (int h=0; h<cdfs.size(); h++)
    for(int i=1; i<nBins; i++) cdfs[h][i] += cdfs[h][i-1];
}

// construct cdf given data vector and wts
//   - Data is indexed between [0, nBins-1]
//   - Class labels are ints.
//   - cdfs is a vector of H (number of classes) vectors
//     of 256 components each (possible values for each indexed data).
void constructCdf( uint8* data, float *wts, 
                   int nBins, int N, uint32 *dids, 
                   std::vector<double>& cdf )
{
  for(int i=0; i<N; i++) cdf[data[dids[i]]] += wts[dids[i]];
  for(int i=1; i<nBins; i++) cdf[i] += cdf[i-1];
}

// [split_val,thrs,gains] = mexFunction(data,hs,wts,order,H,split);
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  int H, N, N1, F1, F, split; 
  uint8 *data; uint32* fids, *dids;
  float *wts, thr;
  double gain; 
  uint32 *hs; //, *order;
  double cfactor, *costs_factor;
  int nBins, nThreads; 
  
  data = (uint8*) mxGetData(prhs[0]);
  hs = (uint32*) mxGetData(prhs[1]);
  wts = (float*) mxGetData(prhs[2]);
  nBins = (int) mxGetScalar(prhs[3]);
  dids = (uint32*) mxGetData(prhs[4]);
  fids = (uint32*) mxGetData(prhs[5]);
  H = (int) mxGetScalar(prhs[6]);
  split = (int) mxGetScalar(prhs[7]);
  // Correction for class probabilities with costs
  costs_factor = (double*) mxGetData(prhs[8]); 
  nThreads = (int) mxGetScalar(prhs[9]);

  N  = (int) mxGetM(prhs[0]); // Num data in the original matrix
  F  = (int) mxGetN(prhs[0]); // Num features in the original matrix
  N1 = (int) mxGetNumberOfElements(prhs[4]); // #Ids of used data
  F1 = (int) mxGetNumberOfElements(prhs[5]); // #Ids of used features

  /*
  std::cout << "N =" << N << std::endl;
  std::cout << "F =" << F << std::endl;
  std::cout << "F1 =" << F1 << std::endl;
  std::cout << "N1 =" << N1 << std::endl;
  */
  
  // create output structures
  plhs[0] = mxCreateNumericMatrix(1,F1,mxDOUBLE_CLASS,mxREAL);
  plhs[1] = mxCreateNumericMatrix(1,F1,mxDOUBLE_CLASS,mxREAL);
  plhs[2] = mxCreateNumericMatrix(1,F1,mxDOUBLE_CLASS,mxREAL);
  double *split_val = (double*) mxGetData(plhs[0]);
  double *thrs = (double*) mxGetData(plhs[1]);
  double *gains = (double*) mxGetData(plhs[2]);
  
  double vInit, w, g;
  std::vector<double> W(H, 0.0);
  
  // perform initialization
  vInit = 0; g = 0; w = 0; 
  for(int j=0; j<N1; j++ ) { w+=wts[dids[j]]; W[hs[dids[j]]-1]+=wts[dids[j]]; }
  for(int i=0; i<H; i++ ) g+=gini((W[i]*costs_factor[i])/w); vInit=(1.-g);
  
  #ifdef USEOMP
  nThreads = std::min(nThreads,omp_get_max_threads());
  #pragma omp parallel for num_threads(nThreads)
  #endif
  for(int i=0; i<F1; i++ ) 
  {
    double wl, wr, gl, gr, v, pl, pr;
    double best = vInit;
    uint8* data1 = (uint8*)(data + (fids[i]*size_t(N)));  
    std::vector<double> cdf(nBins, 0.0);
    std::vector<std::vector<double> > cdfs(H, cdf);

    constructCdfPerClass(data1,hs,wts,nBins,N1,dids,cdfs);
    constructCdf(data1,wts,nBins,N1,dids,cdf);
    for(int j=0; j<nBins-1; j++) {
      gl = 0.; gr = 0.;
      wl = cdf[j]; wr = cdf[nBins-1] - cdf[j];
      // gini = 1-\sum_h p_h^2; v = gini_l*pl + gini_r*pr                    
      for (int h=0; h<H; h++) {
        cfactor = costs_factor[h]; 
        pl = (cdfs[h][j]);///wl;
        pl *= cfactor;
        pr = ((cdfs[h][nBins-1]-cdfs[h][j]));///wr;
        pr *= cfactor;
        gl += gini(pl); // Compute gini value for class h+1 in left node
        gr += gini(pr); // Compute gini value for class h+1 in right node        
      }            
      v = (wl-(gl/wl))/w + (wr-(gr/wr))/w;
      
      if( v<best ) {
         best = v;
         split_val[i] = best;
         gains[i] = vInit - best;
         thrs[i] = j;       
      }     
    }
  }
} 

