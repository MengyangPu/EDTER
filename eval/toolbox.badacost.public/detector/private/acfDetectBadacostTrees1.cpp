/*******************************************************************************
* Added by Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
* Copyright 2016 
* Please email me if you find bugs, or have suggestions or questions!
* Licensed under the Simplified BSD License [see external/bsd.txt]
*******************************************************************************/
#include "mex.h"
#include <vector>
#include <cmath>
#include <limits>
#include <iostream>
#include <blas.h>
using namespace std;

typedef unsigned int uint32;

#ifdef USEOMP
#include <omp.h>
#endif

// Here we asume that first label (h=1 at index 0) is the negative class 
// (the background in detection).
inline void getMinPositiveCost(int num_classes, 
                               double *Cprime, 
                               std::vector<double>& margin_vector, 
                               double& min_value, 
                               int& h)
{
  min_value = std::numeric_limits<double>::max();
  for(int j=1; j < num_classes; j++)
  {
    double cost = 0.0;
    double* cprime_column = Cprime + static_cast<size_t>(j*num_classes);
    for(int i=0; i < num_classes; i++)
    {
      cost += cprime_column[i] * margin_vector[i];
    }

    if (cost < min_value) 
    {
      min_value = cost;
      h = j+1;
    }
  }        
}

inline void getNegativeCost(int num_classes, double *Cprime, 
        std::vector<double>& margin_vector, double& neg_cost)
{
  // The index of the negative class is assumed 1. Therefore, its
  // column in Cprime is the first one (no need to move to its column).
  neg_cost = 0.0;
  double* cprime_column = Cprime;
  for(int i=0; i < num_classes; i++)
  {
    neg_cost += cprime_column[i] * margin_vector[i];
  }
}

inline void getChild( float *chns1, uint32 *cids, uint32 *fids,
   float *thrs, uint32 offset, uint32 &k0, uint32 &k )
{
  float ftr = chns1[cids[fids[k]]];
  k = (ftr<thrs[k]) ? 1 : 2;
  k0=k+=k0*2; k+=offset;
}

// [bbs, labels] = acDetectImgMulticlass1(data, trees, 
//                              shrink, height, width, stride, cascThr,
//
// We assume that the label for the negative class is 1, the rest
// of labels are subclasses of the positive metaclass (for example, 
// different possible views of cars).
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
  // get inputs
  float *chns = (float*) mxGetData(prhs[0]);
  mxArray *trees = (mxArray*) prhs[1];
  const int shrink = (int) mxGetScalar(prhs[2]);
  const int modelHt = (int) mxGetScalar(prhs[3]);
  const int modelWd = (int) mxGetScalar(prhs[4]);
  const int stride = (int) mxGetScalar(prhs[5]);
  const float cascThr = (float) mxGetScalar(prhs[6]);
  
//   std::cout << "cascThr=" << cascThr << std::endl;

  // extract relevant fields from trees
  float *thrs = (float*) mxGetData(mxGetField(trees,0,"thrs"));
  float *hs = (float*) mxGetData(mxGetField(trees,0,"hs"));
  uint32 *fids = (uint32*) mxGetData(mxGetField(trees,0,"fids"));
  uint32 *child = (uint32*) mxGetData(mxGetField(trees,0,"child"));
  const int treeDepth = mxGetField(trees,0,"treeDepth")==NULL ? 0 :
    (int) mxGetScalar(mxGetField(trees,0,"treeDepth"));
   
  // Get the BAdaCost related fields from trees.
  double *Cprime = (double*) mxGetData(mxGetField(trees,0,"Cprime"));
  double *Y = (double*) mxGetData(mxGetField(trees,0,"Y"));
  // weak learner weights from boosting
  double *wl_weights = (double*) mxGetData(mxGetField(trees,0,"wl_weights"));
//   const double num_classes = mxGetField(trees,0,"num_classes")==NULL ? -1 :
//     (int) mxGetScalar(mxGetField(trees,0,"num_classes"));  
  const int num_classes = mxGetField(trees,0,"num_classes")==NULL ? -1 :
    (int) mxGetScalar(mxGetField(trees,0,"num_classes"));  

  
  // get dimensions and constants
  const mwSize *chnsSize = mxGetDimensions(prhs[0]);
// 64 bits change
//  const int height = (int) chnsSize[0];
//  const int width = (int) chnsSize[1];
//  const int nChns = mxGetNumberOfDimensions(prhs[0])<=2 ? 1 : (int) chnsSize[2];
  const mwSize height = (mwSize) chnsSize[0];
  const mwSize width = (mwSize) chnsSize[1];
  const mwSize nChns = mxGetNumberOfDimensions(prhs[0])<=2 ? 1 : (mwSize) chnsSize[2];
  const mwSize *fidsSize = mxGetDimensions(mxGetField(trees,0,"fids"));
// 64 bits change
//   const int nTreeNodes = (int) fidsSize[0];
//   const int nTrees = (int) fidsSize[1];
//   const int height1 = (int) ceil(float(height*shrink-modelHt+1)/stride);
//   const int width1 = (int) ceil(float(width*shrink-modelWd+1)/stride);  
  const mwSize nTreeNodes = (mwSize) fidsSize[0];
  const mwSize nTrees = (mwSize) fidsSize[1];
  const mwSize height1 = (mwSize) ceil(float(height*shrink-modelHt+1)/stride);
  const mwSize width1 = (mwSize) ceil(float(width*shrink-modelWd+1)/stride);

  // Create the margin vector and costs vector for BAdaCost classification.
  std::vector<double> margin_vector(num_classes);
  std::vector<double> costs_vector(num_classes);

  // construct cids array
  int nFtrs = modelHt/shrink*modelWd/shrink*nChns;
  uint32 *cids = new uint32[nFtrs]; 
  // 64 bits change
  //int m=0; 
  mwSize m=0;
  for( int z=0; z<nChns; z++ )
    for( int c=0; c<modelWd/shrink; c++ )
      for( int r=0; r<modelHt/shrink; r++ )
        cids[m++] = z*width*height + c*height + r;

  // apply classifier to each patch
//  int num_windows = width1*height1;
  mwSize num_windows = width1*height1;
  if (num_windows < 0)  // Detection window is too big for the image 
     num_windows = 0;   // Detect on 0 windows in this case (do nothing).
  vector<int> rs(num_windows), cs(num_windows); 
  vector<float> hs1(num_windows), scores(num_windows);
//   cout << "Process n= " << width1*height1 << " windows." <<  endl;

  #ifdef USEOMP
  int nThreads = omp_get_max_threads();
  #pragma omp parallel for num_threads(nThreads)
  #endif      
//   for( int c=0; c<width1; c++ ) 
//   for( int r=0; r<height1; r++ ) {
  for( mwSize c=0; c<width1; c++ ) 
  for( mwSize r=0; r<height1; r++ ) {            
    std::vector<double> margin_vector(num_classes);
    double trace;
    int h;
    float *chns1=chns+(r*stride/shrink) + (c*stride/shrink)*height;
    
    // Initialise the margin_vector memory to 0.0
    for(int i=0; i<num_classes; i++)
    {
      margin_vector[i] = 0.0;
    }
    
//    int t;
    if( treeDepth==1 ) 
    {
      // specialized case for treeDepth==1
      for(int t = 0; t < nTrees; t++ ) 
      {
        uint32 offset=t*nTreeNodes, k=offset, k0=0;
        getChild(chns1,cids,fids,thrs,offset,k0,k);

        // In the BadaCost case we have to:
        // 1) Codify as a margin vector class label
        //    output from each tree (hs[k]) 
        // 2) Add the margin vectors codified weighted 
        //    by the weak learner weight.
        h = static_cast<int>(hs[k]);
//        double* codified = Y + static_cast<size_t>(num_classes*(h-1));  
        double* codified = Y + static_cast<mwSize>(num_classes*(h-1));  
        for(int i=0; i<num_classes; i++)
        {
          margin_vector[i] += codified[i] * wl_weights[t];
        }               
        
        double min_pos_cost;
        double neg_cost;
        // Gets positive class min cost and label in h!        
        getMinPositiveCost(num_classes, Cprime, margin_vector, min_pos_cost, h);
        getNegativeCost(num_classes, Cprime, margin_vector, neg_cost);
        trace = -(min_pos_cost - neg_cost);

        if (trace <=cascThr) break;
      }
    } 
    else if( treeDepth==2 ) 
    {
      // specialized case for treeDepth==2
      for(int t = 0; t < nTrees; t++ ) 
      {
        uint32 offset=t*nTreeNodes, k=offset, k0=0;
        getChild(chns1,cids,fids,thrs,offset,k0,k);
        getChild(chns1,cids,fids,thrs,offset,k0,k);
        
        // In the BadaCost case we have to:
        // 1) Codify as a margin vector class label
        //    output from each tree (hs[k]) 
        // 2) Add the margin vectors codified weighted 
        //    by the weak learner weight.
        h = static_cast<int>(hs[k]);
        //        double* codified = Y + static_cast<size_t>(num_classes*(h-1));  
        double* codified = Y + static_cast<mwSize>(num_classes*(h-1));  
        for(int i=0; i<num_classes; i++)
        {
          margin_vector[i] += codified[i] * wl_weights[t];
        }        

        double min_pos_cost;
        double neg_cost;
        // Gets positive class min cost and label in h!        
        getMinPositiveCost(num_classes, Cprime, margin_vector, min_pos_cost, h);
        getNegativeCost(num_classes, Cprime, margin_vector, neg_cost);
        trace = -(min_pos_cost - neg_cost);
        
        if (trace <=cascThr) break;           
      }
    } 
    else if( treeDepth>2) 
    {
      // specialized case for treeDepth>2
      for(int t = 0; t < nTrees; t++ ) {
        uint32 offset=t*nTreeNodes, k=offset, k0=0;
        for( int i=0; i<treeDepth; i++ )
          getChild(chns1,cids,fids,thrs,offset,k0,k);
        
        // In the BadaCost case we have to:
        // 1) Codify as a margin vector class label
        //    output from each tree (hs[k]) 
        // 2) Add the margin vectors codified weighted 
        //    by the weak learner weight.
        h = static_cast<int>(hs[k]);
//        double* codified = Y + static_cast<size_t>(num_classes*(h-1));  
        double* codified = Y + static_cast<mwSize>(num_classes*(h-1));  
        for(int i=0; i<num_classes; i++)
        {
          margin_vector[i] += codified[i] * wl_weights[t];
        }                                      

        double min_pos_cost;
        double neg_cost;
        // Gets positive class min cost and label in h!        
        getMinPositiveCost(num_classes, Cprime, margin_vector, min_pos_cost, h);
        getNegativeCost(num_classes, Cprime, margin_vector, neg_cost);
        trace = -(min_pos_cost - neg_cost);
        
        if (trace <=cascThr) break;                      
      }
    } 
    else 
    {
      // general case (variable tree depth)
      for(int t = 0; t < nTrees; t++ ) 
      {
        uint32 offset=t*nTreeNodes, k=offset, k0=k;
        while( child[k] ) 
        {
          float ftr = chns1[cids[fids[k]]];
          k = (ftr<thrs[k]) ? 1 : 0;
          k0 = k = child[k0]-k+offset;
        }

        // In the BadaCost case we have to:
        // 1) Codify as a margin vector class label
        //    output from each tree (hs[k]) 
        // 2) Add the margin vectors codified weighted 
        //    by the weak learner weight.
        h = static_cast<int>(hs[k]);
        double* codified = Y + static_cast<size_t>(num_classes*(h-1));        
        for(int i=0; i<num_classes; i++)
        {
          margin_vector[i] += codified[i] * wl_weights[t];
        }                               

        double min_pos_cost;
        double neg_cost;
        // Gets positive class min cost and label in h!        
        getMinPositiveCost(num_classes, Cprime, margin_vector, min_pos_cost, h);
        getNegativeCost(num_classes, Cprime, margin_vector, neg_cost);
        trace = -(min_pos_cost - neg_cost);
        
        if (trace <=cascThr) break;                      
      }
      
//      cout << "trees executed t=" << t << ", of nTrees=" << nTrees << endl;
//       cout << "trace=" << trace << endl;
    
      if (trace < 0) h=1;

      // Negative class is label 1, all the other labels are subclasses 
      // within the positive metaclass (e.g. the different views of a car).    
      if (h > 1) 
      {
        mwSize index = c + (r*width1);
        cs[index] = c; 
        rs[index] = r; 
        hs1[index] = h; 
        scores[index] = trace; 
      }
//       cout << "h=" << h << endl;
//       cout << "==================" << endl;
    }
  }
  delete [] cids; 
  
  // convert to bbs
  mwSize size_output=0;
  for( int i=0; i<hs1.size(); i++ ) {
    if (hs1[i] > 1) 
    {
      size_output++;
    }
  }
  
  plhs[0] = mxCreateNumericMatrix(size_output,5,mxDOUBLE_CLASS,mxREAL);
  double *bbs = (double*) mxGetData(plhs[0]);
  plhs[1] = mxCreateNumericMatrix(size_output,1,mxDOUBLE_CLASS,mxREAL);
  double *labels = (double*) mxGetData(plhs[1]);
  m = 0;
  for( int i=0; i<hs1.size(); i++ ) {
    if (hs1[i]>1) {
      bbs[m+0*size_output]=cs[i]*stride; 
      bbs[m+2*size_output]=modelWd;
      bbs[m+1*size_output]=rs[i]*stride; 
      bbs[m+3*size_output]=modelHt;
      // IMPORTANT! We make the score of BAdacost the minus the diference 
      // between the min positive class cost and the negative class cost
      bbs[m+4*size_output]=scores[i];    
      labels[m]=hs1[i];
      m++;
    }
  }
}
