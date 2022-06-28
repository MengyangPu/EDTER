function [tree, data] = multiclassQuantizedTreeTrain( data, varargin )
% Train a multiclass tree with costs. Optimized code for training 
% decision trees over binary variables.
%
% During training each feature is quantized to lie between [0,nBins-1],
% where nBins<=256. Quantization is expensive and should be performed just
% once if training multiple trees. Note that the second output of the
% algorithm is the quantized data, this can be reused in future training.
%
% Dimensions:
%  F - number features
%  N - number input vectors
%  H - number classes
%
% USAGE
%  tree = multiclassTreeTrain( data, hs, [varargin] )
%
% INPUTS
%  data   - Data for training tree
%   .X      - [NxF] N length F feature vectors
%   .hs     - [Nx1] target output labels in [1,H]
%   .H      - [max(hs)] number of classes
%   .wts    - [] weights used for sampling and weighing each data point
%   .xMin   - [1xF] optional vals defining feature quantization
%   .xStep  - [1xF] optional vals defining feature quantization
%   .xType  - [] optional original data type for features
%  pTree - additional params (struct or name/value pairs)
%   .costs      - [] HxH cost matrix, H is the number of classes and C(i,j) is 
%                 the cost of estimate class label as j whereas the 
%                 ground thruth class is i.
%   .split      - ['gini'] options include 'gini', 'entropy' and 'twoing'
%   .nBins      - [256] maximum number of quanizaton bins (<=256)
%   .maxDepth   - [1] maximum depth of tree
%   .minCount   - [10] minimum number of data in a node to allow split
%   .fracData   - [1] fraction of data to sample for each node split
%   .fracFtrs   - [1] fraction of features to sample for each node split
%   .nThreads   - [inf] max number of computational threads to use
%
% OUTPUTS
%  tree   - learned tree model struct array w the following fields
%   .fids     - [Kx1] feature ids for each node
%   .thrs     - [Kx1] threshold corresponding to each fid
%   .child    - [Kx1] index of child for each node
%   .hs       - [Kx1] lowest cost label at each node
%   .distr    - [KxH] prob distribution at each node
%   .count    - [Kx1] number of data points at each node
%   .depth    - [Kx1] depth of each node
%  data       - data used for training tree (quantized version of input)
%
%
% See also multiclassTreeApply
%
% José M. Buenaposada - Mixed version between binaryTreeTrain and
% forestTrain. There is a fundamental change, the addition of a cost matrix
% to allow BAdaCost training. 
%
% Piotr's Image&Video Toolbox      Version 3.24
% Copyright 2013 Psiotr Dollar.  [pdollar-at-caltech.edu]
% Please email me if you find bugs, or have suggestions or questions!
% Licensed under the Simplified BSD License [see external/bsd.txt]

% get additional parameters and fill in remaining parameters
dfs={'costs', [], ...
     'split','gini',...
     'nBins',256, ...
     'maxDepth',1, ...
     'minCount',10,...
     'fracData',1, ...
     'fracFtrs',1, ...
     'nThreads',1e5}; %,...
%      'dWts',[], 'fWts',[] };
[costs, splitStr, nBins, maxDepth, minCount, fracData, fracFtrs, nThreads] = ...
  getPrmDflt(varargin,dfs,1);
assert(nBins<=256);

% get data and normalize weights
dfs={'X','REQ', 'hs', 'REQ', 'H', [], 'wts',[],  ...
     'xMin',[], 'xStep',[], 'xType',[] };
[X,hs,H,wts,xMin,xStep,xType]=getPrmDflt(data,dfs,1);
[N,F]=size(X); assert(length(hs)==N); 

if(isempty(xType)), xMin=zeros(1,F); xStep=ones(1,F); xType=class(X); end
if(isempty(H)), H=max(hs); end; assert(all(hs>0 & hs<=H));
if(isempty(wts)), wts=ones(1,N,'single'); end; wts=wts/sum(wts);
%split=find(strcmpi(splitStr,{'gini','entropy','twoing'}))-1;
split=find(strcmpi(splitStr,{'gini'}))-1;
if(isempty(split)), error('unknown splitting criteria: %s',splitStr); end

if(isempty(costs)) 
   costs = ones(H,H);
   % Make the costs matrix diagonal zeros.
   costs = costs - diag(diag(data.costs));
end
assert(size(costs, 1) == H);
assert(size(costs, 2) == H); 

% Compute multiplicative factor for probabilities of class in each node.
% This factors, that accounts for the cost, will take care of somewhat
% cost sensitive impurity computation at each node.
costs_factor = sum(costs, 2)';
costs_factor = H*costs_factor / sum(costs_factor);

% quantize data to be between [0,nBins-1] if not already quantized
if( ~isa(data.X,'uint8') )
  xMin = min(X)-.01;
  xMax = max(X)+.01;
  xStep = (xMax-xMin) / (nBins-1);
  X = uint8(bsxfun(@times,bsxfun(@minus,data.X,xMin),1./xStep));
else
  X = data.X;
end
data=struct( 'X',X, 'hs', hs, 'H', H, 'wts',wts,  ...
  'xMin',xMin, 'xStep',xStep, 'xType',xType );

% make sure data has correct types
if(~isa(X,'uint8')), X=single(X); end
if(~isa(hs,'uint32')), hs=uint32(hs); end
if(~isa(wts,'single')), wts=single(wts); end

% train M random trees on different subsets of data
prmTree = {nBins,xMin,xStep,H,fracFtrs,minCount,maxDepth,split,nThreads};
if(fracData==1)
  X1=X; hs1=hs; wts1=wts; 
else
  d=wswor(wts,round(N*fracData),4); X1=X(d,:); hs1=hs(d);
  wts1=wts(d); wts1=wts1/sum(wts1);
end
tree = treeTrain(X1,hs1,wts1,costs,costs_factor,prmTree);
end

function tree = treeTrain( X, hs, wts, costs, costs_factor, prmTree )
% Train single random tree.
[nBins,xMin,xStep,H,fracFtrs,minCount,maxDepth,split,nThreads]=deal(prmTree{:});
N=size(X,1); K=2*N-1; 

thrs=zeros(K,1,'single'); 
distr=zeros(K,H,'single');
fids=zeros(K,1,'uint32'); child=fids; count=fids; depth=fids;
hsn=cell(K,1); dids=cell(K,1); dids{1}=uint32(1:N); k=1; K=2;
while( k < K )
  % get node data and store distribution
  dids1=dids{k}; dids{k}=[]; hs1=hs(dids1); n1=length(hs1); count(k)=n1;
  wts1=wts(dids1);
  pure=all(hs1(1)==hs1);
  if(pure)
    distr(k,hs1(1))=1; 
    hsn{k}=hs1(1); 
  else
    % We use the data weights to compute the probability of each class in 
    % the node. 
    for i=1:H, distr(k,i)=sum(wts1(hs1==i)); end;
    distr(k,:)=distr(k,:)/sum(wts1);
    % In this case we use the minimum costs rule to assign the label
    class_cost = distr(k,:)*costs;
    [~, hsn{k}] = min(class_cost);    
  end; 

  % if pure node or insufficient data don't train split
  if( pure || n1<=minCount || depth(k)>maxDepth ), k=k+1; continue; end
  
  % train split and continue
  F=size(X,2);
  fids1=1:F;  
  if (fracFtrs<1), fids1=randperm(F,floor(F*fracFtrs)); end; % Random election features.  
  fids1=uint32(fids1);
  
  %----------------------------------
  [splitSt,thrsSt,gains]=multiclassQuantizedTreeTrain1(X,hs,...
                                              wts,...
                                              nBins, ...
                                              dids1-1, ...
                                              fids1-1, ...
                                              H, ...
                                              split,...
                                              costs_factor,...
                                              nThreads);
  [~,fid]=min(splitSt); 
  thr=single(thrsSt(fid))+0.5; 
  gain=gains(fid);
  %----------------------------------
  fid=fids1(fid); 
  left=X(dids1,fid)<thr; 
% count0=nnz(left);
%  if( gain>1e-10 && count0>=minChild && (n1-count0)>=minChild )
  if( gain>1e-10 && (any(left) && any(~left)) )      
    thrs(k) = xMin(fid)+xStep(fid)*thr;      
    child(k)=K; fids(k)=fid-1; %thrs(k)=thr;
    dids{K}=dids1(left); dids{K+1}=dids1(~left);
    depth(K:K+1)=depth(k)+1; K=K+2;
  end; k=k+1;
end
% create output model struct
K=1:K-1; 
hsn=[hsn{K}]'; 
tree=struct('fids',fids(K),'thrs',thrs(K),'child',child(K),...
  'distr',distr(K,:),'hs',hsn,'count',count(K),'depth',depth(K));
end

function ids = wswor( prob, N, trials )
% Fast weighted sample without replacement. Alternative to:
%  ids=datasample(1:length(prob),N,'weights',prob,'replace',false);
M=length(prob); assert(N<=M); if(N==M), ids=1:N; return; end
if(all(prob(1)==prob)), ids=randperm(M,N); return; end
cumprob=min([0 cumsum(prob)],1); assert(abs(cumprob(end)-1)<.01);
cumprob(end)=1; [~,ids]=histc(rand(N*trials,1),cumprob);
[s,ord]=sort(ids); K(ord)=[1; diff(s)]~=0; ids=ids(K);
if(length(ids)<N), ids=wswor(cumprob,N,trials*2); end
ids=ids(1:N)';
end
