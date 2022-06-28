function tree = multiclassTreeTrain( data, hs, varargin )
% Train a multiclass tree with costs
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
%  data     - [NxF] N length F feature vectors
%  hs       - [Nx1] or {Nx1} target output labels in [1,H]
%  varargin - additional params (struct or name/value pairs)
%   .H          - [max(hs)] number of classes
%   .costs      - [HxH] H is the number of classes and C(i,j) is 
%                 the cost of estimate class label as j whereas the 
%                 ground thruth class is i.
%   .N1         - [N] number of data points for training each tree
%   .F1         - [sqrt(F)] number features to sample for each node split
%   .split      - ['gini'] optionss include 'gini', 'entropy' and 'twoing'
%   .minCount   - [1] minimum number of data points to allow split
%   .minChild   - [1] minimum number of data points allowed at child nodes
%   .maxDepth   - [64] maximum depth of tree
%   .dWts       - [] weights used for sampling and weighing each data point
%   .fWts       - [] weights used for sampling features
%   .discretize - [] optional function mapping structured to class labels
%                    format: [hsClass,hBest] = discretize(hsStructured,H);
%
% OUTPUTS
%  tree   - learned tree model struct array w the following fields
%   .fids     - [Kx1] feature ids for each node
%   .thrs     - [Kx1] threshold corresponding to each fid
%   .child    - [Kx1] index of child for each node
%   .distr    - [KxH] prob distribution at each node
%   .hs       - [Kx1] or {Kx1} most likely label at each node
%   .count    - [Kx1] number of data points at each node
%   .depth    - [Kx1] depth of each node
%
% EXAMPLE
%  N=10000; H=5; d=2; [xs0,hs0,xs1,hs1]=demoGenData(N,N,H,d,1,1);
%  xs0=single(xs0); xs1=single(xs1);
%  pTrain={'maxDepth',50,'F1',2,'M',150,'minChild',5};
%  tic, forest=forestTrain(xs0,hs0,pTrain{:}); toc
%  hsPr0 = forestApply(xs0,forest);
%  hsPr1 = forestApply(xs1,forest);
%  e0=mean(hsPr0~=hs0); e1=mean(hsPr1~=hs1);
%  fprintf('errors trn=%f tst=%f\n',e0,e1); figure(1);
%  subplot(2,2,1); visualizeData(xs0,2,hs0);
%  subplot(2,2,2); visualizeData(xs0,2,hsPr0);
%  subplot(2,2,3); visualizeData(xs1,2,hs1);
%  subplot(2,2,4); visualizeData(xs1,2,hsPr1);
%
% See also multiclassTreeApply
%
% Piotr's Image&Video Toolbox      Version 3.24
% Copyright 2013 Piotr Dollar.  [pdollar-at-caltech.edu]
% Please email me if you find bugs, or have suggestions or questions!
% Licensed under the Simplified BSD License [see external/bsd.txt]

% get additional parameters and fill in remaining parameters
dfs={ 'H',[], 'costs', [], 'N1',[], 'fracFtrs',[], 'split','gini', 'minCount',1, ...
  'minChild',1, 'maxDepth',[], 'dWts',[], 'fWts',[], 'discretize','' };
[H,costs,N1,fracFtrs,splitStr,minCount,minChild,maxDepth,dWts,fWts,discretize] = ...
  getPrmDflt(varargin,dfs,1);
[N,F]=size(data); assert(length(hs)==N); discr=~isempty(discretize);
minChild=max(1,minChild); minCount=max([1 minCount minChild]);
if(isempty(H)), H=max(hs); end; assert(discr || all(hs>0 & hs<=H));
if(isempty(N1)), N1=round(N); end; N1=min(N,N1);
F1=F; if(~isempty(fracFtrs)),  F1=min(F, round(fracFtrs*F)); end;
if(isempty(dWts)), dWts=ones(1,N,'single'); end; dWts=dWts/sum(dWts);
if(isempty(fWts)), fWts=ones(1,F,'single'); end; fWts=fWts/sum(fWts);
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

% make sure data has correct types
if(~isa(data,'single')), data=single(data); end
if(~isa(hs,'uint32') && ~discr), hs=uint32(hs); end
if(~isa(fWts,'single')), fWts=single(fWts); end
if(~isa(dWts,'single')), dWts=single(dWts); end

% train M random trees on different subsets of data
prmTree = {H,F1,minCount,minChild,maxDepth,fWts,split,discretize};
if(N==N1)
  data1=data; hs1=hs; dWts1=dWts; 
else
  d=wswor(dWts,N1,4); data1=data(d,:); hs1=hs(d);
  dWts1=dWts(d); dWts1=dWts1/sum(dWts1);
end
tree = treeTrain(data1,hs1,dWts1,costs,costs_factor,prmTree);
end

function tree = treeTrain( data, hs, dWts, costs, costs_factor, prmTree )
% Train single random tree.
[H,F1,minCount,minChild,maxDepth,fWts,split,discretize]=deal(prmTree{:});
N=size(data,1); K=2*N-1; 

thrs=zeros(K,1,'single'); 
distr=zeros(K,H,'single');
fids=zeros(K,1,'uint32'); child=fids; count=fids; depth=fids;
hsn=cell(K,1); dids=cell(K,1); dids{1}=uint32(1:N); k=1; K=2;
while( k < K )
  % get node data and store distribution
  dids1=dids{k}; dids{k}=[]; hs1=hs(dids1); n1=length(hs1); count(k)=n1;
  dWts1=dWts(dids1);
  pure=all(hs1(1)==hs1);
  if(pure)
    distr(k,hs1(1))=1; 
    hsn{k}=hs1(1); 
  else
    % We use the data weights to compute the probability of each class in 
    % the node. 
    for i=1:H, distr(k,i)=sum(dWts1(hs1==i)); end;
    distr(k,:)=distr(k,:)/sum(dWts1);
    % In this case we use the minimum costs rule to assign the label
    class_cost = distr(k,:)*costs;
    [~, hsn{k}] = min(class_cost);    
  end; 

  % if pure node or insufficient data don't train split
  if( pure || n1<=minCount || depth(k)>maxDepth ), k=k+1; continue; end
  
  % train split and continue
  fids1=1:size(data,2);
  if (F1<length(fids1)), fids1=wswor(fWts,F1,4); end; % Random election of F1 features.  
  data1=data(dids1,fids1);
  [~,order1]=sort(data1); 
  order1=uint32(order1-1);
  dWts1=dWts(dids1);
  
  %----------------------------------
  [splitSt,thrsSt,gains]=multiclassTreeTrain1(data1,hs1,dWts1,order1,...
                                              H,split,costs_factor);
  [~,fid]=min(splitSt); 
  thr=single(thrsSt(fid)); 
  gain=gains(fid);
  %----------------------------------
  
  fid=fids1(fid); 
  left=data(dids1,fid)<thr; 
  count0=nnz(left);
  if( gain>1e-10 && count0>=minChild && (n1-count0)>=minChild )
    child(k)=K; fids(k)=fid-1; thrs(k)=thr;
    dids{K}=dids1(left); dids{K+1}=dids1(~left);
    depth(K:K+1)=depth(k)+1; K=K+2;
  end; k=k+1;
end
% create output model struct
K=1:K-1; 
%if(discr) 
%  hsn={hsn(K)}; 
%else
  hsn=[hsn{K}]'; 
%end
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
