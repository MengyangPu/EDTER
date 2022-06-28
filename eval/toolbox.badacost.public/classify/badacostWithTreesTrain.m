function [model, cost_curve] = badacostWithTreesTrain( X0, X1, L1, varargin )
%--------------------------------------------------------------------------
% function [model, cost_curve] = badacostWithTreesTrain( X0, X1, L1, varargin)
%
% This function performs Multiclass Bosting ADApted for Costs (badacost) 
% algorithm with trees as weak learner. 
%
% INPUTS
%  X0         - [N0xF] negative feature vectors
%  X1         - [N1xF] positive feature vectors
%  L1         - [N1x1] positive subclasses labels  (in [1, H] where H is the #subclasses)
%  pBoost     - additional params (struct or name/value pairs)
%   .nWeak      - [128] number of trees to learn
%   .shrinkage  - [0.1] learning rate (multiplicative factor for weak learner weights).
%   .resampling - [0.1] sampling proportion of data for weak learner training.
%   .Cost       - [onesH+1,H+1)-eye(H+1))] C(i, j) is the cost of classifing in j whereas the 
%                 real label is i.
%   .stopAtNegWeight [1] Stop iterating badacost when a Weak Learner gets a negative weight
%   .maxDepth   - [] Max depth levels of the decision trees.
%   .quantized  - [0]
%   .use_rus    - [0] Use Random Under Sampling.
%   .verbose    - [0] if true print status information
%
% OUTPUTS
%  model      - learned boosted tree classifier w the following fields
%   .fids       - [K x nWeak] feature ids for each node
%   .thrs       - [K x nWeak] threshold corresponding to each fid
%   .child      - [K x nWeak] index of child for each node (1-indexed)
%   .hs         - [K x nWeak] log ratio (.5*log(p/(1-p)) at each node
%   .weights    - [K x nWeak] total sample weight at each node
%   .depth      - [K x nWeak] depth of each node
%   .errs       - [1 x nWeak] error for each tree (for debugging)
%   .losses     - [1 x nWeak] loss after every iteration (for debugging)
%   .treeDepth  - depth of all leaf nodes (or 0 if leaf depth varies)
%

%                                         
% Author: Jose M. Buenaposada

x = [X0; X1];

% We make 1 the negative class label and add 1 to the rest of labels
y = [ones(size(X0,1), 1); L1(:) + 1]; 
num_classes = length(unique(y));

% get additional parameters
Cost = ones(num_classes, num_classes) - diag(ones(num_classes, 1));
dfs={ 'nWeak',128, 'shrinkage', 0.1, 'resampling', 0.5, 'Cost', Cost, 'stopAtNegWeight', 1, ...
      'fracFtrs', 1, 'minDepth', [], 'maxDepth', [], 'quantized', 0, 'variable_depth', 0, ...
      'use_rus', 0, ...
      'verbose', 0};
[nWeak,shrinkage,resampling,Cost,stopAtNegWeight,fracFtrs,minDepth,maxDepth,quantized,variable_depth,use_rus,verbose]=getPrmDflt(varargin,dfs,1);

opt.SET_MINPARENT_TREE_WL = 1;
opt.learning_rate         = shrinkage;
opt.frac_features         = fracFtrs;
opt.num_iterations        = nWeak;
opt.scalar_class_labels   = 1;
opt.stop_with_negative_wl_weight = stopAtNegWeight;
opt.maxDepth              = maxDepth;
opt.minDepth              = minDepth;
opt.USE_QUANTIZED         = quantized;
opt.use_variable_depth    = variable_depth;
opt.USE_RUS_BALANCED_WL   = use_rus; % Use Random Under Sampling Balancing .
opt.verbose               = verbose;
if opt.USE_QUANTIZED 
  opt.train_weak_learner    = @train_costs_pdollar_quantized_multiclasstree;
else
  opt.train_weak_learner    = @train_costs_pdollar_multiclasstree;
end
opt.classify_weak_learner = @classify_costs_pdollar_multiclasstree; 
if ((resampling >= 1.0) || (resampling < 0))
  opt.USE_SAMPLED_WL        = 0;
  opt.USE_RUS_BALANCED_WL   = 0;
  opt.SAMPLE_PROPORTION_WL  = resampling;
elseif ~opt.USE_RUS_BALANCED_WL
  opt.USE_SAMPLED_WL        = 1;
  opt.SAMPLE_PROPORTION_WL  = resampling;
else
  opt.USE_SAMPLED_WL        = 0;
  opt.SAMPLE_PROPORTION_WL  = resampling;        
end

msg='Training BAdaCost: nWeak=%3i shrinkage=%f resampling=%f fracFtrs=%f\n';
if(verbose), fprintf(msg,nWeak,shrinkage,resampling,fracFtrs); start=clock; end

[classfr, cost_curve] = badacostTrain( x', y, Cost, opt );
%figure; 
%plot(cost_curve);
%xlabel('#weak learners');
%ylabel('cost');

% Now we adapt the output of the badacost to the P.Dollar toolbox.
% create output model struct
k=0; 
nWeak = length(classfr.WEAK_LEARNERS);
for i=1:nWeak
   k=max(k,size(classfr.WEAK_LEARNERS{i}.TREE.fids,1)); 
end
Z = @(type) zeros(k,nWeak,type);
model=struct( 'fids',Z('uint32'), 'thrs',Z('single'), ...
  'child',Z('uint32'), 'hs',Z('single'), 'weights',Z('single'), ...
  'depth',Z('uint32')); %, 'errs',errs, 'losses',losses );
for i=1:nWeak
  T=classfr.WEAK_LEARNERS{i}.TREE; 
  k=size(T.fids,1);
  model.fids(1:k,i)=T.fids; 
  model.thrs(1:k,i)=T.thrs;
  model.child(1:k,i)=T.child; 
  model.hs(1:k,i)=T.hs;
%  model.weights(1:k,i)=T.weights; 
  model.depth(1:k,i)=T.depth;
end
depth = max(model.depth(:));
model.treeDepth = depth * uint32(all(model.depth(~model.child)==depth));
model.num_classes = classfr.num_classes;
model.Cprime      = classfr.Cprime;
model.Y           = classfr.Y;
model.wl_weights  = classfr.WEIGHTS;
model.weak_learner_type = 'trees';

% output info to log
predicted = badacostWithTreesApply(x, model);
error = sum(predicted(:) ~= y(:))/length(predicted);
fp    = sum((predicted(:) ~= 1) & (y == 1))./length(predicted);
fn    = sum((predicted(:) == 1) & (y ~= 1))./length(predicted);
msg='Done training err=%.4f fp=%.4f fn=%.4f (t=%.1fs).\n';
if(verbose)
  fprintf(msg,error,fp,fn,etime(clock,start)); 
end

end
