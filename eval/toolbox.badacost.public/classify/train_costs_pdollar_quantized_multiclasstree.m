function [WEAK_LEARNER, predicted, data] = train_costs_pdollar_quantized_multiclasstree( x, y, SAMPLE_WEIGHTS, C, opt)
% function [WEAK_LEARNER, predicted] = train_costs_multiclasstree( x, y, SAMPLE_WEIGHTS, C, opt)
%
% This function trains a classification tree on a resampled (using SAMPLE_WEIGHTS
% distribution) subset of the training data.
%
% The weak classifier ouput is multiclass with the output codified for 
% the predicetd class k (of c classes):
%
%   [-1/(c-1), -1/(c-1), ..., 1, ..., -1/(c-1)] 
%
%  where the 1 is in the k-th position.
%
%
% Input:
%   x, Pxn matrix (P is the number of features and n is the 
%                  number of samples).
%   y, c x n matrix with a column vector of c components for each of the n
%      observations in x.
%   SAMPLE_WEIGHTS, the weigth of each data in x so far. They are assumed
%   to be normalised to sum 1.
%   C, c x c matrix with the costs of the problem. C(i,j) is the costs of
%   classify on class j, given the datum is of class i.
%
% Output:
%   WEAK_LEARNER, a structure with the lower error weak classifier found
%     WEAK_LEARNER.TREE  -- Trained decision tree (using classregtree)
%     predicted -- c x n matrix (c is the number of classes and n is the 
%                   number of observations) with the lower error weak
%                   classifier output for the input data in x. The i-th 
%                   column is the codified vector corresponding to the 
%                   estimated class for i-th sample.

if ~isfield(opt, 'USE_SAMPLED_WL')
  opt.USE_SAMPLED_WL = 0;
end

if ~isfield(opt, 'SAMPLE_PROPORTION_WL')
  opt.SAMPLE_PROPORTION_WL = 0.1;
end

if ~isfield(opt, 'frac_features')
  opt.frac_features = 1;
end

n  = size(x,2);  % Number of training samples
%n_features  = size(x,1);  % Number of features

num_classes = length(unique(y));

% Build the classification tree.
% Build the classification tree.
pTree.costs    = C;
pTree.maxDepth = opt.maxDepth;
pTree.fracFtrs = opt.frac_features;
if ~isfield(opt, 'data')
  opt.data.X  = x';
  opt.data.hs = y';
end
opt.data.wts = SAMPLE_WEIGHTS;  

if ( ~isa(opt.data.X,'uint8') )
  % Convert all data to uint8 first time
  [~, opt.data] = multiclassQuantizedTreeTrain(opt.data, pTree);
end

if opt.USE_SAMPLED_WL
  n = size(x, 2);
  NUM_SAMPLES = floor(opt.SAMPLE_PROPORTION_WL * n);
  indices = randsample(length(SAMPLE_WEIGHTS),NUM_SAMPLES,true,SAMPLE_WEIGHTS);
  while (length(unique(y(indices))) ~= length(unique(y)))
    indices = randsample(length(SAMPLE_WEIGHTS),NUM_SAMPLES,true,SAMPLE_WEIGHTS);
  end
  
  data2     = opt.data;
  data2.X   = data2.X(indices, :);
  data2.hs  = data2.hs(indices);
  [tree, ~] = multiclassQuantizedTreeTrain(data2, pTree);
  data      = opt.data;
else
  [tree, data] = multiclassQuantizedTreeTrain(opt.data, pTree);
end

% Evaluate the classification tree on the training data.
predicted = multiclassTreeApply(x', tree);
predicted = predicted(:)';
 
WEAK_LEARNER.TREE         = tree;
WEAK_LEARNER.num_classes  = num_classes;

end

