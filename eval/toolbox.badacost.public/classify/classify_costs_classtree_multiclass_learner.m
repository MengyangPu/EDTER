function predicted = classify_costs_classtree_multiclass_learner( CLASSIFIER, x )
% function predicted = classify_costs_classtree_multiclass_learner( CLASSIFIER, x )
% Classify a set of projected using pixel-wise grey level difference.
%
% Input:
%   CLASSIFIER, a structure returned by train_classtree_multiclass_learner.m
%   x, Pxn matrix (P is the number of pixels per image n is the number 
%      of images).
%
% Output:
%     predicted -- 1 x n matrix with the lower cost  weak
%                   classifier output for the input data in x. 
%
% See also:
%   train_COST_SAMME.m

% Author: Jose M. Buenaposada
n           = size(x,2);
  
% Evaluate the classification tree on the training data.
z = CLASSIFIER.TREE.eval( x' );
predicted = cellfun(@(x) str2double(x), z);






