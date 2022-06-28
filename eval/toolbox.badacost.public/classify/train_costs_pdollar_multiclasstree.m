function [WEAK_LEARNER, predicted] = train_costs_pdollar_multiclasstree( x, y, SAMPLE_WEIGHTS, C, opt)
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

% Sample with remplacement the minority classes to balance the dataset
% (classes with the same number of training samples)
%[x2, y2, w2] = resample_minority_classes( x, y, SAMPLE_WEIGHTS);

%% Undersample the majority classes to balance the dataset (classes with
%% same number of training samples).
%[x2, y2, w2] = sample_majority_classes( x, y, SAMPLE_WEIGHTS);

% Do nothing
x2 = x;
y2 = y;
w2 = SAMPLE_WEIGHTS;

% Build the classification tree.
if opt.USE_SAMPLED_WL
  n = size(x2, 2);
  NUM_SAMPLES = floor(opt.SAMPLE_PROPORTION_WL * n);
  indices = randsample(length(w2),NUM_SAMPLES,true,w2);
  while (length(unique(y2(indices))) ~= length(unique(y2)))
    indices = randsample(length(w2),NUM_SAMPLES,true,w2);
  end
  pTree.costs    = C;
  pTree.dWts     = w2;
%  pTree.minCount = floor(0.1*length(indices)/(num_classes));
%  pTree.minChild = floor(pTree.minCount/2);
  pTree.maxDepth = opt.maxDepth;
  pTree.fracFtrs = opt.frac_features;
  tree           = multiclassTreeTrain(x2(:,indices)', y2(indices)', pTree);
else
  pTree.costs    = C;
  pTree.dWts     = w2;
%  pTree.minCount = floor(0.1*size(x2,2)/(num_classes));
%  pTree.minChild = floor(pTree.minCount/2);
  pTree.maxDepth = opt.maxDepth;
  pTree.fracFtrs = opt.frac_features;
  tree           = multiclassTreeTrain(x2', y2', pTree);
end
                       
% Evaluate the classification tree on the training data.
predicted = multiclassTreeApply(x', tree);
predicted = predicted(:)';
 
WEAK_LEARNER.TREE         = tree;
WEAK_LEARNER.num_classes  = num_classes;

end

function [x2, y2, w2] = sample_majority_classes( x, y, w)
% If we have different number of elements on each class, we 
% sample the majority clases in order to have same number of data 
% on each class (the same as the minority class):
num_classes = length(unique(y));
T = tabulate(y);
[minority_num_data, minority_index]  = min(T(:,2));
x2 = cell(1, num_classes);
y2 = cell(num_classes, 1);
w2 = cell(num_classes, 1);
x2{minority_index} = x(:, y == T(minority_index,1));
y2{minority_index} = minority_index*ones(minority_num_data, 1);
w2{minority_index} = w(y == T(minority_index,1));
for i=[1:minority_index-1, minority_index+1:num_classes]
  x_i = x(:, y == T(i,1));
  w_i = w(y == T(i,1));
  num_i   = size(x_i,2);
  if (num_i == minority_num_data)
    x2{i}   = x_i;
    y2{i}   = i*ones(minority_num_data, 1);
    w2{i}   = w_i;
  else
    % Sampling is needed ...
    % Sample minority_num_data elements (with replacement) from 
    % [1, size(x_i, 2)] range.
    indices = randi(size(x_i, 2), minority_num_data, 1);
    
    x2{i}   = x_i(:, indices);
    y2{i}   = i*ones(minority_num_data, 1);
    w2{i}   = w_i(indices);
  end
end
x2 = cell2mat(x2);
y2 = cell2mat(y2);
w2 = cell2mat(w2);
w2 = w2/sum(w2);

end

function [x2, y2, w2] = resample_minority_classes( x, y, w)
% If we have different number of elements on each class, we 
% resample the minority classes in order to have same number of data 
% on each class:
num_classes = length(unique(y));
T = tabulate(y);
[majority_num_data, majority_index]  = max(T(:,2));
x2 = cell(1, num_classes);
y2 = cell(num_classes, 1);
w2 = cell(num_classes, 1);
x2{majority_index} = x(:, y == T(majority_index,1));
y2{majority_index} = majority_index*ones(majority_num_data, 1);
w2{majority_index} = w(y == T(majority_index,1));
for i=[1:majority_index-1, majority_index+1:num_classes]
  x_i = x(:, y == T(i,1));
  w_i = w(y == T(i,1));
  num_i   = size(x_i,2);
  if (num_i == majority_num_data)
    x2{i}   = x_i;
    y2{i}   = i*ones(majority_num_data, 1);
    w2{i}   = w_i;
  else
    % Sampling is needed ...
    % Sample majority_num_data elements (with replacement) from 
    % [1, size(x_i, 2)] range.
    indices = randi(size(x_i, 2), majority_num_data, 1);
    x2{i}   = x_i(:, indices);
    y2{i}   = i*ones(majority_num_data, 1);
    w2{i}   = w_i(indices);
  end
end
x2 = cell2mat(x2);
y2 = cell2mat(y2);
w2 = cell2mat(w2);
w2 = w2/sum(w2);
end

% function ids = wswor( prob, N, trials )
% % Fast weighted sample without replacement. Alternative to:
% %  ids=datasample(1:length(prob),N,'weights',prob,'replace',false);
% M=length(prob); assert(N<=M); if(N==M), ids=1:N; return; end
% if(all(prob(1)==prob)), ids=randperm(M,N); return; end
% cumprob=min([0 cumsum(prob)],1); assert(abs(cumprob(end)-1)<.01);
% cumprob(end)=1; [~,ids]=histc(rand(N*trials,1),cumprob);
% [s,ord]=sort(ids); K(ord)=[1; diff(s)]~=0; ids=ids(K);
% if(length(ids)<N), ids=wswor(cumprob,N,trials*2); end
% ids=ids(1:N)';
% end
