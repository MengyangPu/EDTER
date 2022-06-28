function predicted = badacostApply(X, classfr)
% Apply learned badacost weak learners ensemble 
%
% USAGE
%  hs = badacostApply( X, model, [maxDepth], [minWeight], [nThreads] )
%
% INPUTS
%  X          - [FXN] N num vectors to classfy, F num feature vectors
%  model      - learned boosted tree classifier
%
% OUTPUTS
%  predicted  - [Nx1] predicted output labels
%
% EXAMPLE
%
% See also badacostTrain
%
% Author: Antonio Baldera, modified by Jose M. Buenaposada

n          = size(X,2);
margin_vec = zeros(classfr.num_classes, n);

for i=1:length(classfr.WEAK_LEARNERS) 
  % z is a row vector with the labels
  z  = classfr.classify_weak_learner(classfr.WEAK_LEARNERS{i}, X);   
  
  for j=1:n
    margin_vec(:,j) = margin_vec(:,j) + (classfr.WEIGHTS(i).*classfr.Y(:, z(j)));
  end
end;

% WARNING: Change to accomodate with theory (2016/11)
%[~, predicted] = min(classfr.Cprime' * margin_vec);
[~, predicted] = min(classfr.Cprime * margin_vec);
predicted = predicted(:); 

end
