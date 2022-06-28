function [thr, h1] = badacostCalibrateCascade(X0, X1, clf, use_trees)
% Classify all the data with the classifier and check the 
% costs traces. We compare positive class trace (label 1) with 
% the min cost of the positive classes. We use the formula:
%
%  traces(i,:) = -(min_pos_costs - costs(1,:));
%
%  Where traces(i, :) are the traces for the i-th badacost weak learner,
%  costs(1,:) are the costs associated with the negative class at 
%  i-th badacost weak-learner and min_pos_cost is the min cost at the 
%  i-th weak-learner of the positive classes.

if nargin < 4
  use_trees = 1;
end

% 1. Compute costs traces ...
if use_trees
  nWeaks = size(clf.fids, 2);
else
  nWeaks = length(clf.WEAK_LEARNERS);
end
n0 = size(X0, 1);
n1 = size(X1, 1);
%n  = n0 + n1; 
margin_vec0 = zeros(clf.num_classes, n0);
margin_vec1 = zeros(clf.num_classes, n1);
traces0     = zeros(nWeaks, n0);
traces1     = zeros(nWeaks, n1);

for i=1:nWeaks
  if use_trees
    % z is a row vector with the labels
    ids0 = forestInds(X0,clf.thrs(:,i),clf.fids(:,i),clf.child(:,i), 1e5);           
    for j=1:n0
      z = clf.hs(ids0(j),i);
      margin_vec0(:,j) = margin_vec0(:,j) + ...
                        (clf.wl_weights(i).*clf.Y(:, z));
    end
  else % General case with other weak learners
    % z is a row vector with the labels
    z  = clf.classify_weak_learner(clf.WEAK_LEARNERS{i}, X0');     
    for j=1:n0
      margin_vec0(:,j) = margin_vec0(:,j) + (clf.WEIGHTS(i).*clf.Y(:, z(j)));
    end
  end
  % WARNING: Change to accomodate with theory (2016/11)
%  costs = clf.Cprime' * margin_vec0;  
  costs = clf.Cprime * margin_vec0;  
  [min_pos_costs , ~] = min(costs(2:end,:));
  traces0(i,:) = -(min_pos_costs - costs(1,:));

  if use_trees
    ids1 = forestInds(X1,clf.thrs(:,i),clf.fids(:,i),clf.child(:,i), 1e5);           
    for j=1:n1
      z = clf.hs(ids1(j),i);
      margin_vec1(:,j) = margin_vec1(:,j) + ...
                        (clf.wl_weights(i).*clf.Y(:, z));
    end
  else % general case with other weak learners.
    % z is a row vector with the labels
    z  = clf.classify_weak_learner(clf.WEAK_LEARNERS{i}, X1');     
    for j=1:n1
      margin_vec1(:,j) = margin_vec1(:,j) + (clf.WEIGHTS(i).*clf.Y(:, z(j)));
    end      
  end
  % WARNING: Change to accomodate with theory (2016/11)
%  costs = clf.Cprime' * margin_vec1;    
  costs = clf.Cprime * margin_vec1;    
  [min_pos_costs , ~] = min(costs(2:end,:));
  traces1(i,:) = -(min_pos_costs - costs(1,:));
end

% 2. plot the min and max trace at last weak learner of positives and 
%    negatives data
[~ , min_neg_trace_index] = min(traces0(nWeaks,:));
[~ , max_neg_trace_index] = max(traces0(nWeaks,:));

[~ , min_pos_trace_index] = min(traces1(nWeaks,:));
[~ , max_pos_trace_index] = max(traces1(nWeaks,:));

h1 = figure;
plot(traces1(:,min_pos_trace_index), 'g-', 'LineWidth', 1.5);
hold on;
plot(traces1(:,max_pos_trace_index), 'g-', 'LineWidth', 1.5);
plot(traces0(:,min_neg_trace_index), 'r-', 'LineWidth', 1.5);
plot(traces0(:,max_neg_trace_index), 'r-', 'LineWidth', 1.5);

% 3. Compute the optimal threshold (for same success rate but with
% less average weak-learners evaluated per feature vector). For doing
% so, we find the trace of the last positive example that went over 
% 0 at the last badacost weak-learner.
indices       = find(traces1(nWeaks,:) > 0); 
[~, index] = min(traces1(nWeaks, indices));
index = indices(index);

figure(h1);
hold on; 
plot(traces1(:,index), 'b-', 'LineWidth', 2);

min_last_traces1_pos = min(traces1(:,index));
thr = min_last_traces1_pos;

hold on; 
plot([1 nWeaks], [min_last_traces1_pos min_last_traces1_pos], 'k-', 'LineWidth', 2);
plot([1 nWeaks], [thr thr], 'm-', 'LineWidth', 2);
hold off;
 
