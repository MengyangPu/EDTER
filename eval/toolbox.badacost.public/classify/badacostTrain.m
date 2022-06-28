function [classfr, cost_curve] = badacostTrain( x, y, C, opt )
%--------------------------------------------------------------------------
% function [classfr, cost_curve] = badacostTrain( x, y, C, opt)
%
% This function performs Multiclass Bosting ADApted for Costs (badacost) 
% algorithm. The weak learner has to be provided in the train_weak_learner 
%    function. 
%
% Input:
%   x, Pxn matrix (P is the number of features per observation and n 
%      is the number of observations).
%   y, 1xn the corresponding class label of each of the n observations.
%   C, cost matrix of the problem. C(i,j) is the cost of estimate class
%        label j whereas the ground thruth class is i.
%   opt, a structure that controls the behavior of the classifier
%        opt.num_iterations is the number of weak classifiers to use.
%        opt.train_weak_learner is the function to call for the weak classifier. 
%        opt.classify_weak_learner is the function to call to classify with 
%                                   a trained weak classifier. 
%        opt.learning_rate  should be between 0 and 1, but better to be <=
%                           0.1. It is a factor that multiplies alpha (the
%                           weight of the weak learner)
%        opt.stop_with_negative_wl_weight If 1 stop at first negative WL
%                                         weight.
%
% Output:
%  classfr, a structure that is used in classify_adaboost.m
%     classfr.WEAK_LEARNERS -- A cell array with the trained weak classifiers
%     classfr.WEIGHTS       -- A column vector with the per weak classifier weight
%     classfr.classify_weak_learner -- a function to be used in
%                                      badacostApply
%
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
% Author: Antonio Baldera, modified by Jose M. Buenaposada

if length(y) ~= size(x,2) 
  error('Number of observations for x_train should be the same than for y_train');
end

if ~isfield(opt, 'learning_rate')
  opt.learning_rate = 0.1; % Shrinkage
end

if ~isfield(opt, 'stop_with_negative_wl_weight')
  opt.stop_with_negative_wl_weight = 1; % Stop with negative weight always
end

y= y(:)';

%MIN_COST_INC                     = 10^(-10);
n                             = size(x,2); % Number of samples
num_classes                   = length(unique(y));
num_features                  = size(x,1);
opt.num_classes               = num_classes;
classfr.num_classes           = num_classes;
classfr.classify_weak_learner = opt.classify_weak_learner;

if (nargout > 1) 
  cost_curve = zeros(opt.num_iterations, 1);
  margin_vec = zeros(classfr.num_classes, n);
end

% Initialise sample weights
SAMPLE_WEIGHTS                 = (1/n).*ones(n, 1);

% Margin vectors in matrix form.
Y                 = (-1/(num_classes-1))*ones(num_classes,num_classes) + ...
                        (num_classes/(num_classes-1))*eye(num_classes);

% Reduce the numeric range of the cost matrix in order to avoid numerical
% problems with the exp funcions. This transformation does not change the 
% estimated class boundaries.
if max(max(C)) > 1
  C =C./max(max(C));
end

% Cprime is an special matrix used in the BAdaCost loss function.
Cprime         = C - diag(sum(C, 2));
classfr.Cprime = Cprime;
C2             = Cprime * Y;

% Preparing boosting loop
msg='Training BadaCost: nWeak=%3i nFtrs=%i\n';
if(opt.verbose), fprintf(msg,opt.num_iterations,num_features); start=clock; end

classfr.SAMPLE_WEIGHTS      = cell(1, opt.num_iterations);
classfr.WEIGHTS             = zeros(opt.num_iterations,1);
classfr.WEAK_LEARNERS       = cell(1, opt.num_iterations);
classfr.Cprime              = Cprime;
classfr.Y                   = Y;
if isfield(opt, 'use_variable_depth') % We have tree WL
  if opt.use_variable_depth 
    depth = opt.minDepth;
  end
end
for i=1:opt.num_iterations % opt.num_iterations is the number of weak classifiers to retain
    
  alpha     = 1.0;
  Cexp      = translate_to_cost_matrix(C2, alpha);
  
  % Quantize data before badacost ...
  if isfield(opt, 'use_variable_depth') % We have tree WL
    if opt.use_variable_depth 
  %    depth = opt.minDepth-1;
      train_wl = 1;
      while (train_wl)
  %      depth = depth + 1;                                              
        opt2 = opt; 
        opt2.maxDepth = depth;
        if opt.USE_QUANTIZED
          [WL, pred_wl, data] = opt.train_weak_learner(x, y, SAMPLE_WEIGHTS, Cexp, opt2);
          opt.data        = data; % The weak learner trainin quantize data first time.
        else
          [WL, pred_wl] = opt.train_weak_learner(x, y, SAMPLE_WEIGHTS, Cexp, opt2);
        end
        alpha           = compute_weak_learner_weight(C2, ...
                                                      SAMPLE_WEIGHTS(:), ...
                                                      pred_wl, y);                                           
        train_wl = (alpha < 0) && (depth < opt.maxDepth);                                                
        if train_wl
          depth = depth + 1;                                              
        end
      end
    else
      depth = opt.maxDepth;
      if opt.USE_QUANTIZED
        [WL, pred_wl, data] = opt.train_weak_learner(x, y, SAMPLE_WEIGHTS, Cexp, opt);
        opt.data        = data;
      else
        [WL, pred_wl] = opt.train_weak_learner(x, y, SAMPLE_WEIGHTS, Cexp, opt);
      end
      alpha           = compute_weak_learner_weight(C2, ...
                                                    SAMPLE_WEIGHTS(:), ...
                                                    pred_wl, y);                                           
    end
  else % Any other weak learner.
      [WL, pred_wl] = opt.train_weak_learner(x, y, SAMPLE_WEIGHTS, Cexp, opt);
      alpha         = compute_weak_learner_weight(C2, ...
                                                  SAMPLE_WEIGHTS(:), ...
                                                  pred_wl, y);                                             
  end
  
  if (alpha <= 0)
    fprintf(1, 'Best weak learner has a non-positive weight: %8.3e\n', alpha);
    if opt.stop_with_negative_wl_weight 
      break;
    end;
  end              
  classfr.WEAK_LEARNERS{i} = WL;                                           

  % Smoothing the learning (Shrinkage)
  alpha = opt.learning_rate * alpha;
    
  % update sample weights: only the right classified samples changes the
  % weight.
  for j=1:size(x, 2)
    exp_j                   = C2(y(j), pred_wl(j)); 
    SAMPLE_WEIGHTS(j)       = SAMPLE_WEIGHTS(j) * exp(alpha*exp_j);
  end
  
  % sum to 1 weights normalisation
  SAMPLE_WEIGHTS            = SAMPLE_WEIGHTS./sum(SAMPLE_WEIGHTS(:));
  
  classfr.WEIGHTS(i)        = alpha;
  classfr.SAMPLE_WEIGHTS{i} = SAMPLE_WEIGHTS(:);   

  % Average cost for the strong classifier so far.
  if (nargout > 1) 
    for j=1:n
      margin_vec(:,j) = margin_vec(:,j) + ...
                        (classfr.WEIGHTS(i).*Y(:, pred_wl(j)));
    end
    
    % WARNING: Change to accomodate with theory (2016/11)
%    [~, predicted] = min(Cprime' * margin_vec);
    [~, predicted] = min(Cprime * margin_vec);
    cost_curve(i) = compute_strong_learner_cost(predicted, y, C2);
  end
  wl_cost = compute_strong_learner_cost(pred_wl, y, C2);
  if isfield(opt, 'use_variable_depth') % We have tree WL
    msg=' i=%4i alpha=%.3f depth=%d w.l.cost=%.3f s.l.cost=%.2e\n';
    if(mod(i,opt.verbose)==0), fprintf(msg,i,alpha,depth,wl_cost,cost_curve(i)); end
  else
    msg=' i=%4i alpha=%.3f w.l.cost=%.3f s.l.cost=%.2e\n';
    if(mod(i,opt.verbose)==0), fprintf(msg,i,alpha,wl_cost,cost_curve(i)); end
  end
end

% Remove un-trained weak learners (if stopped earlier because of alpha<0).
if opt.stop_with_negative_wl_weight && (alpha <= 0)
  % If stopped because of alpha being negative
  classfr.WEIGHTS = classfr.WEIGHTS(1:i-1);
  classfr.SAMPLE_WEIGHTS = classfr.SAMPLE_WEIGHTS(1:i-1);
  classfr.WEAK_LEARNERS = classfr.WEAK_LEARNERS(1:i-1);
  if (nargout > 2) 
     cost_curve = cost_curve(1:i-1);
  end
end

msg='Done training s.l.cost=%.4f (t=%.1fs).\n';
if(opt.verbose), 
  fprintf(msg,cost_curve(end),etime(clock,start)); end

end

% ------------------------------------------------------------------------
function cost = compute_strong_learner_cost(prediction, y, C2)
  cost = 0.0;
  for i=1:length(prediction)
    cost = cost + exp(C2(y(i), prediction(i)));
  end
  cost = cost / length(prediction);
end

% ------------------------------------------------------------------------
%function cost = compute_weak_learner_cost(prediction, y, alpha, C2, SAMPLE_WEIGHTS)
%  cost = 0;
%  for i=1:length(prediction)
%    cost = cost + SAMPLE_WEIGHTS(i) * exp(alpha*C2(y(i), prediction(i)));
%  end
%end

%% ------------------------------------------------------------------------
function Cexp = translate_to_cost_matrix(C2, alpha)
  K    = size(C2,1);
  Cexp = exp(alpha * C2);
  for j = 1:K 
    Cexp(j,:) = Cexp(j,:) - Cexp(j,j)*ones(1, K);
  end
end

%% ------------------------------------------------------------------------
function alpha = compute_weak_learner_weight(Cprime, W, pred, y)
% Computes  Weak Learner weight (\alpha) in order to minimize the 
% cost sensitive loss function.
%
% Input:
% - Cprime : Problem cost function with diagonals elements being minus sum of row costs.
% - W      : Sample weights to iteration t-1 in CostSAMME.
% - pred   : column vector with the per-sample label prediction. Values in {1,2,...,K}.
% - y      : column vector with the per-sample ground thruth labels. Values in {1,2,...,K}.

K          = size(Cprime,1);
WeightsSum = zeros(K,K);
for i = 1:K
  for j = 1:K
    predicted_as_j_being_i = ((y  == i) & (pred == j));
    WeightsSum(i,j) = W(:)'*double(predicted_as_j_being_i(:));
  end
end

alpha0  = 1.0;
options = optimset('GradObj','on');
alpha   = fminsearch( @(x) cost_sensitive_loss_function(x, Cprime, WeightsSum), alpha0, options);

end

%% ------------------------------------------------------------------------
function [func_value, deriv_value] = cost_sensitive_loss_function(alpha, Cprime, WeightsSum)
% Loss function and derivative of the loss function computation for a given
% alpha (weak learner weight in the CostSAMME algorithm).
%
% alpha      : is the weight computed for the weak learner.
% Cprime     : is the problem cost matrix with diagonal elements being minus sum of
%              row costs
% WeightsSum : WeightsSum(i,j) is the sum of the sample weights when prediction is
%              class j and  real label is i
%

K        = size(Cprime,1);

% Compute cost sensitive loss function value for a given \alpha.
func_value = 0;
for i = 1:K
  for j = 1:K
    func_value = func_value + (WeightsSum(i,j)*exp(alpha*Cprime(i,j)));
  end
end

% Compute derivative value of cost sensitive loss function for a given \alpha.
deriv_value = 0;
for i = 1:K
  for j = 1:K
    deriv_value = deriv_value + (WeightsSum(i,j)*Cprime(i,j)*exp(alpha*Cprime(i,j)));
  end
end
end














