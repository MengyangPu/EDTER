function detector = acfTrainBadacostTrees( varargin )
% Train aggregate channel features object detector.
%
% -------------------------------------------------------------------------
%  MODIFIED BY Jose M. Buenaposada
%  NOTE: Same as P.Dollar's implementation but changing to multiclass
%  classifiers for detection. For example, the detector can be trained 
%  taking into account the car orientation. One of the classes is used as 
%  negative (background).
% -------------------------------------------------------------------------
%
% Train aggregate channel features (ACF) object detector as described in:
%  P. Doll�r, R. Appel, S. Belongie and P. Perona
%   "Fast Feature Pyramids for Object Detection", PAMI 2014.
% The ACF detector is fast (30 fps on a single core) and achieves top
% accuracy on rigid object detection. Please see acfReadme.m for details.
%
% Takes a set of parameters opts (described in detail below) and trains a
% detector from start to finish including performing multiple rounds of
% bootstrapping if need be. The return is a struct 'detector' for use with
% acfDetect.m which fully defines a sliding window detector. Training is
% fast (on the INRIA pedestrian dataset training takes ~10 minutes on a
% single core or ~3m using four cores). Taking advantage of parallel
% training requires launching matlabpool (see help for matlabpool). The
% trained detector may be altered in certain ways via acfModify(). Calling
% opts=acfTrain() returns all default options.
%
% (1) Specifying features and model: The channel features are defined by
% 'pPyramid'. See chnsCompute.m and chnsPyramid.m for more details. The
% channels may be convolved by a set 'filters' to remove local correlations
% (see our NIPS14 paper on LDCF), improving accuracy but slowing detection.
% If 'filters'=[wFilter,nFilter] these are automatically computed. The
% model dimensions ('modelDs') define the window height and width. The
% padded dimensions ('modelDsPad') define the extended region around object
% candidates that are used for classification. For example, for 100 pixel
% tall pedestrians, typically a 128 pixel tall region is used to make a
% decision. 'pNms' controls non-maximal suppression (see bbNms.m), 'stride'
% controls the window stride, and 'cascThr' and 'cascCal' are the threshold
% and calibration used for the constant soft cascades. Typically, set
% 'cascThr' to -1 and adjust 'cascCal' until the desired recall is reached
% (setting 'cascCal' shifts the final scores output by the detector by the
% given amount). Training alternates between sampling (bootstrapping) and
% training an AdaBoost classifier (clf). 'nWeak' determines the number of
% training stages and number of trees after each stage, e.g. nWeak=[32 128
% 512 2048] defines four stages with the final clf having 2048 trees.
% 'pBoost' specifies parameters for AdaBoost, and 'pBoost.pTree' are the
% decision tree parameters, see adaBoostTrain.m for details. Finally,
% 'seed' is the random seed used and makes results reproducible and 'name'
% defines the location for storing the detector and log file.
%
% (2) Specifying training data location and amount: The training data can
% take on a number of different forms. The positives can be specified using
% either a dir of pre-cropped windows ('posWinDir') or dirs of full images
% ('posImgDir') and ground truth labels ('posGtDir'). The negatives can by
% specified using a dir of pre-cropped windows ('negWinDir'), a dir of full
% images without any positives and from which negatives can be sampled
% ('negImgDir'), and finally if neither 'negWinDir' or 'negImgDir' are
% given negatives are sampled from the images in 'posImgDir' (avoiding the
% positives). For the pre-cropped windows all images must have size at
% least modelDsPad and have the object (of size exactly modelDs) centered.
% 'imreadf' can be used to specify a custom function for loading an image,
% and 'imreadp' are custom additional parameters to imreadf. When sampling
% from full images, 'pLoad' determines how the ground truth is loaded and
% converted to a set of positive bbs (see bbGt>bbLoad). 'nPos' controls the
% total number of positives to sample for training (if nPos=inf the number
% of positives is limited by the training set). 'nNeg' controls the total
% number of negatives to sample and 'nPerNeg' limits the number of
% negatives to sample per image. 'nAccNeg' controls the maximum number of
% negatives that can accumulate over multiple stages of bootstrapping.
% Define 'pJitter' to jitter the positives (see jitterImage.m) and thus
% artificially increase the number of positive training windows. Finally if
% 'winsSave' is true cropped windows are saved to disk as a mat file.
%
% USAGE
%  detector = acfTrainBadacostTrees.m ( opts )
%  opts = acfTrainTrainBadacostTrees.m ()
%
% INPUTS
%  opts       - parameters (struct or name/value pairs)
%   (1) features and model:
%   .aRatioType - ['mean'] per class computation bb aspect ratio 
%                 computation (see computePerClassAspectRatios)
%   .aRatioFixedWidth [0]
%   .pPyramid   - [{}] params for creating pyramid (see chnsPyramid)
%   .filters    - [] [wxwxnChnsxnFilter] filters or [wFilter,nFilter]
%   .modelDs    - [] model height+width without padding (eg [100 41])
%   .modelDsPad - [] model height+width with padding (eg [128 64])
%   .pNms       - [..] params for non-maximal suppression (see bbNms.m)
%   .stride     - [4] spatial stride between detection windows
%   .cascCal    - [.1] cascade calibration (affects speed/accuracy)
%   .nWeak      - [128] vector defining number weak clfs per stage
%   .pBoost     - [..] parameters for boosting (see badacostTrain.m)
%   .seed       - [0] seed for random stream (for reproducibility)
%   .name       - [''] name to prepend to clf and log filenames
%   (2) training data location and amount:
%   .posGtDir   - [''] dir containing ground truth
%   .posImgDir  - [''] dir containing full positive images
%   .negImgDir  - [''] dir containing full negative images
%   .posWinDir  - [''] dir containing cropped positive windows
%   .negWinDir  - [''] dir containing cropped negative windows
%   .imreadf    - [@imread] optional custom function for reading images
%   .imreadp    - [{}] optional custom parameters for imreadf
%   .pLoad      - [..] params for bbGt>bbLoad (see bbGt)
%   .nPos       - [inf] max number of pos windows to sample
%   .nNeg       - [5000] max number of neg windows to sample
%   .nPerNeg    - [25]  max number of neg windows to sample per image
%   .nAccNeg    - [10000] max number of neg windows to accumulate
%   .pJitter    - [{}] params for jittering pos windows (see jitterImage)
%   (3) save training results:
%   .winsSave   - [0] if true save cropped windows at each stage to disk
%   .savePath   - ['.'] path to save results in.
%
% OUTPUTS
%  detector   - trained object detector 
%   .opts       - input parameters used for model training
%   .clf        - learned boosted tree classifier (see badacostTrain)
%   .info       - info about channels (see chnsCompute.m)
%
% EXAMPLE
%
% See also acfDetectBadacostTrees, acfTestBadacostTrees, chnsCompute,
% chnsPyramid, badacostTrain, bbGt, bbNms, jitterImage
%
% Piotr's Computer Vision Matlab Toolbox      Version NEW
% Copyright 2014 Piotr Dollar.  [pdollar-at-gmail.com]
% Licensed under the Simplified BSD License [see external/bsd.txt]

% initialize opts struct
opts = initializeOpts( varargin{:} );
if(nargin==0), detector=opts; return; end

% load or initialize detector and begin logging
nm=fullfile(opts.savePath, [opts.name 'Detector.mat']); t=exist(nm,'file');
if(t), if(nargout), t=load(nm); detector=t.detector; end; return; end
t=fileparts(nm); if(~isempty(t) && ~exist(t,'dir')), mkdir(t); end
detector = struct( 'opts',opts, 'clf',[], 'info',[] );
startTrain=clock; nm=fullfile(opts.savePath, [opts.name 'Log.txt']);
if(exist(nm,'file')), diary(nm); diary('off'); delete(nm); end; diary(nm);
RandStream.setGlobalStream(RandStream('mrg32k3a','Seed',opts.seed));

% iterate bootstraping and training
for stage = 0:numel(opts.nWeak)-1
  diary('on'); fprintf([repmat('-',[1 75]) '\n']);
  fprintf('Training stage %i\n',stage); startStage=clock;
  
  % sample positives and compute info about channels
  if( stage==0 )
    [Is1,Ls1,IsOrig1] = sampleWins( detector, stage, 1 );
    t=ndims(Is1); if(t==3), t=Is1(:,:,1); else t=Is1(:,:,:,1); end
    t=chnsCompute(t,opts.pPyramid.pChns); detector.info=t.info;
  end
  
  % compute local decorrelation filters
  if( stage==0 && length(opts.filters)==2 )
    fs = opts.filters; opts.filters = [];
    X1 = chnsCompute1( IsOrig1, opts );
    fs = chnsCorrelation( X1, fs(1), fs(2) );
    opts.filters = fs; detector.opts.filters = fs;
  end
  
  % compute lambdas
  if( stage==0 && isempty(opts.pPyramid.lambdas) )
    fprintf('Computing lambdas... '); start=clock;
    ds=size(IsOrig1); ds(1:end-1)=1; IsOrig1=mat2cell2(IsOrig1,ds);
    ls=chnsScaling(opts.pPyramid.pChns,IsOrig1,0);
    ls=round(ls*10^5)/10^5; detector.opts.pPyramid.lambdas=ls;
    fprintf('done (time=%.0fs).\n',etime(clock,start));
  end
  
  % compute features for positives
  if( stage==0 )
    X1 = chnsCompute1( Is1, opts );
    X1 = reshape(X1,[],size(X1,4))';
    clear Is1 IsOrig1 ls fs ds t;
  end
  
  % sample negatives and compute features
  [Is0, ~] = sampleWins( detector, stage, 0 );
  X0 = chnsCompute1( Is0, opts ); clear Is0;
  X0 = reshape(X0,[],size(X0,4))';
  
  % accumulate negatives from previous stages
  if( stage>0 )
    n0=size(X0p,1); n1=max(opts.nNeg,opts.nAccNeg)-size(X0,1);
    if(n0>n1 && n1>0), X0p=X0p(randSample(n0,n1),:); end
    if(n0>0 && n1>0), X0=[X0p; X0]; end %#ok<AGROW>
  end; X0p=X0;
  
%   save_file = fullfile(opts.savePath, sprintf('%s_STAGE_%d_NEGATIVE_DATA.mat', opts.name, stage));
%   if ~exist(save_file, 'file')
%     save(save_file, 'X0p');
%   else
%     load(save_file);
%   end
    
  % train boosted BAdaCost clf and calibrate the cascade threshold.
  save_file_detector = fullfile(opts.savePath, sprintf('%s_STAGE_%d_Detector.mat', opts.name, stage));
  if ~exist(save_file_detector, 'file')
    detector.opts.pBoost = opts.pBoost;
    detector.opts.pBoost.nWeak = opts.nWeak(stage+1);
    [detector.clf, cost_curve] = badacostWithTreesTrain(X0, X1, Ls1, detector.opts.pBoost);   
    
    % Plot and save cost curve figure
    h = figure; 
    plot(cost_curve);
    xlabel('#weak learners');
    ylabel('cost');
    cost_curve_file = fullfile(opts.savePath, sprintf('%s_STAGE_%d_COST_CURVE', opts.name, stage));
    save([cost_curve_file '.txt'], '-ascii', 'cost_curve');
    figure(h);
    saveas(h, [cost_curve_file '.eps'], 'epsc');
%    saveas(h, [cost_curve_file '.fig'], 'fig');
   
%    [detector.opts.cascThr, h_calCascFig] = badacostCalibrateCascade(X0, X1, detector.clf, opts.cascCal);
    [detector.opts.cascThr, h_calCascFig] = badacostCalibrateCascade(X0, X1, detector.clf);
    % Watch out!!! This is faster but you can miss detections!!
%    detector.opts.cascThr=-2;

    % Compute the Training Confusion Matrix:
    xx = [X0; X1];
    yy = [ones(1,size(X0,1)), Ls1+1];
    yy_predicted = badacostWithTreesApply(xx, detector.clf);
    
    ConfMatrix = confusionmat(yy, yy_predicted);
    disp('Confusion Matrix:');
    disp(ConfMatrix);
    conf_mat_file = fullfile(opts.savePath, sprintf('%s_STAGE_%d_CONFUSION_MATRIX', opts.name, stage));
    save([conf_mat_file '.mat'], 'ConfMatrix');
    save([conf_mat_file '.txt'], '-ascii', 'ConfMatrix');
    
    % Plot cascade calibration figure 
    casc_calib_file = fullfile(opts.savePath, sprintf('%s_STAGE_%d_CASCADE_CALIBRATION', opts.name, stage));
    figure(h_calCascFig);
    saveas(h_calCascFig,[casc_calib_file '.eps'], 'epsc');
%    saveas(h_calCascFig,[casc_calib_file '.fig'], 'fig');
    
    % Save trained detector
    save(save_file_detector, 'detector');
  else
    load(save_file_detector);
    
    %% Uncomment if you want to recompute the training confusion matrix.
    % Compute the Training Confusion Matrix:
    xx = [X0; X1];
    yy = [ones(1,size(X0,1)), Ls1+1];
    yy_predicted = badacostWithTreesApply(xx, detector.clf);
    
    ConfMatrix = confusionmat(yy, yy_predicted);
    disp('Confusion Matrix:');
    disp(ConfMatrix);
    conf_mat_file = fullfile(opts.savePath, sprintf('%s_STAGE_%d_CONFUSION_MATRIX', opts.name, stage));
    save([conf_mat_file '.mat'], 'ConfMatrix');
    save([conf_mat_file '.txt'], '-ascii', 'ConfMatrix');
  end

  % update log
  fprintf('Done training stage %i (time=%.0fs).\n',...
    stage,etime(clock,startStage)); diary('off');
end

% JMBUENA: Compute the window size for every subclass of the 
% positive metaclass. We are going to keep this window size
% within the detector classifier. We will train the detector with
% fixed big window (selecting the features in this window) but we
% finally return the "per class" best fit window.
detector.clf.aRatio = computePerClassAspectRatios(detector.opts.posImgDir, ...
  detector.opts.posGtDir, detector.opts.pLoad, opts.aRatioType);
detector.clf.aRatioFixedWidth = opts.aRatioFixedWidth; 

% save detector
save(fullfile(opts.savePath, [opts.name 'Detector.mat']),'detector');

% finalize logging
diary('on'); fprintf([repmat('-',[1 75]) '\n']);
fprintf('Done training (time=%.0fs).\n',...
  etime(clock,startTrain)); diary('off');
end

function opts = initializeOpts( varargin )
% Initialize opts struct.
dfs= {'aRatioType', 'mean', ...
  'pPyramid',{}, 'filters', [], ...
  'modelDs',[100 41], 'modelDsPad',[128 64], ...
  'pNms',struct(), 'stride',4, 'cascCal',.1, ...
  'nWeak',128, 'pBoost', {}, 'seed',0, 'name','', ...,
  'aRatioFixedWidth', 0, ...
  'posGtDir','', ...
  'posImgDir','', 'negImgDir','', 'posWinDir','', 'negWinDir','', ... 
  'imreadf',@imread, 'imreadp',{}, 'pLoad',{}, 'nPos',inf, 'nNeg',5000, ...
  'nPerNeg',25, 'nAccNeg',10000, 'pJitter',{}, 'winsSave', 0, 'savePath', '.'};
opts = getPrmDflt(varargin,dfs,1);
% fill in remaining parameters
p=chnsPyramid([],opts.pPyramid); p=p.pPyramid;
p.minDs=opts.modelDs; shrink=p.pChns.shrink;
opts.modelDsPad=ceil(opts.modelDsPad/shrink)*shrink;
p.pad=ceil((opts.modelDsPad-opts.modelDs)/shrink/2)*shrink;
p=chnsPyramid([],p); p=p.pPyramid; p.complete=1;
p.pChns.complete=1; opts.pPyramid=p;
% initialize pNms, pBoost, pBoost.pTree, and pLoad
dfs={ 'type','maxg', 'overlap',.65, 'ovrDnm','min' };
opts.pNms=getPrmDflt(opts.pNms,dfs,-1);
dfs={ 'Cost',{}, 'stopAtNegWeight', 1, 'shrinkage', {}, 'resampling', {}, 'nWeak',0, ...
      'minDepth', 1, 'maxDepth', 4, 'verbose', 16, 'quantized', 0, ...
      'use_rus', 0, 'fracFtrs', 1, ...
      'variable_depth', 0};    
opts.pBoost=getPrmDflt(opts.pBoost,dfs,1);
%dfs={'nBins',256,'maxDepth',2,'minWeight',.01,'fracFtrs',1,'nThreads',1e5};
%opts.pBoost.pTree=getPrmDflt(opts.pBoost.pTree,dfs,1);
opts.pLoad=getPrmDflt(opts.pLoad,{'squarify',{0,1}},-1);
opts.pLoad.squarify{2}=opts.modelDs(2)/opts.modelDs(1);
end

function [Is,Ls,IsOrig] = sampleWins( detector, stage, positive )
% Load or sample windows for training detector.
opts=detector.opts; start=clock;
if( positive ), n=opts.nPos; else n=opts.nNeg; end
% JMBUENA FIXME! How to deal with positive subclass labels from cropped images?
if( positive ), crDir=opts.posWinDir; else crDir=opts.negWinDir; end
if( exist(crDir,'dir') && stage==0 )
  % if window directory is specified simply load windows
  fs=bbGt('getFiles',{crDir}); nImg=length(fs); assert(nImg>0);
  if(nImg>n), fs=fs(:,randSample(nImg,n)); else n=nImg; end
  for i=1:n, fs{i}=[{opts.imreadf},fs(i),opts.imreadp]; end
  Is=cell(1,n); parfor i=1:n, Is{i}=feval(fs{i}{:}); end
else
  % sample windows from full images using sampleWins1()
  hasGt=positive||isempty(opts.negImgDir); fs={opts.negImgDir};
  if(hasGt), fs={opts.posImgDir,opts.posGtDir}; end
  fs=bbGt('getFiles',fs); nImg=size(fs,2); assert(nImg>0);
  if(~isinf(n)), fs=fs(:,randperm(nImg)); end; 
  Is=cell(nImg*1000,1); Ls=cell(nImg*1000,1);
  diary('off'); tid=ticStatus('Sampling windows',1,30); k=0; i=0; batch=64; 
  while( i<nImg && k<n )
    batch=min(batch,nImg-i); Is1=cell(1,batch); lbls=cell(1, batch);
    parfor j=1:batch, ij=i+j; 
      I = feval(opts.imreadf,fs{1,ij},opts.imreadp{:}); %#ok<PFBNS>
      gt=[]; objs=[]; if(hasGt), [objs,gt]=bbGt('bbLoad',fs{2,ij},opts.pLoad); end
      [Is1{j}, lbls{j}]  = sampleWins1( I, objs, gt, detector, stage, positive );
    end
    Is1=[Is1{:}]; k1=length(Is1); Is(k+1:k+k1)=Is1; 
    lbls=[lbls{:}]; Ls(k+1:k+k1)=lbls; k=k+k1;
    if(k>n), inds=randSample(k,n); Is=Is(inds); Ls=Ls(inds); k=n; end
    i=i+batch; tocStatus(tid,max(i/nImg,k/n));
  end
  Is=Is(1:k); Ls=Ls(1:k); diary('on'); 
  fprintf('Sampled %i windows from %i images.\n',k,i);
end
% optionally jitter positive windows
if(length(Is)<2), Is={}; return; end
nd=ndims(Is{1})+1; Is=cat(nd,Is{:}); Ls=cat(2,Ls{:}); IsOrig=Is;
if( positive && isstruct(opts.pJitter) )
  opts.pJitter.hasChn=(nd==4); Is=jitterImage(Is,opts.pJitter);
  ds=size(Is); 
  Ls=repmat(Ls, [ds(nd),1]);
  Ls=reshape(Ls,[1,ds(nd)*length(Ls)]);
  ds(nd)=ds(nd)*ds(nd+1); Is=reshape(Is,ds(1:nd));
end
% make sure dims are divisible by shrink and not smaller than modelDsPad
ds=size(Is); cr=rem(ds(1:2),opts.pPyramid.pChns.shrink); s=floor(cr/2)+1;
e=ceil(cr/2); Is=Is(s(1):end-e(1),s(2):end-e(2),:,:); ds=size(Is);
if(any(ds(1:2)<opts.modelDsPad)), error('Windows too small.'); end
% optionally save windows to disk and update log
nm=fullfile(opts.savePath, [opts.name 'Is' int2str(positive) 'Stage' int2str(stage)]);
if( opts.winsSave ), save(nm,'Is','-v7.3'); end
fprintf('Done sampling windows (time=%.0fs).\n',etime(clock,start));
diary('off'); diary('on');
end

function [Is,lbls] = sampleWins1( I, objs, gt, detector, stage, positive )
% Sample windows from I given its ground truth gt.
opts=detector.opts; shrink=opts.pPyramid.pChns.shrink;
modelDs=opts.modelDs; modelDsPad=opts.modelDsPad;
if( positive ), bbs=gt; bbs=bbs(gt(:,5)==0,:); objs=objs(gt(:,5)==0,:); else
  if( stage==0 )
    % generate candidate bounding boxes in a grid
    [h,w,~]=size(I); h1=modelDs(1); w1=modelDs(2);
    n=opts.nPerNeg; ny=sqrt(n*h/w); nx=n/ny; ny=ceil(ny); nx=ceil(nx);
    [xs,ys]=meshgrid(linspace(1,w-w1,nx),linspace(1,h-h1,ny));
    bbs=[xs(:) ys(:)]; bbs(:,3)=w1; bbs(:,4)=h1; bbs=bbs(1:n,:);
  else
    % run detector to generate candidate bounding boxes
    bbs=acfDetectBadacost(I,detector); [~,ord]=sort(bbs(:,5),'descend');
    bbs=bbs(ord(1:min(end,opts.nPerNeg)),1:4);
  end
  if( ~isempty(gt) )
    % discard any candidate negative bb that matches the gt
    n=size(bbs,1); keep=false(1,n);
    for i=1:n, keep(i)=all(bbGt('compOas',bbs(i,:),gt,gt(:,5))<.1); end
    bbs=bbs(keep,:);
    objs=[];
  end
end
% grow bbs to a large padded size and finally crop windows
modelDsBig=max(8*shrink,modelDsPad)+max(2,ceil(64/shrink))*shrink;
r=modelDs(2)/modelDs(1); assert(all(abs(bbs(:,3)./bbs(:,4)-r)<1e-5));
r=modelDsBig./modelDs; bbs=bbApply('resize',bbs,r(1),r(2));
Is=bbApply('crop',I,bbs,'replicate',modelDsBig([2 1]));
if isempty(Is); lbls=Is; else
  lbls=cell(1,length(Is));
  if( positive ) 
    lbls(:)={objs(:).subclass};
  else
    lbls(:)={-1}; 
  end
end
end

function chns = chnsCompute1( Is, opts )
% Compute single scale channels of dimensions modelDsPad.
if(isempty(Is)), chns=[]; return; end
fprintf('Extracting features... '); start=clock; fs=opts.filters;
pChns=opts.pPyramid.pChns; smooth=opts.pPyramid.smooth;
dsTar=opts.modelDsPad/pChns.shrink; ds=size(Is); ds(1:end-1)=1;
Is=squeeze(mat2cell2(Is,ds)); n=length(Is); chns=cell(1,n);
parfor i=1:n
  C=chnsCompute(Is{i},pChns); C=convTri(cat(3,C.data{:}),smooth);
  if(~isempty(fs)), C=repmat(C,[1 1 size(fs,4)]);
    for j=1:size(C,3), C(:,:,j)=conv2(C(:,:,j),fs(:,:,j),'same'); end; end
  if(~isempty(fs)), C=imResample(C,.5); shr=2; else shr=1; end
  ds=size(C); cr=ds(1:2)-dsTar/shr; s=floor(cr/2)+1; e=ceil(cr/2);
  C=C(s(1):end-e(1),s(2):end-e(2),:); chns{i}=C;
end; chns=cat(4,chns{:});
fprintf('done (time=%.0fs).\n',etime(clock,start));
end

function filters = chnsCorrelation( chns, wFilter, nFilter )
% Compute filters capturing local correlations for each channel.
fprintf('Computing correlations... '); start=clock;
[~,~,m,n]=size(chns); w=wFilter; wp=w*2-1;
filters=zeros(w,w,m,nFilter,'single');
for i=1:m
  % compute local auto-scorrelation using Wiener-Khinchin theorem
  mus=squeeze(mean(mean(chns(:,:,i,:)))); sig=cell(1,n);
  parfor j=1:n
    T=fftshift(ifft2(abs(fft2(chns(:,:,i,j)-mean(mus))).^2));
    sig{j}=T(floor(end/2)+1-w+(1:wp),floor(end/2)+1-w+(1:wp));
  end
  sig=double(mean(cat(4,sig{mus>1/50}),4));
  sig=reshape(full(convmtx2(sig,w,w)),wp+w-1,wp+w-1,[]);
  sig=reshape(sig(w:wp,w:wp,:),w^2,w^2); sig=(sig+sig')/2;
  % compute filters for each channel from sig (sorted by eigenvalue)
  [fs,D]=eig(sig); fs=reshape(fs,w,w,[]);
  [~,ord]=sort(diag(D),'descend');
  fs=flipdim(flipdim(fs,1),2); %#ok<DFLIPDIM>
  filters(:,:,i,:)=fs(:,:,ord(1:nFilter));
end
fprintf('done (time=%.0fs).\n',etime(clock,start));
end
