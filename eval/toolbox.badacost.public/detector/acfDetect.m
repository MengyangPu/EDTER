function bbs = acfDetect( I, detector, fileName )
% Run aggregate channel features object detector on given image(s).
%
% The input 'I' can either be a single image (or filename) or a cell array
% of images (or filenames). In the first case, the return is a set of bbs
% where each row has the format [x y w h score] and score is the confidence
% of detection. If the input is a cell array, the output is a cell array
% where each element is a set of bbs in the form above (in this case a
% parfor loop is used to speed execution). If 'fileName' is specified, the
% bbs are saved to a comma separated text file and the output is set to
% bbs=1. If saving detections for multiple images the output is stored in
% the format [imgId x y w h score] and imgId is a one-indexed image id.
%
% A cell of detectors trained with the same channels can be specified,
% detected bbs from each detector are concatenated. If using multiple
% detectors and opts.pNms.separate=1 then each bb has a sixth element
% bbType=j, where j is the j-th detector, see bbNms.m for details.
%
% USAGE
%  bbs = acfDetect( I, detector, [fileName] )
%
% INPUTS
%  I          - input image(s) of filename(s) of input image(s)
%  detector   - detector(s) trained via acfTrain
%  fileName   - [] target filename (if specified return is 1)
%
% OUTPUTS
%  bbs        - [nx5] array of bounding boxes or cell array of bbs
%
% EXAMPLE
%
% See also acfTrain, acfModify, bbGt>loadAll, bbNms
%
% Piotr's Computer Vision Matlab Toolbox      Version 3.40
% Copyright 2014 Piotr Dollar.  [pdollar-at-gmail.com]
% Licensed under the Simplified BSD License [see external/bsd.txt]

% run detector on every image
if(nargin<3), fileName=''; end; multiple=iscell(I);
if(~isempty(fileName) && exist(fileName,'file')), bbs=1; return; end
if(~multiple), bbs=acfDetectImg(I,detector); else
  n=length(I); bbs=cell(n,1);
  for i=1:n, disp(I{i}), bbs{i}=acfDetectImg(I{i},detector); end
  %parfor i=1:n, disp(I{i}), bbs{i}=acfDetectImg(I{i},detector); end
end

% write results to disk if fileName specified
if(isempty(fileName)), return; end
d=fileparts(fileName); if(~isempty(d)&&~exist(d,'dir')), mkdir(d); end
if( multiple ) % add image index to each bb and flatten result
  for i=1:n, bbs{i}=[ones(size(bbs{i},1),1)*i bbs{i}]; end
  bbs=cell2mat(bbs);
end
dlmwrite(fileName,bbs); bbs=1;

end

function bbs = acfDetectImg( I, detector )

% JMBUENA: Workaround for BUG in gradientHist with images bigger than 4900 px in either
% dimension
sz = size(I);
old_w = sz(2);
old_h = sz(1);
big_image = 0; 
%big_size = 4900*3900*3; % 4900*3900 pixels x 3 channels ~ 54 MBytes (1 byte/channel)
%max_size = 4900;
big_size = 4100*3500*3; % 4100*3500 pixels x 3 channels ~ 41 MBytes (1 byte/channel)
max_size = 3500;
if  (prod(sz) > big_size)
  big_image = 1; 
  if (old_h == max(sz))
    new_w = round(max_size * (old_w/old_h));
    new_h = max_size;
  else
    new_h = round(max_size * (old_h/old_w));
    new_w = max_size;
  end
%  I = imResampleMex(I, new_h, new_w, 1);  
  I = imResample(I, [new_h, new_w]);
end

% Run trained sliding-window object detector on given image.
Ds=detector; if(~iscell(Ds)), Ds={Ds}; end; nDs=length(Ds);
opts=Ds{1}.opts; pPyramid=opts.pPyramid; pNms=opts.pNms;
imreadf=opts.imreadf; imreadp=opts.imreadp;
shrink=pPyramid.pChns.shrink; pad=pPyramid.pad;
separate=nDs>1 && isfield(pNms,'separate') && pNms.separate;
% read image and compute features (including optionally applying filters)
if(all(ischar(I))), I=feval(imreadf,I,imreadp{:}); end
P=chnsPyramid(I,pPyramid); bbs=cell(P.nScales,nDs);
if(isfield(opts,'filters') && ~isempty(opts.filters)), shrink=shrink*2;
  for i=1:P.nScales, fs=opts.filters; C=repmat(P.data{i},[1 1 size(fs,4)]);
    for j=1:size(C,3), C(:,:,j)=conv2(C(:,:,j),fs(:,:,j),'same'); end
    P.data{i}=imResample(C,.5);
  end
end
% apply sliding window classifiers
for i=1:P.nScales
  for j=1:nDs, opts=Ds{j}.opts;
    modelDsPad=opts.modelDsPad; modelDs=opts.modelDs;
    bb = acfDetect1(P.data{i},Ds{j}.clf,shrink,...
      modelDsPad(1),modelDsPad(2),opts.stride,opts.cascThr);
    shift=(modelDsPad-modelDs)/2-pad;
    bb(:,1)=(bb(:,1)+shift(2))/P.scaleshw(i,2);
    bb(:,2)=(bb(:,2)+shift(1))/P.scaleshw(i,1);
    bb(:,3)=modelDs(2)/P.scales(i);
    bb(:,4)=modelDs(1)/P.scales(i);
    if(separate), bb(:,6)=j; end; bbs{i,j}=bb;
  end
end; bbs=cat(1,bbs{:});
% % JMBUENA: For AFW/AFLW
% bbs = bbApply('squarify', bbs, 2, 1./1.18);
if(~isempty(pNms)), bbs=bbNms(bbs,pNms); end

% JMBUENA: AVOID BUG of gradientHist with too big images.
if (big_image)
  bbs(:,1) = round(bbs(:,1) * (old_w/new_w));
  bbs(:,2) = round(bbs(:,2) * (old_h/new_h));
  bbs(:,3) = round(bbs(:,3) * (old_w/new_w));
  bbs(:,4) = round(bbs(:,4) * (old_h/new_h));
end
end
