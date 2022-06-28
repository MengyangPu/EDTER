function aRatio = computePerClassAspectRatios(posImgDir, posGtDir, pLoad, aRatioType, plot_orientation_dist)

if nargin<5
  plot_orientation_dist = 0;
end
    
% JMBUENA!!
% Compute the average window size for every subclass of the 
% positive metaclass. We are going to keep this average window size
% within the detector classifier. We will train the detector with
% fixed big window (selecting the features in this window) but we
% finally return the "per class" best fit window.
tid  = ticStatus('Per class aspect ratio',1,30); 
fs={posImgDir,posGtDir};
fs=bbGt('getFiles',fs); nImg=size(fs,2); assert(nImg>0);
gt   = cell(nImg,1);
lbls = cell(nImg,1);
pLoad2=pLoad;
%if iscell(pLoad)
%  pLoad2 = pLoad{:};
%end

use_mean_aratio = 0;
if iscell(pLoad)
  index = find(strcmp(pLoad2, 'squarify'));
  if ~isempty(index)
    pLoad2{index+1} = {};
  end
elseif isstruct(pLoad2)
  if isfield(pLoad2, 'squarify')
    pLoad2.squarify = {};
  end
end

for i=1:nImg
  [objs_,gt_] = bbGt('bbLoad',fs{2,i},pLoad2);
  indices = gt_(:,5)==0;
  if (sum(indices)>0)
    lbls{i} = [objs_(indices).subclass]';
    gt{i}   = gt_(indices,:);
  end
  tocStatus(tid,i/nImg);
end
gt   = cell2mat(gt);
lbls = cell2mat(lbls);
num_pos_classes = length(unique(lbls));
ratios = gt(:,3)./gt(:,4);
aRatio = zeros(num_pos_classes, 1);
for i=1:num_pos_classes
  if (strcmp(aRatioType, 'mean') == 1) 
    aRatio(i) = mean(ratios(lbls==i));    
  else %(strcmp(aRatioType, 'median') == 1) 
    aRatio(i) = median(ratios(lbls==i));    
  end
%  aRatio(i) = prctile(ratios(lbls==i),60);
%   a_ratio_class_std(i) = std(ratios(lbls==i));    
end

if plot_orientation_dist
  % Plot orientation distributios
  for i=1:num_pos_classes
    figure; 
    haxes = axes;
    data_i = ratios(lbls == i);
    [h, x] = hist(data_i,50); 
  %  hf = h ./ length(data_i);
    stairs(x, h, 'g-'); 
    axis([0 4 0 max(h)]);  
    mean_i = mean(data_i);
    median_i = median(data_i);
    hold on;
    line([mean_i, mean_i], get(haxes, 'YLim'), 'Color', [1 0 0]);
    line([median_i, median_i], get(haxes, 'YLim'), 'Color', [1 0 1]);      
    hold off;
    title(sprintf('orientation %d', i)); 
  end;
end;