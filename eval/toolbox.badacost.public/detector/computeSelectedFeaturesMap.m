
function [featMap, featChnMaps, nFilters] = computeSelectedFeaturesMap(detector)

%chnnSizes = opts.modelDs / opts.pPyramid.pChns.shrink;
%chnnLength = chnnSizes(1)*chnnSizes(2);
%nChns = size(X1,2)/chnnLength;
%
%for i=1:nChns
%  chnnImg = reshape(X1(1, (i-1)*chnnLength+1:(i*chnnLength)), ...
%                    chnnSizes);
%  figure; imshow(chnnImg);    
%end

wl_weights = detector.clf.wl_weights;
wl_weights = wl_weights ./ sum(wl_weights);
wl_weights = repmat(wl_weights(:)', size(detector.clf.fids,1), 1);

chnSizes  = detector.opts.modelDsPad / detector.opts.pPyramid.pChns.shrink;
nFilters = 1;
if (isfield(detector.opts, 'filters'))
  if (~isempty(detector.opts.filters))
    nFilters = size(detector.opts.filters,4);
    chnSizes = floor(chnSizes / 2);
  end
end
chnLength = chnSizes(1)*chnSizes(2);
nChns = 0;
for i=1:length(detector.info)
  nChns = nChns + detector.info(i).nChns;
end


fids = detector.clf.fids(detector.clf.fids > 0);
wl_weights = wl_weights(detector.clf.fids > 0);

% Global selected feature map
featMap = zeros(chnSizes);
for i=1:length(fids)
  id = mod(double(fids(i)-1),chnLength)+1;
  featMap(id) = featMap(id) + wl_weights(i); 
end

% Selected feature map per channel
featChnMaps = cell(nChns*nFilters, 1);
thOld = 1;
for c=1:nChns*nFilters
  th = c*chnLength;
  chnFids = fids((fids>thOld) & (fids<=th));
  chnWeights = wl_weights((fids>thOld) & (fids<=th));
  chnWeights = chnWeights ./ sum(chnWeights);
  featMapAux = zeros(chnSizes);
  for i=1:length(chnFids)
    id = mod(double(chnFids(i)-1),chnLength)+1;
    featMapAux(id) = featMapAux(id) + chnWeights(i); 
  end  
  featChnMaps{c} = featMapAux;
  thOld = th;
end