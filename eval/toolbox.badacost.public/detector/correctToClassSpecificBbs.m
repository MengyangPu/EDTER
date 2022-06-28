function dt1 = correctToClassSpecificBbs(dt, aRatio, fixedWidth, keepAll)
% 
% We assume that the background class has index 1. So we have to
% remove 1 to the positive class index to get the correct median 
% aspect ratio. Therefore, if the positive class label is i, we get
% its aspect ratio for its BBoxes as aRatio(i-1).
%

if isempty(dt)
  dt1 = dt; 
  return;
end;

if nargin < 4
  keepAll = 0;
end

if nargin == 2
  fixedWidth = 0; % In this case we keep the h fixed and modify w
end

if fixedWidth
  squarify_param = 2; % use original w, alter h
else 
  squarify_param = 3; % use original h, alter w
end

maxIndex = 5;

if iscell(dt)
  if keepAll
    maxIndex = size(dt{1},2);
  end
  dt1 = cell(length(dt),1);
  for i=1:length(dt)
    if isempty(dt{i})
      continue;
    end
    for j=1:length(aRatio)
      indices = (dt{i}(:,6) == (j+1));
      dt{i}(indices,:)=bbApply('squarify', dt{i}(indices,:), squarify_param, aRatio(j));
    end
    dt1{i} = dt{i}(:,1:maxIndex);
  end
else 
  if keepAll
    maxIndex = size(dt,2);
  end
  for i=1:length(aRatio)
    indices = (dt(:,6) == (i+1));
    dt(indices,:)=bbApply('squarify', dt(indices,:), squarify_param, aRatio(i));
  end
  dt1 = dt(:,1:maxIndex);
end
