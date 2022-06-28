function plotSelectedFeaturesMap(exp_name, dataOutputDir, featMap, featChnMaps, nFilters)

% figure; 
% imagesc(featMap); 
% title('Full Feature Map');
% saveas(gcf, [exp_name '_FeatMap.eps'], 'epsc');
% saveas(gcf, [exp_name '_FeatMap.png'], 'png');
% 
% for f=1:nFilters
%   figure; 
%   for i=1:length(featChnMaps)/nFilters
%     subplot(4,3,i);
%     imagesc(featChnMaps{i*f}); 
%     title(sprintf('Feature %d', i));
%   end
%   saveas(gcf,sprintf([exp_name '_PerFeatMap_%d.eps'], f), 'epsc');
%   saveas(gcf,sprintf([exp_name '_PerFeatMap_%d.png'], f), 'png');
% end

figure;
imagesc(featMap);
colormap('jet');
title('Full Feature Map');
saveas(gcf, fullfile(dataOutputDir, [exp_name '_FeatMap.eps']), 'epsc');
saveas(gcf, fullfile(dataOutputDir,[exp_name '_FeatMap.png']), 'png');
img = featMap;
img = (img - min(min(img))) ./ (max(max(img)) - min(min(img)));
ime = imresize(img, 100., 'nearest');
img = ind2rgb(gray2ind(img, 255), jet(255));
imwrite(img, fullfile(dataOutputDir, [exp_name '_FeatMap_FULL_IMG.png']), 'png');

figure;
NUM_CHANNELS = 10;
NUM_FILTERS = length(featChnMaps)/NUM_CHANNELS;
images = zeros(size(featMap,1), size(featMap,2)*NUM_CHANNELS);
for i=1:NUM_CHANNELS
  index = 1+(i-1)*size(featMap,2);
  for j=1:NUM_FILTERS
    images(:, index:index+size(featMap,2)-1) = images(:, index:index+size(featMap,2)-1) + featChnMaps{i+(j-1)*NUM_CHANNELS};
  end
end
images = (images - min(min(images))) ./ (max(max(images)) - min(min(images)));
images2 = imresize(images, 20., 'nearest');
images2 = ind2rgb(gray2ind(images2, 255), jet(255));
imshow(images2, 'Border', 'tight');
saveas(gcf, fullfile(dataOutputDir, [exp_name '_PerFeatMap.eps']), 'epsc');
saveas(gcf, fullfile(dataOutputDir, [exp_name '_PerFeatMap.png']), 'png');

for i=1:NUM_CHANNELS
  index = 1+(i-1)*size(featMap,2);
  img = images(:, index:index+size(featMap,2)-1);
  img = imresize(img, 10., 'nearest');
  img = ind2rgb(gray2ind(img, 255), jet(255));
  imwrite(img, fullfile(dataOutputDir, sprintf([exp_name '_PerFeatMap_%d.png'], i)), 'png');
end

figure;
c = 1;
for i=1:NUM_CHANNELS
  for j=1:NUM_FILTERS
    img = featChnMaps{c};
    img = (img - min(min(img))) ./ (max(max(img)) - min(min(img)));
    img = imresize(img, 10., 'nearest');
    img = ind2rgb(gray2ind(img, 255), jet(255));
    imwrite(img, fullfile(dataOutputDir, sprintf([exp_name '_PerFeatMap_filter_%d_channel_%d.png'], i, j)), 'png');
    imshow(img);
    %pause;
    c = c + 1;
  end
end
