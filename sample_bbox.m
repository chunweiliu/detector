function bbox = sample_bbox(pyramid, w, h, offset)
[xmin, ymin] = meshgrid(1:offset:size(pyramid,2)-w+1, ...
                        1:offset:size(pyramid,1)-h+1);
xmin = xmin(:);
ymin = ymin(:);
xmax = xmin + w - 1;
ymax = ymin + h - 1;
bbox = [xmin, ymin, xmax, ymax];


% if sum(xmax > size(pyramid,2)) > 1 || sum(ymax > size(pyramid,1)) > 1
%     bbox = [];
% end