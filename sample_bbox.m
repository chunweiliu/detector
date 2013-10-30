function bbox = sample_bbox(pyramid, w, h, offset)
[minx, miny] = meshgrid(1:offset:size(pyramid,2)-w+1, ...
                        1:offset:size(pyramid,1)-h+1);
minx = minx(:);
miny = miny(:);
maxx = minx + w - 1;
maxy = miny + h - 1;
bbox = [minx, miny, maxx, maxy];
