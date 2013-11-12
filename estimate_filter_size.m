function [w,h] = estimate_filter_size(cls, year, cellsize)

% Get data from a class
[pos, neg, posim] = pascal_data(cls, year);

meanbox = zeros(1,4);
for i=1:length(pos)
    meanbox = meanbox + pos(i).boxes;
end
meanbox = meanbox / length(pos);

% Due to multiple pyramids, it is ok to reduce filter size by 2
scale_factor = 2; % bigger, the window get smaller
w=ceil((meanbox(3)-meanbox(1))/(scale_factor*cellsize));
h=ceil((meanbox(4)-meanbox(2))/(scale_factor*cellsize));