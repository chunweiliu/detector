function plot_boxes(im, boxes, color)
if nargin < 3
    color = 'g';
end

figure(1)
imshow(im)
hold on

for i=1:size(boxes,1)
    rectangle('Position', [boxes(i,1) boxes(i,2) boxes(i,3)-boxes(i,1) boxes(i,4)-boxes(i,2)],...
              'EdgeColor', color);
end
hold off