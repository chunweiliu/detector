function feature = get_feature(feat, bbox)
% Get a row feature vector from a feature map

xmin=bbox(1);
ymin=bbox(2);
xmax=bbox(3);
ymax=bbox(4);

feature = reshape(feat(ymin:ymax, xmin:xmax, :), 1, []);