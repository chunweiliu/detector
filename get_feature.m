%function feature = get_feature(pyramid, xmin, xmax, ymin, ymax, level)
function feature = get_feature(pyramid, bbox, level)

% Get a row feature vector from a feature pyramid
xmin=bbox(1);
ymin=bbox(2);
xmax=bbox(3);
ymax=bbox(4);

pyramid_level = pyramid{level};

if ymax > size(pyramid_level,1) || xmax > size(pyramid_level,2)
    assert('index out of boundary');
end

feature = reshape(pyramid_level(ymin:ymax, xmin:xmax, :), 1, []);