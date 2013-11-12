function box = box2pixel(box, scale, cellsize)
box = bsxfun(@rdivide, box, scale) * cellsize;