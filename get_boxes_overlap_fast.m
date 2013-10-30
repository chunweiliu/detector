function ovlp = get_boxes_overlap_fast(bs, bg)
% ovlp is a matrix contained the overlapping between bs and bg, where row
% is bs and column is bg. Both bs and bg are boxes [xmin xmax ymin ymax]. 
nbs = size(bs, 1);
nbg = size(bg, 1);
ovlp = zeros(nbs, nbg);
sbs  = (bs(:, 3) - bs(:, 1) + 1) .* (bs(:, 4) - bs(:, 2) + 1);
sbg  = (bg(:, 3) - bg(:, 1) + 1) .* (bg(:, 4) - bg(:, 2) + 1);
idx  = find(sbg > 0);
for ii = 1:length(idx)
  i = idx(ii);
  g = bg(i, :);
  bi = [bsxfun(@max, bs(:, 1:2), g(1:2)), ...
    bsxfun(@min, bs(:, 3:4), g(3:4))];
  iw = bi(:, 3) - bi(:, 1) + 1;
  ih = bi(:, 4) - bi(:, 2) + 1;
  si = iw .* ih;
  ind = iw > 0 & ih > 0;
  ovlp(ind, i) = si(ind) ./ (sbs(ind)-si(ind) + sbg(i));
end

      
