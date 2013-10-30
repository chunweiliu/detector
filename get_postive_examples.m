function [ret_posdata, ret_negdata] = get_postive_examples(VOCopts, pos...
    sample_params)
% return positive examples and negative examples (which is partical
% overlapping with the ground truth).

% initialize parameters
params=get_default_params;

% define filter size
w = sample_params.w; % 128/8
h = sample_params.h; % 80/8
offset = sample_params.offset; % 80/8

tic;
ret_posdata = [];
ret_negdata = [];
for i=1:length(pos)

    % report
    if toc>1
        fprintf('-> get pos: %d/%d\n', i, length(pos));
        drawnow;
        tic;
    end
    
    % get data if it exist
    posdata = [];
    negdata2 = [];
    posdatapath = sprintf(VOCopts.posdatapath, pos(i).id);
    negdata2path = sprintf(VOCopts.negdata2path, pos(i).id);
    if exist(posdatapath, 'file') && exist(negdata2path, 'file')
        load(posdatapath, 'posdata');
        load(negdata2path, 'negdata2');
    else 
        % initial data
        posdata = [];
        negdata2 = [];
        
        % get pyramid
        pyramid_path = sprintf(VOCopts.exfdpath, pos(i).id);
        if exist(pyramid_path, 'file')
            load(pyramid_path, 'fd', 'sc');
        else
            I = imread(sprintf(VOCopts.imgpath, pos(i).id));
            [fd,sc] = esvm_pyramid(double(I), params);
            save(pyramid_path, 'fd', 'sc');
        end

        % get positive examples based on overlaping score
        bboxgt = zeros(length(fd), 4);
        bboxes = [];
        levels = [];
        for j=1:length(fd)
            newbbs = sample_bbox(fd{j}, w, h, offset);
            bboxes = [bboxes; newbbs];
            levels = [levels; j*ones(size(newbbs,1),1)];

            bboxgt(j,:) = pos(i).boxes * sc(j);
        end
        ovlp = get_boxes_overlap_fast(bboxgt, bboxes);

        % get positive examples
        [gti, bbi] = ind2sub(size(ovlp), find(ovlp >= VOCopts.minoverlap));
        for j=1:length(gti)
            posdata = [posdata; get_feature(fd, bboxes(bbi(j),:), levels(gti(j)))];
        end

        % get negative examples (overlaping not enough)
        [gti, bbi] = ind2sub(size(ovlp), find(ovlp > 0.3 & ovlp < 0.4));
        for j=1:length(gti)
            negdata2 = [negdata2; get_feature(fd, bboxes(bbi(j),:), levels(gti(j)))];
        end
        
        % save files
        save(posdatapath, 'posdata');
        save(negdata2path, 'negdata2');
    end
    
    ret_posdata = [ret_posdata; posdata];
    ret_negdata = [ret_negdata; negdata2];
    
end