function ret_negdata = get_negative_examples(VOCopts, neg, nnegmax)

% initialize parameters
params = get_default_params;
nneg = 100;

% define filter size
w = 16; % 128/8
h = 10; % 80/8
offset = 10; % 80/8

tic;
ret_negdata = [];
for i=1:length(neg)
    
    % break
    if size(ret_negdata,1) > nnegmax
        break;
    end

    % report
    if toc>1
        fprintf('-> get neg: %d/%d\n', i, length(neg));
        drawnow;
        tic;
    end
    
    % if file exist directly load
    negdata1 = [];
    negdata1path = sprintf(VOCopts.negdata1path, neg(i).id);
    if exist(negdata1path, 'file')
        load(negdata1path, 'negdata1');
    else

        % get pyramid
        pyramid_path = sprintf(VOCopts.exfdpath, neg(i).id);
        if exist(pyramid_path, 'file')
            load(pyramid_path, 'fd', 'sc');
        else
            I = imread(sprintf(VOCopts.imgpath, neg(i).id));
            [fd,sc] = esvm_pyramid(double(I), params);
            save(pyramid_path, 'fd', 'sc');
        end

        % get all negative examples based on uniform sampling
        bboxes = [];
        levels = [];
        for j=1:length(fd)

            newbbs = sample_bbox(fd{j}, w, h, offset);
            bboxes = [bboxes; newbbs];
            levels = [levels; j*ones(size(newbbs,1),1)];

        end

        for j = 1:length(bboxes)
            negdata1 = [negdata1; get_feature(fd, bboxes(j,:), levels(j))];
        end
        
        % save file
        save(negdata1path, 'negdata1');
    end
    
    ret_negdata = [ret_negdata; negdata1];
    
    
end