function ret_negdata = get_negative_examples(neg, nnegmax, ...
    sample_params, cls, year)

conf       = voc_config('pascal.year', year);
dataset_fg = conf.training.train_set_fg;
dataset_bg = conf.training.train_set_bg;
cachedir   = conf.paths.model_dir;
VOCopts    = conf.pascal.VOCopts;

% initialize parameters
params = get_default_params;

% define filter size
w = sample_params.w; % 128/8
h = sample_params.h; % 80/8
offset = sample_params.offset; % 80/8

try

    load([cachedir cls '_' dataset_bg '_' year '_neg_' num2str(w) '_' num2str(h) '_' num2str(offset)]);
catch
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

    save([cachedir cls '_' dataset_bg '_' year '_neg_' num2str(w) '_' num2str(h) '_' num2str(offset)], 'ret_negdata');
end
