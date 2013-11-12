function [ret_posdata, ret_negdata2, ret_posids, ret_negids2, ret_posboxes, ret_negboxes2]...
    = get_positive_examples(posim, sample_params, cls, year)
% return positive examples and negative examples (which is partical
% overlapping with the ground truth).

VOCinit;

conf       = voc_config('pascal.year', year);
dataset_fg = conf.training.train_set_fg;
%dataset_bg = conf.training.train_set_bg;
cachedir   = conf.paths.model_dir;
%VOCopts    = conf.pascal.VOCopts;

% initialize parameters
params=get_pyramid_params;

% define filter size
w = sample_params.w; % 128/8
h = sample_params.h; % 80/8
d = sample_params.d;
offset = sample_params.offset; % 80/8
cellsize = sample_params.cellsize;

respath = [cachedir cls '_' dataset_fg '1_' year '_pos_' num2str(w) '_' num2str(h) '_' num2str(offset)];
try
    load(respath);
catch
    
    tic;
    %for i=1:length(pos) % for each bounding box
    for i=1:length(posim) % for each image
        
        % report
        if toc>1
            fprintf('-> get pos: %d/%d (%s%s)\n', i, length(posim), cls, year);
            drawnow;
            tic;
        end
        
        % write the data to the disk
        hogdatapath = sprintf(VOCopts.hogdatapath, posim(i).id, posim(i).flip, ...
            w, h, offset);
        if ~exist(hogdatapath, 'file')
            % do nothing if the file already exist
        else
            % get the feature pyramid for positive boxes i
            pyramid_path = sprintf(VOCopts.hogpath, posim(i).id, posim(i).flip);
            if exist(pyramid_path, 'file')
                load(pyramid_path, 'fd', 'sc');
            else
                I = imread(sprintf(VOCopts.imgpath, posim(i).id));
                if posim(i).flip == 1
                    I = flipdim(I,2);
                end
                [fd, sc] = esvm_pyramid(double(I), params);
                save(pyramid_path, 'fd', 'sc');
            end
            
            % get ground truth bounding boxes and sample lots of hypothesis bounding boxes
            %bboxes = [];
            nmax = ceil(length(fd)*size(fd{1},2)*size(fd{1},1)/(offset^2));
            levels = zeros(nmax, 1);
            featxy = zeros(nmax, 4); % for indexing feature
            nb = 0;
            for j=1:length(fd)
                
                % for each resolution, sample bounding boxes in feature domain
                newbxy = sample_bbox(fd{j}, w, h, offset);
                featxy(1+nb:nb+size(newbxy,1),:) = newbxy;
                %featxy = [featxy; newbxy];
                
                % track the resolution of the sample boxes
                %levels = [levels; j*ones(size(newbxy,1),1)];
                levels(1+nb:nb+size(newbxy,1)) = j*ones(size(newbxy,1),1);
                
                nb = nb + size(newbxy,1);
            end
            featxy = featxy(1:nb,:);
            levels = levels(1:nb);
                        
            % convert the sample bboxes to the pixel domain
            bboxes = box2pixel(featxy, sc(levels)', cellsize);
            
            % compute overlap for all hypotheses
            ovlp = get_boxes_overlap_fast(posim(i).boxes, bboxes); % small boxes not in current searching
            
            % get positive examples
            [gti, bbi] = ind2sub(size(ovlp), find(ovlp >= VOCopts.minoverlappos));
            
            posdata = zeros(length(bbi),w*h*d);
            posboxes = zeros(length(bbi),4);
            for j=1:length(bbi)

                % get feature vector from the feature map and the bbox  
                posdata(j,:) = get_feature(fd{levels(bbi(j))}, featxy(bbi(j),:));
                
                % convert the sample bbox from feature domain to pixel domain
                posboxes(j,:) = bboxes(bbi(j),:);
                
            end
            posids = repmat({posim(i).id}, size(posdata,1), 1);
            
            
            % get negative examples (overlaping not enough)
            [gti, bbi] = ind2sub(size(ovlp), find(ovlp > VOCopts.minoverlapneg...
                & ovlp < VOCopts.maxoverlapneg));
            negdata2 = zeros(length(bbi), w*h*d);
            negboxes2 = zeros(length(bbi), 4);            
            for j=1:length(bbi)
                
                % get feature vector from the feature map and the bbox
                negdata2(j,:) = get_feature(fd{levels(bbi(j))}, featxy(bbi(j),:));
                
                % convert the bounding box into original scale
                negboxes2(j,:) = bboxes(bbi(j),:);
            end
            negids2 = repmat({posim(i).id}, size(negdata2,1), 1);
            
            % save the file
            save(hogdatapath,'posdata','posids','posboxes',...
                'negdata2','negids2','negboxes2');
        end
    end
    
    % cascate data
    tic;
    nmax = length(posim)*100;
    ret_posdata = zeros(nmax,w*h*d); % take times
    ret_posids = cell(nmax,1);
    ret_posboxes = zeros(nmax,4);
    ret_negdata2 = zeros(nmax,w*h*d);
    ret_negids2 = cell(nmax,1);
    ret_negboxes2 = zeros(nmax,4);
    np = 0;
    nn = 0;
    for i=1:length(posim) 
        % report
        if toc>1
            fprintf('-> load pos: %d/%d (%s%s)\n', i, length(posim), cls, year);
            drawnow;
            tic;
        end

        hogdatapath = sprintf(VOCopts.hogdatapath, posim(i).id, posim(i).flip,...
            w, h, offset);
        load(hogdatapath)
       
        ret_posdata(1+np:np+size(posdata,1),:) = posdata;
        ret_posids(1+np:np+size(posids,1)) = posids;
        ret_posboxes(1+np:np+size(posboxes,1),:) = posboxes;
        np = np + size(posboxes,1);
         
        ret_negdata2(1+nn:nn+size(negdata2,1),:) = negdata2;
        ret_negids2(1+nn:nn+size(negids2,1)) = negids2;
        ret_negboxes2(1+nn:nn+size(negboxes2,1),:) = negboxes2;
        nn = nn + size(negboxes2,1);
    end
    ret_posdata = ret_posdata(1:np,:);
    ret_posids = ret_posids(1:np);
    ret_posboxes = ret_posboxes(1:np,:);
    ret_negdata2 = ret_negdata2(1:nn,:);
    ret_negids2 = ret_negids2(1:nn);
    ret_negboxes2 = ret_negboxes2(1:nn,:);
    
    
    save(respath, 'ret_posdata', 'ret_negdata2', 'ret_posids',...
        'ret_negids2', 'ret_posboxes', 'ret_negboxes2', '-v7.3');
    
end

