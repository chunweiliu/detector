function [ret_negdata1, ret_negids1, ret_negboxes1] = ...
    get_negative_examples(neg, sample_params, cls, year)

VOCinit;

conf       = voc_config('pascal.year', year);
%dataset_fg = conf.training.train_set_fg;
dataset_bg = conf.training.train_set_bg;
cachedir   = conf.paths.model_dir;
%VOCopts    = conf.pascal.VOCopts;

% initialize parameters
params = get_pyramid_params;

% define filter size
w = sample_params.w; % 128/8
h = sample_params.h; % 80/8
d = sample_params.d;
offset = sample_params.offset; % 80/8
cellsize = sample_params.cellsize;

% Using selective search to purn negative examples
% define selective search parameters
colorTypes = {'Hsv', 'Lab', 'RGI', 'H', 'Intensity'};
colorType = colorTypes{1}; % Single color space for demo

% Here you specify which similarity functions to use in merging
simFunctionHandles = {@SSSimColourTextureSizeFillOrig, @SSSimTextureSizeFill, @SSSimBoxFillOrig, @SSSimSize};
simFunctionHandles = simFunctionHandles(1:2); % Two different merging strategies

% Thresholds for the Felzenszwalb and Huttenlocher segmentation algorithm.
% Note that by default, we set minSize = k, and sigma = 0.8.
k = 200; % controls size of segments of initial segmentation. 
minSize = k;
sigma = 0.8;


try
    load([cachedir cls '_' dataset_bg '_' year '_neg_' num2str(w) '_' num2str(h) '_' num2str(offset)]);
catch
    tic;
    for i=1:length(neg)
        
        % report
        if toc>1
            fprintf('-> get neg: %d/%d (%s%s)\n', i, length(neg), cls, year);
            drawnow;
            tic;
        end
        
        % write the data to the disk
        hogdatapath = sprintf(VOCopts.hogdatapath, neg(i).id, neg(i).flip, ...
            w, h, offset);
        if exist(hogdatapath, 'file')
            % do nothing
        else
            
            % get pyramid
            im = imread(sprintf(VOCopts.imgpath, neg(i).id));
            pyramid_path = sprintf(VOCopts.hogpath, neg(i).id, neg(i).flip);
            if exist(pyramid_path, 'file')
                load(pyramid_path, 'fd', 'sc');
            else
                %I = imread(sprintf(VOCopts.imgpath, neg(i).id));
                if neg(i).flip == 1
                    im = flipdim(im,2);
                end
                [fd, sc] = esvm_pyramid(double(im), params);
                save(pyramid_path, 'fd', 'sc');
            end
            
             
            
            
            % get valid bbox hypotheses
            nmax = ceil(length(fd)*size(fd{1},2)*size(fd{1},1)/(offset^2));
           
            featxy = zeros(nmax,4);
            levels = zeros(nmax,1);
            nb = 0;
            for j=1:length(fd)
                
                % for each resolution, sample bounding boxes in feature domain
                newbxy = sample_bbox(fd{j}, w, h, offset);
                featxy(1+nb:nb+size(newbxy,1),:) = newbxy;
                
                % track the resolution of the sample boxes
                levels(1+nb:nb+size(newbxy,1)) = j*ones(size(newbxy,1),1);
                
                nb = nb + size(newbxy,1);
                
            end
            featxy = featxy(1:nb,:);
            levels = levels(1:nb);
            
            bboxes = box2pixel(featxy, sc(levels)', cellsize);
            
            % compute overlap on ssbboxes for all hypotheses
            % Perform Selective Search
            [ssboxes blobIndIm blobBoxes hierarchy] = Image2HierarchicalGrouping(im, sigma, k, minSize, colorType, simFunctionHandles);
            ssboxes = BoxRemoveDuplicates(ssboxes);
            
            ovlp = get_boxes_overlap_fast(ssboxes, bboxes); % small boxes not in current searching
           
            
             % get negative examples
            [gti, bbi] = ind2sub(size(ovlp), find(ovlp >= VOCopts.minoverlappos));
            
            negdata1 = zeros(length(bbi),w*h*d);
            negboxes1 = zeros(length(bbi),4);
            for j=1:length(bbi)

                % get feature vector from the feature map and the bbox  
                negdata1(j,:) = get_feature(fd{levels(bbi(j))}, featxy(bbi(j),:));
                
                % convert the sample bbox from feature domain to pixel domain
                negboxes1(j,:) = bboxes(bbi(j),:);
                
            end
            negids1 = repmat({neg(i).id}, size(negdata1,1), 1);   
            
            % save file
            save(hogdatapath, 'negdata1', 'negids1', 'negboxes1');
        end
    end
    
    % load file
    tic;
    nmax = length(neg)*1000;
    ret_negdata1 = zeros(nmax,w*h*d);
    ret_negids1 = cell(nmax,1);
    ret_negboxes1 = zeros(nmax,4);
    nn = 0;
    for i=1:length(neg)
        % report
        if toc>1
            fprintf('-> load neg: %d/%d (%s%s)\n', i, length(neg), cls, year);
            drawnow;
            tic;
        end

        hogdatapath = sprintf(VOCopts.hogdatapath, neg(i).id, neg(i).flip,...
            w, h, offset);
        load(hogdatapath)
        
        ret_negdata1(1+nn:nn+size(negdata1,1),:) = negdata1;
        ret_negids1(1+nn:nn+size(negdata1,1)) = negids1;
        ret_negboxes1(1+nn:nn+size(negdata1,1),:) = negboxes1;
        
        nn = nn + size(negdata1,1);
    end
    ret_negdata1 = ret_negdata1(1:nn,:);
    ret_negids1 = ret_negids1(1:nn);
    ret_negboxes1 = ret_negboxes1(1:nn,:);
    
    save([cachedir cls '_' dataset_bg '_' year '_neg_' num2str(w) '_' num2str(h) '_' num2str(offset)],...
        'ret_negdata1', 'ret_negids1', 'ret_negboxes1', '-v7.3');
end
