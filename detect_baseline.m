function detect_baseline(cls, year, feature_type, sample_params)

% load model
VOCinit;

% create results file
respath =sprintf(VOCopts.detrespath,'comp3',cls,feature_type,...
    sample_params.w,sample_params.h,sample_params.offset);
if ~exist(respath, 'file')
    
    % Load model 
    filepath = sprintf([VOCopts.localdir 'model_%s_%s_%s_%d_%d_%d.mat'],...
        cls, year, feature_type, sample_params.w, sample_params.h, sample_params.offset);
    load(filepath)
    
    % initialize parameters
    params = get_default_params;
    
    % load test set ('val' for development kit)
    [ids,gt]=textread(sprintf(VOCopts.imgsetpath,VOCopts.testset),'%s %d');
    
    fid=fopen(respath, 'w');
    
    % apply detector to each image
    tic;
    for i=1:length(ids)
        % display progress
        if toc>1
            fprintf('-> test: %d/%d (%s%s)\n',i,length(ids),cls,year);
            drawnow;
            tic;
        end
        
        % get the feature pyramid
        I = imread(sprintf(VOCopts.imgpath, ids{i}));
        pyramid_path = sprintf(VOCopts.hogpath, ids{i}, 0);
        if exist(pyramid_path, 'file')
            load(pyramid_path, 'fd', 'sc');
        else
            [fd, sc] = esvm_pyramid(double(I), params);
            save(pyramid_path, 'fd', 'sc');
        end
        
        % compute confidence of positive classification and bounding boxes
        [c,BB]=detect(model2,params2,fd,sc,sample_params,I, feature_type);
        
        % write to results file
        for j=1:length(c)
            fprintf(fid,'%s %f %d %d %d %d\n',ids{i},c(j),BB(j,:));
        end
        
    end
    fclose(fid);
end

function [c BB] = detect(model, params, fd, sc, sample_params,im, feature_type)

% Using selective search to purn negative examples
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


% get valid bbox hypotheses
w = sample_params.w;
h = sample_params.h;
d = sample_params.d;
cellsize = sample_params.cellsize;
offset = sample_params.offset;
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
[ssboxes blobIndIm blobBoxes hierarchy] = Image2HierarchicalGrouping(im, sigma, k, minSize, colorType, simFunctionHandles);
ssboxes = BoxRemoveDuplicates(ssboxes);

% get overlaping
ovlp = get_boxes_overlap_fast(ssboxes, bboxes); % small boxes not in current searching

% get examples
[gti, bbi] = ind2sub(size(ovlp), find(ovlp >= VOCopts.minoverlappos));

data = zeros(length(bbi),w*h*d);
boxes = zeros(length(bbi),4);
for j=1:length(bbi)
    
    % get feature vector from the feature map and the bbox
    data(j,:) = get_feature(fd{levels(bbi(j))}, featxy(bbi(j),:));
    
    % convert the sample bbox from feature domain to pixel domain
    boxes(j,:) = bboxes(bbi(j),:);
    
end

% apply wta hash
if strcmp(feature_type, 'wta')
    [data, ~] = wtahash(data,params.k,params.m,params.thetas,1);
end

% predict scores for the hypotheses
[~,~,scores] = predict(ones(size(data,1),1), sparse(data), model, '-b 0 -q');

[pick, BB] = wl_nms([boxes scores], 0.5);

c = BB(:,end);
BB = BB(:,1:4);