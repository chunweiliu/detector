function compute_hog_pyramid_on_voc

% change this path if you install the VOC code elsewhere
addpath([cd '/VOCcode']);
addpath([cd '/features']);

% initialize VOC options
VOCinit;

% initialize parameters
params=get_default_params;
%params.feature_type='hog';

% compute feature for each class
for i=1:1%VOCopts.nclasses
    
    cls=VOCopts.classes{i};
    compute_hog_pyramid(VOCopts,cls,params);
    
end

function compute_hog_pyramid(VOCopts,cls,params)

% load training set

cp=sprintf(VOCopts.annocachepath,VOCopts.trainset);
if exist(cp,'file')
    fprintf('%s: loading training set\n',cls);
    load(cp,'gtids','recs');
else
    tic;
    gtids=textread(sprintf(VOCopts.imgsetpath,VOCopts.trainset),'%s');
    for i=1:length(gtids)
        % display progress
        if toc>1
            fprintf('%s: load: %d/%d\n',cls,i,length(gtids));
            drawnow;
            tic;
        end

        % read annotation
        recs(i)=PASreadrecord(sprintf(VOCopts.annopath,gtids{i}));
    end
    save(cp,'gtids','recs');
end

% compute pyramid for each training image

tic;
for i=1:length(gtids)
    % display progress
    if toc>1
        fprintf('%s: train: %d/%d\n',cls,i,length(gtids));
        drawnow;
        tic;
    end
    
    % find objects of class and extract difficult flags for these objects
    clsinds=strmatch(cls,{recs(i).objects(:).class},'exact');
    diff=[recs(i).objects(clsinds).difficult];
    
    % assign ground truth class to image
    if isempty(clsinds)
        gt=-1;          % no objects of class
    elseif any(~diff)
        gt=1;           % at least one non-difficult object of class
    else
        gt=0;           % only difficult objects
    end

    if gt
        % extract features for image
        fdp=sprintf(VOCopts.exfdpath,gtids{i});
        if exist(fdp,'file')
            % load features
            load(fdp,'fd','sc');
        else
            % compute and save features
            I=imread(sprintf(VOCopts.imgpath,gtids{i}));
            [fd,sc]=extractfd(I,params);
            save(fdp,'fd','sc');
        end
    end
end

function [fd,sc] = extractfd(I,params)
[fd,sc]=esvm_pyramid(double(I),params);


