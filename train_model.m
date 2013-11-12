function [bestModel, bestParams, bestAP] = ...
    train_model(labels, data, boxes, ids, modelName, feature_type)

% setup
VOCinit

% split data
fold = 3;
cvIds = wl_cvIds(ids, labels, fold);

% do cross validation on a given set of paramters
bestAP = 0;
cs = [0.001 0.01 0.1 1 10 100];

if strcmp(feature_type, 'wta')
    isWTA = 1;
    oridata = data;
    
    ks = [2 4 8];
    ms = [10 100 1000];
    [css, kss, mss] = meshgrid(cs, ks, ms);
else
    css = cs;
end


for ci = 1:numel(css)
    % for each parameter
    c = css(ci);
    
    if strcmp(feature_type,'wta')
        k = kss(ci);
        m = mss(ci);
        [data, thetas]= wtahash(oridata, k, m, [], isWTA);
        fprintf('%s: c %f k %d m %d\n', modelName, c, k, m);
    else
        fprintf('%s: c %f\n', modelName, c);
    end
      
    
    ap = 0;
    for i = 1:fold
        % for each training fold
        trainIds = [];
        for j = 1:fold
            if j ~= i
                trainIds = [trainIds; cvIds{j}];
            end
        end
        testIds = cvIds{i};

        % train a linear model
        w1 = sqrt(sum(labels(trainIds)~=1)/sum(labels(trainIds)==1));
        options = sprintf('-s 3 -c %f -w1 %f -w-1 1 -B 1', c, w1);
        model = train(labels(trainIds), sparse(data(trainIds,:)), options);

        % evaluate on validation set
        options = sprintf('-b 0');
        [plabels, acc, pscores] = predict(labels(testIds), sparse(data(testIds,:)), model, options);

        % addjust score
        pscores = pscores * model.Label(1);

        %
        dets{1} = ids(testIds);
        dets{2} = boxes(testIds,:);
        dets{3} = pscores;
        ap = ap + wl_evalAP(modelName, dets, VOCopts.trainset);
   
    end

    ap = ap / fold;
    
    if ap > bestAP
        bestAP = ap;
        bestParams.c = c;
        
        if strcmp(feature_type,'wta')
            bestParams.k = k;
            bestParams.m = m;
            bestParams.thetas = thetas;
            bestParams.isWTA = isWTA;
        end
        
        %bestModel = model;
    end
end

% Train the model using the best parameter
if strcmp(feature_type, 'wta')
    data = wtahash(oridata,bestParams.k,bestParams.m,bestParams.thetas,bestParams.isWTA);
end
options = sprintf('-s 3 -c %f -w1 %f -w-1 1 -B 1', bestParams.c, w1);
bestModel = train(labels, sparse(data), options);



function cvIds = wl_cvIds(ids, labels, k)
% wl_cvIds() will partition the labels into k parts with equally
% number of images
% Input:
%	ids: the image name for all the features
%	labels: the label for all the features
%	k: the number of partitions
%

% step 1: hash the image names
hash = VOChash_init(ids);

% step 2: get the unique name of the images
imgNames = unique(ids);

% step 2.1: get the positive image names
posImgNames = unique(ids(labels==1));
nPosImgs = length(posImgNames);

% step 2.2: get the negative image names
negImgNames = setdiff(imgNames, posImgNames);
nNegImgs = length(negImgNames);

% step 3: randomly split the positive image names
if nPosImgs ~= 0
    % step 3.1: randomly permute the postive image names
    posImgNames = posImgNames(randperm(nPosImgs));
    % step 3.2: split the image names into k parts
    n = floor(nPosImgs/k);
    count = 0;
    i = 1;
    cvIds{i} = [];
    for d=1:nPosImgs
        imgName = posImgNames{d};
        count = count + 1;
        % step 4.1: get the indices for the image name
        idx = VOChash_lookup(hash, imgName);
        if count < n || i==k
            cvIds{i} = [cvIds{i}; idx'];
        else
            cvIds{i} = [cvIds{i}; idx'];
            i = i+1;
            count = 0;
            cvIds{i} = [];
        end
    end
end

% step 4: randomly split the negative image names
if nNegImgs ~= 0
    % step 4.1: randomly permute the negative image names
    negImgNames = negImgNames(randperm(nNegImgs));
    % step 3.2: split the image names into k parts
    n = floor(nNegImgs/k);
    count = 0;
    i = 1;
    for d=1:nNegImgs
        imgName = negImgNames{d};
        count = count + 1;
        % step 4.1: get the indices for the image name
        idx = VOChash_lookup(hash, imgName);
        if count < n || i==k
            cvIds{i} = [cvIds{i}; idx'];
        else
            cvIds{i} = [cvIds{i}; idx'];
            i = i+1;
            count = 0;
        end
    end
end


