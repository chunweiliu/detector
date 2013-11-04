function model = train_model(labels, data, boxes, ids)

% setup
%global VOCopts
VOCinit

% split data
k = 2;
cvIds = wl_cvIds(ids, labels, k);

% do cross validation on a given set of paramters
bestC = [];
bestAP = 0;
cs = [0.001 0.01 0.1 1 10 100];
for ci = 1:length(cs)
    % for each parameter
    c = cs(ci);
    ids2 = [];
    boxes2 = [];
    confidence2 = [];
    for i = 1:k
        % for each training fold
        trainIds = [];
        for j = 1:k
            if j ~= i
                trainIds = [trainIds; cvIds{j}];
            end
        end
        testIds = cvIds{i};

        % train a linear model
        w1 = sqrt(sum(labels(trainIds)~=1)/sum(labels(trainIds)==1));
        options = sprintf('-s 3 -c %f -w1 %f -w-1 1 -B 1 -q', c, w1);
        model = train(labels(trainIds), sparse(data(trainIds,:)), options);

        % evaluate on validation set
        options = sprintf('-b 0 -q');
        [plabels, acc, pscores] = predict(labels(testIds), sparse(data(testIds,:)), model, options);

        % addjust score
        pscores = pscores * model.Label(1);

        % gather the detection results
        ids2 = [ids2; ids(testIds)];
        boxes2 = [boxes2; boxes(testIds,:)];
        confidence2 = [confidence2; pscores];
    end

    % compute the AP for the detection results
    dets{1} = ids2;
    dets{2} = boxes2;
    dets{3} = confidence2;
    modelName = 'aeroplane';

    % debug
    ap = wl_evalAP(modelName, dets, VOCopts.trainset);
   
    if ap > bestAP
        bestAP = ap;
        bestC = c;
    end
end




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


