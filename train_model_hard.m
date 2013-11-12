function [bestModel, bestParams, bestAP] =...
    train_model_hard(labels, data, boxes, ids, modelName, model, params, feature_type)

posidx = find(labels==1);
negidx = find(labels==-1);

%[poslabels, posdata, posboxes, posids] = ...
%    get_the_representative(labels(posidx), data(posidx,:), boxes(posidx,:), ids(posidx), model);
poslabels = labels(posidx);
posdata = data(posidx,:);
posboxes = boxes(posidx,:);
posids = ids(posidx);

[neglabels, negdata, negboxes, negids] = ...
    get_the_representative(labels(negidx), data(negidx,:), boxes(negidx,:), ids(negidx),...
    model, params, feature_type);

% cascade
labels = [poslabels; neglabels];
data = [posdata; negdata];
boxes = [posboxes; negboxes];
ids = [posids; negids];
[bestModel, bestParams, bestAP] = train_model(labels, data, boxes, ids, modelName, feature_type);

function [ret_labels, ret_data, ret_boxes, ret_ids] = ...
    get_the_representative(labels, data, boxes, ids, model, params, feature_type)

%oridata = data;
if strcmp(feature_type, 'wta')
    wtadata = wtahash(data,params.k,params.m,params.theta,params.isWTA);
    [~,~,scores] = predict(labels, sparse(wtadata), model, '-b 0');
else
    [~,~,scores] = predict(labels, sparse(data), model, '-b 0');
end


% correct the score according to the model
scores = scores*model.Label(1);

% get unique image ids
imgid = unique(ids);

ret_labels = labels;
ret_data = data;
ret_boxes = boxes;
ret_ids = ids;
n = 0;
for i=1:length(imgid)
    
    % find bounding boxes in the same image
    id = imgid{i};
    idx = find(strcmp(ids,id)==1);
    
    % nms perfom on all bounding boxes in an image
    [pick, newboxes] = wl_nms([boxes(idx,:), scores(idx)], 0.5);
    newdata = get_new(data(idx,:), pick);
    newids = get_new(ids(idx), pick);
    newlabels = get_new(labels(idx), pick);
    
    ret_data(1+n:n+size(newids,1),:)=newdata;
    ret_boxes(1+n:n+size(newids,1),:)=newboxes(:,1:4);
    ret_ids(1+n:n+size(newids,1))=newids;
    ret_labels(1+n:n+size(newids,1))=newlabels;
    
    n = n + size(newids,1);
    
end
ret_data = ret_data(1:n,:);
ret_boxes = ret_boxes(1:n,:);
ret_ids = ret_ids(1:n);
ret_labels = ret_labels(1:n);

function data = get_new(data, idx)
data = data(idx,:);