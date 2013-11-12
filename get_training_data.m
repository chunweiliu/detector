function [labels, data, boxes, ids] = get_training_data(cls, year, sample_params)

% Get annotations from the class
[pos, neg, posim] = pascal_data(cls, year);

% Get training examples from the class
[posdata, negdata2, posids, negids2, posboxes, negboxes2] = ...
    get_positive_examples(posim, sample_params, cls, year);

sample_params.offset = 8*sample_params.offset; % negative examples are too many
[negdata1, negids1, negboxes1] = ...
    get_negative_examples(neg, sample_params, cls, year);

% Get training data 
data = [posdata; negdata1; negdata2];
labels = [ones(size(posdata,1),1); -ones(size(negdata1,1),1); -ones(size(negdata2,1),1)];
boxes = [posboxes; negboxes1; negboxes2];
ids = [posids; negids1; negids2];