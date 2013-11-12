function train_baseline(cls, year, feature_type, sample_params)
% Initial setting
VOCinit;

filepath = sprintf([VOCopts.localdir 'model_%s_%s_%s_%d_%d_%d.mat'],...
    cls, year, feature_type, sample_params.w, sample_params.h, sample_params.offset);
if exist(filepath, 'file')
    % do nothing
else
    
    % Get training data
    [labels, data, boxes, ids] = get_training_data(cls, year, sample_params);
    
    % Train first model
    %[model1, c1, k1, m1, thetas1, ap1] = train_wta_model(labels, data, boxes, ids, cls);
    [model1, params1, ap1] = train_model(labels, data, boxes, ids, cls, feature_type);
    
    % Retraing the model
    [model2, params2, ap2] = train_model_hard(labels, data, boxes, ids, cls, model1, params1);
    
    % Save model
    save(filepath, 'model1', 'params1', 'ap1','model2', 'params2', 'ap2', 'sample_params');   
    fprintf('The baseline of the class %s%s is %f\n', cls, year, ap2);
end
