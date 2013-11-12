
addpath('VOCcode')

% initialize VOC options
VOCinit;

addpath('data')
addpath('utils')
addpath('VOCcode')
addpath('features')
addpath('3rdparty/liblinear-1.93/matlab/');
addpath(genpath('3rdparty/SelectiveSearchCodeIJCV/'));

year = '2007';
%feature_type = 'hog';
feature_type = 'wta';

% compute feature for each class
aps = zeros(VOCopts.nclasses,1);
for i=1:VOCopts.nclasses
    
    cls=VOCopts.classes{i};
    
    if strcmp(feature_type, 'hog')
        train_baseline(cls,year);
        detect_baseline(cls,year);
    else
        train_baseline_wta(cls,year);
        detect_baseline_wta(cls,year);
    end
    
    
    % Evalutation
    [recall,prec,aps(i)]=VOCevaldet(VOCopts,'comp3',cls,true, year,feature_type);  % compute and display PR
    fprintf('The test AP of the class %s%s is %f\n', cls, year, aps(i));
end
