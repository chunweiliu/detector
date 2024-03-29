clear VOCopts

% dataset
%
% Note for experienced users: the VOC2008-10 test sets are subsets
% of the VOC2010 test set. You don't need to do anything special
% to submit results for VOC2008-10.

VOCopts.dataset='VOC2007';

% get devkit directory with forward slashes
devkitroot=strrep(fileparts(fileparts(mfilename('fullpath'))),'\','/');

% change this path to point to your copy of the PASCAL VOC data
%VOCopts.datadir=[devkitroot '/'];
%VOCopts.datadir='/Users/chunweiliu/Data/VOCdevkit/';
%VOCopts.datadir='/Users/chunweiliu/Documents/MATLAB/VOC2007/VOCdevkit/';
VOCopts.datadir='/mnt/raid/data/chunwei/pascal/';

% change this for your specific project name
VOCopts.projname='detector';

% change this path to a writable directory for your results
%VOCopts.resdir=[devkitroot '/results/' VOCopts.dataset '/'];
VOCopts.resdir=[VOCopts.datadir VOCopts.projname '/results/' VOCopts.dataset '/'];
if ~exist(VOCopts.resdir, 'dir')
  mkdir(VOCopts.resdir)
end

% change this path to a writable local directory for the example code
VOCopts.localdir=[VOCopts.datadir VOCopts.projname '/local/' VOCopts.dataset '/'];
if ~exist(VOCopts.localdir, 'dir')
  mkdir(VOCopts.localdir)
end

% initialize the training set

%VOCopts.trainset='train'; % use train for development
VOCopts.trainset='trainval'; % use train+val for final challenge

% initialize the test set

%VOCopts.testset='val'; % use validation data for development test set
VOCopts.testset='test'; % use test set for final challenge

% initialize main challenge paths

VOCopts.annopath=[VOCopts.datadir VOCopts.dataset '/Annotations/%s.xml'];
VOCopts.imgpath=[VOCopts.datadir VOCopts.dataset '/JPEGImages/%s.jpg'];
VOCopts.imgsetpath=[VOCopts.datadir VOCopts.dataset '/ImageSets/Main/%s.txt'];
VOCopts.clsimgsetpath=[VOCopts.datadir VOCopts.dataset '/ImageSets/Main/%s_%s.txt'];
VOCopts.clsrespath=[VOCopts.resdir 'Main/%s_cls_' VOCopts.testset '_%s.txt'];
VOCopts.detrespath=[VOCopts.resdir 'Main/%s_det_' VOCopts.testset '_%s.txt'];

% initialize segmentation task paths

VOCopts.seg.clsimgpath=[VOCopts.datadir VOCopts.dataset '/SegmentationClass/%s.png'];
VOCopts.seg.instimgpath=[VOCopts.datadir VOCopts.dataset '/SegmentationObject/%s.png'];

VOCopts.seg.imgsetpath=[VOCopts.datadir VOCopts.dataset '/ImageSets/Segmentation/%s.txt'];

VOCopts.seg.clsresdir=[VOCopts.resdir 'Segmentation/%s_%s_cls'];
VOCopts.seg.instresdir=[VOCopts.resdir 'Segmentation/%s_%s_inst'];
VOCopts.seg.clsrespath=[VOCopts.seg.clsresdir '/%s.png'];
VOCopts.seg.instrespath=[VOCopts.seg.instresdir '/%s.png'];

% initialize layout task paths

VOCopts.layout.imgsetpath=[VOCopts.datadir VOCopts.dataset '/ImageSets/Layout/%s.txt'];
VOCopts.layout.respath=[VOCopts.resdir 'Layout/%s_layout_' VOCopts.testset '.xml'];

% initialize action task paths

VOCopts.action.imgsetpath=[VOCopts.datadir VOCopts.dataset '/ImageSets/Action/%s.txt'];
VOCopts.action.clsimgsetpath=[VOCopts.datadir VOCopts.dataset '/ImageSets/Action/%s_%s.txt'];
VOCopts.action.respath=[VOCopts.resdir 'Action/%s_action_' VOCopts.testset '_%s.txt'];

% initialize the VOC challenge options

% classes

VOCopts.classes={...
    'aeroplane'
    'bicycle'
    'bird'
    'boat'
    'bottle'
    'bus'
    'car'
    'cat'
    'chair'
    'cow'
    'diningtable'
    'dog'
    'horse'
    'motorbike'
    'person'
    'pottedplant'
    'sheep'
    'sofa'
    'train'
    'tvmonitor'};

VOCopts.nclasses=length(VOCopts.classes);	

% poses

VOCopts.poses={...
    'Unspecified'
    'Left'
    'Right'
    'Frontal'
    'Rear'};

VOCopts.nposes=length(VOCopts.poses);

% layout parts

VOCopts.parts={...
    'head'
    'hand'
    'foot'};    

VOCopts.nparts=length(VOCopts.parts);

VOCopts.maxparts=[1 2 2];   % max of each of above parts

% actions

VOCopts.actions={...    
    'other'             % skip this when training classifiers
    'jumping'           % new in VOC2011
    'phoning'
    'playinginstrument'
    'reading'
    'ridingbike'
    'ridinghorse'
    'running'
    'takingphoto'
    'usingcomputer'
    'walking'};

VOCopts.nactions=length(VOCopts.actions);

% overlap threshold

VOCopts.minoverlap=0.5;

% annotation cache for evaluation

VOCopts.annocachepath=[VOCopts.localdir '%s_anno.mat'];

%% options for example implementations

%VOCopts.exfdpath=[VOCopts.localdir '%s_fd.mat'];
VOCopts.hogpath=[VOCopts.localdir '%s_%d_hog.mat']; % id, flip
VOCopts.hogdatapath=[VOCopts.localdir '%s_%d_%d_%d_%d_hogdata.mat'];
VOCopts.prpath=[VOCopts.resdir 'Main/%s_%s_%s_pr.png'];
VOCopts.detrespath=[VOCopts.resdir 'Main/%s_det_' VOCopts.testset '_%s_%s_%d_%d_%d.txt']; %cls feature_type, sample_params

VOCopts.minoverlappos=0.8;
VOCopts.minoverlapneg=0.2;
VOCopts.maxoverlapneg=0.5;