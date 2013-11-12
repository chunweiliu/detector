% setup the environment
curDir = pwd;
dependDir = sprintf('%s/3rdparty', curDir);
mkdir(dependDir);

% download and compile liblinear
%% libsvm
cd(dependDir);
display('Downloading LIBLINEAR...');
cmd = 'wget http://www.csie.ntu.edu.tw/~cjlin/liblinear/liblinear-1.93.tar.gz; tar -xvf liblinear-1.93.tar.gz; rm -f liblinear-1.93.tar.gz;';
unix(cmd);
cd('liblinear-1.93/matlab');
make;
cd(curDir);

%% selective search
cd(dependDir);
display('Downloading Selective Search...');
cmd = 'wget http://koen.me/research/downloads/SelectiveSearchCodeIJCV.zip; unzip SelectiveSearchCodeIJCV.zip; rm -f SelectiveSearchCodeIJCV.zip;';
unix(cmd);
cd('SelectiveSearchCodeIJCV/');
% compile
if(~exist('anigauss'))
    fprintf('Compiling the anisotropic gauss filtering of:\n');
    fprintf('   J. Geusebroek, A. Smeulders, and J. van de Weijer\n');
    fprintf('   Fast anisotropic gauss filtering\n');
    fprintf('   IEEE Transactions on Image Processing, 2003\n');
    fprintf('Source code/Project page:\n');
    fprintf('   http://staff.science.uva.nl/~mark/downloads.html#anigauss\n\n');
    mex anigaussm/anigauss_mex.c anigaussm/anigauss.c -output anigauss
end

% Compile the code of Felzenszwalb and Huttenlocher, IJCV 2004.
if(~exist('mexFelzenSegmentIndex'))
    fprintf('Compiling the segmentation algorithm of:\n');
    fprintf('   P. Felzenszwalb and D. Huttenlocher\n');
    fprintf('   Efficient Graph-Based Image Segmentation\n');
    fprintf('   International Journal of Computer Vision, 2004\n');
    fprintf('Source code/Project page:\n');
    fprintf('   http://www.cs.brown.edu/~pff/segment/\n');
    fprintf('Note: A small Matlab wrapper was made. See demo.m for usage\n\n');
%     fprintf('   
    mex FelzenSegment/mexFelzenSegmentIndex.cpp -output mexFelzenSegmentIndex;
end
cd(curDir);

