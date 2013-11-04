% setup the environment
curDir = pwd;
dependDir = sprintf('%s/3rdparty', curDir);
mkdir(dependDir);

% download and compile liblinear
cd(dependDir);
display('Downloading LIBLINEAR...');
cmd = 'wget http://www.csie.ntu.edu.tw/~cjlin/liblinear/liblinear-1.93.tar.gz; tar -xvf liblinear-1.93.tar.gz; rm -f liblinear-1.93.tar.gz;';
unix(cmd);
cd('liblinear-1.93/matlab');
make;
cd(curDir);


