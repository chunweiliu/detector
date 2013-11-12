%% Winner Takes All (WTA) hashing
%  The algorithm do following:
%  1. get the largest feature's index from the first k random permutated features
%  2. repeat (1) m times
% input
%  x - n x d feature vectors
%  k - number of feature we look at
%  m - number of perumation
%  thetas - d x m permutation vector in each column
% output
%  b - n x m*lessbit binary feature
%  c - n x m real number feature
%  thetas
% usage
%  y = wtahash(x, k, m, thetas, isWTA);

function [y, thetas] = wtahash(x, k, m, thetas, isWTA)

%% Setting
% DEFALUT_K = 16;
% DEFALUT_M = 24;
% switch nargin
%     case 4
%         thetas = zeros(size(x,2),m);
%         for i = 1:m
%             thetas(:,i) = reshape(randperm(size(x,2)),[],1);
%         end
%     case 3
%         m = DEFALUT_M;
%         thetas = zeros(size(x,2),m);
%         for i = 1:m
%             thetas(:,i) = reshape(randperm(size(x,2)),[],1);
%         end
%     case 2
%         % b = wtahash(x), using default setting from the paper
%         k = min(DEFALUT_K, size(x,2));
%         m = DEFALUT_M;
%         thetas = zeros(size(x,2),m);
%         for i = 1:m
%             thetas(:,i) = reshape(randperm(size(x,2)),[],1);
%         end
% end

if isempty(thetas)
    thetas = zeros(size(x,2),m);
    for i = 1:m
        thetas(:,i) = reshape(randperm(size(x,2)),[],1);
    end
end    

%% Real value representation
c = zeros(size(x,1), m);
for i = 1:m
    theta = thetas(:,i);
    [~,c(:,i)] = max(x(:,theta(1:k)), [], 2);
end

%% Output
switch isWTA
    case 1
        % Unary representation
        % Transform real value to unary representation (m*k)
        % * there are only m non-zero entry in each representation
        u = zeros(size(x,1), m*k);
        for n = 1:size(x,1)
            for i = 1:m
                u(n,(i-1)*k+c(n,i)) = 1;        
            end
        end
        y = u;
        
    case 2
        %% Bits representation (using in hardware)
        lessbit = floor(log2(k)+1); % Matlab's index start from 1
        b = zeros(size(c,1), m*lessbit);
        for i = 1:size(b,1)
            binstr = reshape((dec2bin(c(i,:),lessbit))',1,[]); % reshape is row-major, need to transpose
            b(i,:) = double(binstr) - double('0'); % by Jared
        end
        y = b;
        
    otherwise
        y = c;
end


