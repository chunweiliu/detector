function ind = VOChash_lookup(hash,s,year)

hsize=numel(hash.key);

if nargin < 3
   year = 2007;
end

if year == 2007
    h=mod(str2double(s),hsize)+1;
else
    h=mod(str2double(s([3:4 6:11 13:end])),hsize)+1;
end
    
ind=hash.val{h}(strmatch(s,hash.key{h},'exact'));
