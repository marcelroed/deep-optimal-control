function cmap = redblue(n)

if nargin < 1
    n = 256;
end

cmap = redbluecmap(21);
cmap = imresize(cmap, [n, 3]);  % original color map contain just 11 colors, this increase it to 64
cmap = min(max(cmap, 0), 1);

end
