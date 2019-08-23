% This file can be used to reproduce all figures in
%?Benning, Celledoni, Ehrhardt, Owren & Schönlieb (2019). 
% Deep learning as optimal control problems: models and 
% numerical methods. http://arxiv.org/abs/1904.05657
%
% Create a red and blue colormap with n colors.
function cmap = redblue(n)
    if nargin < 1
        n = 256;
    end
    cmap = redbluecmap(11);
    cmap = imresize(cmap, [n, 3]); % original color map contain 11 colors
    cmap = min(max(cmap, 0), 1);