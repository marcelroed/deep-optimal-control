% This file can be used to reproduce all figures in
%?Benning, Celledoni, Ehrhardt, Owren & Schönlieb (2019). 
% Deep learning as optimal control problems: models and 
% numerical methods. http://arxiv.org/abs/1904.05657
%
% Add an ambient or "augmented" dimension to features x.
function z = add_ambient_dimension(x, ambient_dim)
    z = zeros(size(x, 1) + ambient_dim, size(x, 2));
    z(1: size(x, 1), :) = x;