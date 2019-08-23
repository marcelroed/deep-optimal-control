% This file can be used to reproduce all figures in
%?Benning, Celledoni, Ehrhardt, Owren & Schönlieb (2019). 
% Deep learning as optimal control problems: models and 
% numerical methods. http://arxiv.org/abs/1904.05657
% 
% Returns a function handle to the least squares data fit and its derivative.
function [fun, der] = get_least_squares()
    fun = @(x, y) 1/2 * sum((x(:) - y(:)).^2);
    der = @(x, y) x - y;