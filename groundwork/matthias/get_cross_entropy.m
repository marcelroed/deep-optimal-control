% This file can be used to reproduce all figures in
%?Benning, Celledoni, Ehrhardt, Owren & Schönlieb (2019). 
% Deep learning as optimal control problems: models and 
% numerical methods. http://arxiv.org/abs/1904.05657
% 
% Returns a function handle tp the cross entropy data fit and its derivative.
function [fun, der] = get_cross_entropy(eps)
    if nargin < 1
        eps = 1e-6;
    end

    fun = @(x, y) - sum(sum((1 + y).*log(1 + x + eps) + (1 - y).*log(1 - x + eps) - 2 * log(2)));
    der = @(x, y) - (1 + y)./(1 + x + eps) + (1 - y)./(1 - x + eps);