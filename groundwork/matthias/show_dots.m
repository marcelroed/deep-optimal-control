% This file can be used to reproduce all figures in
%?Benning, Celledoni, Ehrhardt, Owren & Schönlieb (2019). 
% Deep learning as optimal control problems: models and 
% numerical methods. http://arxiv.org/abs/1904.05657
%
% It creates a scatter plot of the points with spatial 
% coordinates x and y.
function show_dots(x, y, is_test)
    if nargin < 3
        is_test = false;
    end

    if is_test
        scatter(x(1,:), x(2,:), 10, y, 'filled')
    else
        scatter(x(1,:), x(2,:), 10, y, 'filled')
    end