% This file can be used to reproduce all figures in
%?Benning, Celledoni, Ehrhardt, Owren & Schönlieb (2019). 
% Deep learning as optimal control problems: models and 
% numerical methods. http://arxiv.org/abs/1904.05657
%
% Returns samples from a checkerboard data set. Inspired by
% https://playground.tensorflow.org.
function [features, labels] = get_data_squares_2d(n_samples) 
    if nargin < 1
        n_samples = 1000;
    end
    
    a = 0.7;
    features = 2 * a * rand([2, n_samples]) - a;    
    labels = features(1,:) .* features(2,:) > 0;        
    features = features + .1 * randn(size(features));
    
    if nargout < 1
        figure()
        show_dots(features, labels);
        colormap(redblue);
        colorbar;
        caxis([0, 1]);
    end