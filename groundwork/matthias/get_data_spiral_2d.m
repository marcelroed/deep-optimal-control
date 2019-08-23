% This file can be used to reproduce all figures in
%?Benning, Celledoni, Ehrhardt, Owren & Schönlieb (2019). 
% Deep learning as optimal control problems: models and 
% numerical methods. http://arxiv.org/abs/1904.05657
%
% Returns samples from a spiral data set. Inspired by
% https://playground.tensorflow.org.
function [features, labels] = get_data_spiral_2d(n_samples) 
    if nargin < 1
        n_samples = 1000;
    end
    
    m1 = ceil(n_samples / 2);
    m2 = n_samples - m1;
    
    n_turns = 1;
    
    phi1 = pi;
    r1 = linspace(.1,1,m1);
    a1 = linspace(.1, 2*pi * n_turns, m1);    
    d1 = [r1 .* cos(a1+phi1); r1 .* sin(a1+phi1)];
    
    phi2 = mod(phi1 + pi, 2*pi);
    r2 = linspace(.1,1,m2);
    a2 = linspace(.1, 2*pi * n_turns, m2);
    d2 = [r2 .* cos(a2+phi2); r2 .* sin(a2+phi2)];    
    
    features = [d1, d2];  
    
    labels = [1 * ones(1, m1), 0 * ones(1, m2)];
    i = randperm(n_samples);
    features = features(:, i);
    labels = labels(i);
    
    features = features + .05 * randn(size(features));
    
    if nargout < 1
        figure()
        show_dots(features, labels);
        colormap(redblue);
        colorbar;
        caxis([0, 1]);
    end