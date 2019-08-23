% This file can be used to reproduce all figures in
%?Benning, Celledoni, Ehrhardt, Owren & Schönlieb (2019). 
% Deep learning as optimal control problems: models and 
% numerical methods. http://arxiv.org/abs/1904.05657
%
% Get trivial data set as described in [1].
%?[1] C. F. Higham and D. J. Higham, ?Deep Learning: An Introduction for 
% Applied Mathematicians,? ?arXiv:1801.05894v1, 2018.
function [features, labels] = get_data_trivial() 
    features = [0.1 0.3 0.1 0.6 0.4 0.6 0.5 0.9 0.4 0.7; ...
                0.1 0.4 0.5 0.9 0.2 0.3 0.6 0.2 0.4 0.6];
    labels = [1 1 1 1 1 0 0 0 0 0];
        
    if nargout < 1
        figure()
        show_dots(features, labels);
        colormap(redblue);
        colorbar;
        caxis([0, 1]);
    end