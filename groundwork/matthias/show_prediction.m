% This file can be used to reproduce all figures in
%?Benning, Celledoni, Ehrhardt, Owren & Schönlieb (2019). 
% Deep learning as optimal control problems: models and 
% numerical methods. http://arxiv.org/abs/1904.05657
%
% It creates an image with the predicted density in the background
% and the training data as a scatter plot ontop.
function show_prediction(network, features, labels, domain, fignum)  
    if nargin < 5
        fignum = 1;
    end

    n = 300;
    r1 = linspace(domain(1), domain(2), n);
    r2 = linspace(domain(3), domain(4), n);
    [x1, x2] = meshgrid(r1, r2);
    x = [x1(:)'; x2(:)'];
    prediction = network.forward(x);
    y = 1 * reshape(prediction, n, []);

    figure(fignum); clf;
    imagesc(r1, r2, y); hold on;
    show_dots(features, 0.3 + 0.4 * labels);
    colormap(redblue);
    colorbar;
    caxis([0, 1]);