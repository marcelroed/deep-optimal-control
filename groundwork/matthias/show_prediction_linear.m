% This file can be used to reproduce all figures in
%?Benning, Celledoni, Ehrhardt, Owren & Schönlieb (2019). 
% Deep learning as optimal control problems: models and 
% numerical methods. http://arxiv.org/abs/1904.05657
%
% It creates an image with the linear classification in the background
% and the transformed training data as a scatter plot ontop.
function show_prediction_linear(network, features, labels, ...
    domain, frame, fignum)
    if nargin < 6
        fignum = 1;
    end

    if nargin < 5 || isempty(frame)
        frame = network.nlayers+1;
    end

    n = 300;
    r1 = linspace(domain(1), domain(2), n);
    r2 = linspace(domain(3), domain(4), n);
    [x1, x2] = meshgrid(r1, r2);
    x = [x1(:)'; x2(:)'];
    [~, z] = network.forward(features);
    lin_class = @(x) network.hypothesis(network.W * x + network.mu);
    prediction = lin_class(x);
    y = 1 * reshape(prediction, n, []);

    figure(fignum); clf;
    imagesc(r1, r2, y); hold on;
    show_dots(z{frame}, 0.3 + 0.4 * labels);
    colormap(redblue);
    colorbar;
    caxis([0, 1]);