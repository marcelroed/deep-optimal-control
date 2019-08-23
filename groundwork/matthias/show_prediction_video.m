% This file can be used to reproduce all figures in
%?Benning, Celledoni, Ehrhardt, Owren & Schönlieb (2019). 
% Deep learning as optimal control problems: models and 
% numerical methods. http://arxiv.org/abs/1904.05657
%
% It creates a video of frames, each of which are shown similar to
% "show_prediction_linear". I.e. a frame is an image with the linear 
% classification in the background and the transformed training data 
% as a scatter plot ontop.
function show_prediction_video(network, features, labels, domain, ...
    filename, fignum)
    if nargin < 6
        fignum = 1;
    end

    v = VideoWriter(filename, 'MPEG-4');

    nframes = network.nlayers+1;
    totaltime = 3;
    v.FrameRate = nframes / totaltime;
    open(v);

    n = 300;
    r1 = linspace(domain(1), domain(2), n);
    r2 = linspace(domain(3), domain(4), n);
    [x1, x2] = meshgrid(r1, r2);
    x = [x1(:)'; x2(:)'];
    [~, z] = network.forward(features);
    lin_class = @(x) network.hypothesis(network.W * x + network.mu);
    prediction = lin_class(x);
    y = 1 * reshape(prediction, n, []);

    for k = 1 : nframes
        figure(fignum); clf;
        imagesc(r1, r2, y); hold on;
        show_dots(z{k}, 0.3 + 0.4 * labels);
        colormap(redblue);
        colorbar;
        caxis([0, 1]);
        set(gca, 'nextplot', 'replacechildren');
        frame = getframe(gcf);
        writeVideo(v,frame);
    end

    close(v);