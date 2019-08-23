% This file can be used to reproduce all figures in
%?Benning, Celledoni, Ehrhardt, Owren & Schönlieb (2019). 
% Deep learning as optimal control problems: models and 
% numerical methods. http://arxiv.org/abs/1904.05657
datasets = {'donut1d', 'donut2d', 'spiral2d', 'trivial', 'squares2d'};

for dataset = datasets
    rng(seed);
    switch dataset{1}
        case 'donut1d'
            [features, labels] = get_data_donut_1d(500);
            [features_val, labels_val] = get_data_donut_1d(500);
            niter = 8000;
        case 'donut2d'
            [features, labels] = get_data_donut_2d(1000);
            [features_val, labels_val] = get_data_donut_2d(1000);
            niter = 8000;
        case 'spiral2d'
            [features, labels] = get_data_spiral_2d(1000);
            [features_val, labels_val] = get_data_spiral_2d(1000);
            niter = 30000;
        case 'squares2d'
            [features, labels] = get_data_squares_2d(1000);
            [features_val, labels_val] = get_data_squares_2d(1000);
            niter = 30000;
        case 'trivial'
            [features, labels] = get_data_trivial();
            [features_val, labels_val] = get_data_trivial();
            niter = 5000;
    end
    
    figure(1); clf;
    show_dots(features, 0.3 + 0.4*labels);
    colormap(redblue);
    colorbar;
    caxis([0, 1]);
    axis([-1, 1, -1, 1]);
    
    filename = ['./pics/example_' dataset{1} '.png'];    
    saveas(gcf, filename)
end