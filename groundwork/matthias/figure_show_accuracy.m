% This file can be used to reproduce all figures in
%?Benning, Celledoni, Ehrhardt, Owren & Schönlieb (2019). 
% Deep learning as optimal control problems: models and 
% numerical methods. http://arxiv.org/abs/1904.05657

main_folder = {'pics'};
folders = {'example_donut1d_20layers_backtracking1', ...
    'example_donut2d_20layers_backtracking1', ...
    'example_spiral2d_20layers_backtracking1', ...
    'example_squares2d_20layers_backtracking1'};
filenames = ...
    {{'Net_squared_seed1', ...
    'ResNet_squared_seed1', ...
    'ODENet_squared_seed1', ...
    'ODENetSimplex_squared_seed1'}};
legends = {{'Net', 'ResNet', 'ODENet','ODENet+Simplex'}};
filename_out = {'seed1_accuracy'};

for i = 1 : length(folders)
    figure(1); clf    
    filenames_i = filenames{1};
    
    for j = 1 : length(filenames_i)
        load([main_folder{1} '/' folders{i} '/' filenames_i{j} '.mat']);
        ax = gca;
        ax.ColorOrderIndex = j;
        semilogx(accuracy * 100, '-', 'Linewidth', 2)
        hold on;
    end

    for j = 1 : length(filenames_i)
        load([main_folder{1} '/' folders{i} '/' filenames_i{j} '.mat']);
        ax = gca;
        ax.ColorOrderIndex = j;
        semilogx(accuracy_val * 100, '--', 'Linewidth', 2)
    end
    
    xlabel('gradient descent iteration');
    ylabel('accuracy [%]');
    xlim([1, length(function_values)+1])
    ylim([0.4, 1] * 100)    
    set(gca,'FontSize', 20)    
    legend(legends{1}, 'Location', 'Southeast')
    saveas(gcf, [main_folder{1} '/' folders{i} '/' folders{i} '_' filename_out{1} '.png']);
end