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
    {{'ResNet_squared_seed1', ...
    'ODENet_squared_seed1', ...
    'ODENetSimplex_squared_seed1'}};
legends = {{'ResNet', 'ODENet', 'ODENet+Simplex'}};
filename_out = {'seed1_timesteps'};

for i = 1 : length(folders)
    figure(1); clf
    colorOrder = get(gca, 'ColorOrder');
    filenames_i = filenames{1};
    
    for j = 1 : length(filenames_i)
        load([main_folder{1} '/' folders{i} '/' filenames_i{j} '.mat']);        
        ax = gca;
        ax.ColorOrderIndex = j;                          
        plot(network.dt, '-o', 'Linewidth', 2, 'Color', colorOrder(j+1, :))   
        hold on;
    end
    
    set(gca,'FontSize',20)    
    xlabel('time step / layer');
    ylabel('\Delta t^{[j]}');
    axis tight;    
    legend(legends{1}, 'Location', 'Southeast')
    saveas(gcf, [main_folder{1} '/' folders{i} '/' folders{i} '_' filename_out{1} '.png']);
end