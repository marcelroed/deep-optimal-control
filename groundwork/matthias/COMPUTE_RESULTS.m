% This file can be used to reproduce all figures in
%?Benning, Celledoni, Ehrhardt, Owren & Schönlieb (2019). 
% Deep learning as optimal control problems: models and 
% numerical methods. http://arxiv.org/abs/1904.05657
%
% This file runs examples on various data sets and other settings.
domain = [-1, 1, -1, 1];
plot_every_iterate = 0;

%fits = {'squared', 'crossentropy'};
fits = {'squared'};
%datasets = {'donut1d', 'donut2d', 'spiral2d', 'trivial', 'squares2d'};
datasets = {'donut1d', 'donut2d', 'spiral2d', 'squares2d'};
%seeds = [1, 2];
seeds = [1];
backtrackings = [true];
% layers = [5, 20, 100];
layers = [20];
networks = {'Net', 'ResNet', 'ODENet', 'ODENetSimplex'};

accuracy_fun = @(x, y) mean(round(2 .* x .* y - y - x + 1));

for fit = fits
    switch fit{1}
        case 'squared'
            [fit_fun, fit_der] = get_least_squares();    
        case 'crossentropy'
            [fit_fun, fit_der] = get_cross_entropy();
    end

    for seed = seeds
        for dataset = datasets
            
            rng(seed);
            switch dataset{1}
                case 'donut1d'
                    [features, labels] = get_data_donut_1d(500);
                    [features_val, labels_val] = get_data_donut_1d(500);
                    niter = 5000;
                case 'donut2d'
                    [features, labels] = get_data_donut_2d(1000);
                    [features_val, labels_val] = get_data_donut_2d(1000);
                    niter = 8000;
                case 'spiral2d'
                    [features, labels] = get_data_spiral_2d(1000);
                    [features_val, labels_val] = get_data_spiral_2d(1000);
                    niter = 20000;
                case 'squares2d'
                    [features, labels] = get_data_squares_2d(1000);
                    [features_val, labels_val] = get_data_squares_2d(1000);
                    niter = 8000;
                case 'trivial'
                    [features, labels] = get_data_trivial();
                    [features_val, labels_val] = get_data_trivial();
                    niter = 5000;
            end
            
            figure(1); clf;
            show_dots(features, labels)
            colormap(redblue)
            
            for layer = layers
                for backtracking = backtrackings
                    
                    dim = size(features, 1);
                    ndata = size(features, 2);
                                      
                    folder_out = ['./pics/example_' dataset{1} ...
                        '_' num2str(layer) 'layers_backtracking' num2str(backtracking)];
                    mkdir(folder_out);
                    save_images = true;
                    
                    for inet = 1 : length(networks)
                        rng(seed);
                        prefix = [folder_out '/' networks{inet} '_' fit{1} '_seed' num2str(seed)];
                                                
                        networks{inet}
                        
                        switch networks{inet}
                            case 'Net'
                                K = cell(layer, 1);
                                b = cell(layer, 1);
                                for k = 1 : layer
                                    K{k} = randn(2);
                                    b{k} = zeros(2,1);
                                end
                                W = randn(1,2);
                                W = [10, 10];
                                mu = 0;
                                dt = ones(1, layer);
                                alpha = zeros(1, layer);
                                network = ODENet(dim, layer, K, b, W, mu, dt, ...
                                    alpha, [], [], [], [], fit_fun, fit_der, 1 / ndata);
                                
                            case {'ResNet', 'ODENet', 'ODENetSimplex'}
                                K = cell(layer, 1);
                                b = cell(layer, 1);
                                for k = 1 : layer
                                    K{k} = randn(2);
                                    b{k} = zeros(2,1);
                                end
                                W = randn(1,2);
                                W = [10, 10];
                                mu = 0;
                                network = ODENet(dim, layer, K, b, W, mu, [], ...
                                    [], [], [], [], [], ...
                                    fit_fun, fit_der, 1 / ndata);
                        end
                        
                        lipschitz = .1;
                            
                        bt_end = 200;
                        bt_up = 1.5;
                        bt_down = 0.99;
                        
                        function_values = nan(niter+1, 1);
                        function_values_val = nan(niter+1, 1);
                        
                        accuracy = nan(niter+1, 1);
                        accuracy_val = nan(niter+1, 1);
                        
                        lipschitz_constants = nan(niter+1, 1);
                        iterates = cell(niter+1, 1);
                        
                        [prediction, y, a] = network.forward(features); % not needed due to defaults in "objective"
                        function_values(1) = network.objective(prediction, labels);
                        accuracy(1) = accuracy_fun(prediction, labels);
                        [prediction_val, ~, ~] = network.forward(features_val); % not needed due to defaults in "objective"
                        function_values_val(1) = network.objective(prediction_val, labels_val);
                        accuracy_val(1) = accuracy_fun(prediction_val, labels_val);
                        lipschitz_constants(1) = lipschitz;
                        iterates{1} = copy(network);
                        
                        show_prediction(network, features, labels, domain, 1)
                        
                        if save_images
                            saveas(gcf, [prefix '_prediction_init.png']);
                        end
                        
                        show_prediction_linear(network, features, labels, 2*domain, [], 2)
                        
                        if save_images
                            saveas(gcf, [prefix '_transformed_init.png']);
                        end
                        
                        network_old = copy(network);
                        
                        for outer_iter = 1 : niter
                            
                            % compute gradient
                            grad = network.gradient(labels, y, a);
                            
                            for bt_iter = 1 : bt_end
                                
                                inner = 0;
                                norm = 0;
                                for k = 1 : length(network.K)
                                    network.K{k} = network_old.K{k} - 1 / lipschitz * grad.K{k};
                                    network.b{k} = network_old.b{k} - 1 / lipschitz * grad.b{k};
                                    
                                    dk = network.K{k} - network_old.K{k};
                                    db = network.b{k} - network_old.b{k};
                                    inner = inner + sum(sum(grad.K{k} .* dk))+ sum(sum(grad.b{k} .* db));
                                    norm = norm + sum(sum(dk.^2)) + sum(sum(db.^2));
                                end
                                
                                dw = network.W - network_old.W;
                                dmu = network.mu - network_old.mu;
                                inner = inner + sum(sum(grad.W .* dw))+ sum(sum(grad.mu .* dmu));
                                norm = norm + sum(sum(dw.^2)) + sum(sum(dmu.^2));
                                
                                if strcmp(networks{inet}, 'ODENet')
                                    network.dt = network_old.dt - 1 / lipschitz * grad.dt;
                                    
                                    ddt = network.dt - network_old.dt;
                                    inner = inner + sum(sum(grad.dt .* ddt));
                                    norm = norm + sum(sum(ddt.^2));
                                end
                                
                                if strcmp(networks{inet}, 'ODENetSimplex')
                                    network.dt = proj_simplex(network_old.dt - 1 / lipschitz * grad.dt);
                                    
                                    ddt = network.dt - network_old.dt;
                                    inner = inner + sum(sum(grad.dt .* ddt));
                                    norm = norm + sum(sum(ddt.^2));
                                end
                                
                                [prediction, y, a] = network.forward(features);
                                new_function_value = network.objective(prediction, labels);
                                
                                if backtracking
                                    fprintf('!')
                                    
                                    if new_function_value <= function_values(outer_iter) + inner + lipschitz / 2 * norm
                                        lipschitz = lipschitz * bt_down;
                                        fprintf('\n')
                                        break
                                    else
                                        if bt_iter == bt_end
                                            fprintf('\n')
                                        end
                                        lipschitz = lipschitz * bt_up;
                                    end
                                else
                                    fprintf('\n')
                                    break
                                end
                                
                            end
                            
                            network_old = copy(network);
                            
                            function_values(outer_iter+1) = new_function_value;
                            accuracy(outer_iter+1) = accuracy_fun(prediction, labels);
                        
                            [prediction_val, ~, ~] = network.forward(features_val);
                            function_values_val(outer_iter+1) = network.objective(prediction_val, labels_val);
                            accuracy_val(outer_iter+1) = accuracy_fun(prediction_val, labels_val);
                        
                            lipschitz_constants(outer_iter+1) = lipschitz;
                            
                            if mod(outer_iter, 100) == 0
                                iterates{outer_iter+1} = copy(network);
                            end
                            
                            fprintf('iter:%i, obj:%3.5e, lipschitz:%3.5e', ...
                                outer_iter, function_values(outer_iter), lipschitz)
                            
                            if mod(outer_iter, plot_every_iterate) == 0
                                show_prediction(network, features, labels, [-1, 1, -1, 1])
                                pause(0.1)
                            end
                        end
                        
                        figure(1); clf;
                        hold off;
                        yyaxis left
                        loglog(function_values);
                        hold on;
                        ylabel('function value');
                        loglog(function_values_val);
                        legend('training', 'test');
                        yyaxis right
                        loglog(lipschitz_constants)
                        ylabel('estimated Lipschitz constant');
                        
                        if save_images
                            saveas(gcf, [prefix '_stepsizes.png']);
                        end
                        
                        figure(1); clf;
                        hold off;
                        loglog(accuracy);
                        hold on;
                        ylabel('accuracy');
                        loglog(accuracy_val);
                        legend('training', 'test');
                        
                        if save_images
                            saveas(gcf, [prefix '_accuracy.png']);
                        end
                        
                        show_prediction(network, features, labels, domain)
                        
                        if save_images
                            saveas(gcf, [prefix '_prediction.png']);
                        end
                        
                        show_prediction_linear(network, features, labels, 2*domain)
                        
                        if save_images
                            saveas(gcf, [prefix '_transformed.png']);
                        end
                        
                        if save_images
                            show_prediction_video(network, features, labels, 2*domain, [prefix '_video'])
                        end
                        
                        if save_images
                            save([prefix '.mat'], 'function_values', 'function_values_val', 'accuracy', 'accuracy_val', 'lipschitz_constants', 'iterates', 'network')
                        end
                        
                        if save_images                  
                            figure(1)
                            clf()
                            hold on;
                            K = length(iterates);
                            cmap = colormap(parula(K));
                            for k = 1 : K
                                if ~isempty(iterates{k})
                                    plot(iterates{k}.dt, 'Color', cmap(k, :), 'Linewidth', 2)
                                end
                            end
                            ylabel('ODE time step');
                            xlabel('layer');
                            saveas(gcf, [prefix '_hs.png']);
                        end
                    end
                    
                    if save_images
                        figure(1)
                        clf
                        hold on;
                            
                        for inet = 1 : length(networks)
                            prefix = [folder_out '/' networks{inet} '_' fit{1} '_seed' num2str(seed)];
                            load([prefix '.mat'])

                            plot(network.dt, 'Linewidth', 2)
                            ylabel('ODE time step');
                            xlabel('layer');
                        end
                        legend(networks)
                        saveas(gcf, [folder_out '/' fit{1} '_seed' num2str(seed) '_comparison_h.png']);
                    end
                    
                    if save_images
                        figure(2)
                        hold off;
                        
                        for inet = 1 : length(networks)
                            prefix = [folder_out '/' networks{inet} '_' fit{1} '_seed' num2str(seed)];
                            load([prefix '.mat'])
                            
                            ax = gca;
                            ax.ColorOrderIndex = inet;
                            loglog(function_values, '-', 'Linewidth', 2)
                            hold on;
                            ylabel('objective value');
                            xlabel('iteration');
                        end
                        legend(networks)
                        
                        for inet = 1 : length(networks)
                            prefix = [folder_out '/' networks{inet} '_' fit{1} '_seed' num2str(seed)];
                            load([prefix '.mat'])
                            
                            ax = gca;
                            ax.ColorOrderIndex = inet;
                            loglog(function_values_val, '--', 'Linewidth', 2)
                        end
                        xlim([1, niter+1])
                        saveas(gcf, [folder_out '/' fit{1} '_seed' num2str(seed) '_comparison_objective.png']);
                    end
                end
            end
        end
    end
end