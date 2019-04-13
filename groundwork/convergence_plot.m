% Script to generate convergence plots (function_values)

nlayers=15;
channels=2;
ndata=1000;
seed=4656;

dirname=sprintf('LearnedData-New-s%d',seed);
plotsdir = sprintf('RKfigs-New-s%d/convergence',seed);
if ~exist(plotsdir,'dir')
    mkdir(plotsdir)
end


meths = {'Euler','ImprovedEuler','kutta3','kutta4'};

dsets = {'donut1d','donut2d','spiral2d','squares2d'};

legends = {{'Resnet/Euler', 'ImprovedEuler', 'Kutta(3)','Kutta(4)'}};


for dataset = dsets
    figure(1), hold off
    cindex=1;
    for rkmethod = meths
        basename = sprintf('Ds=%s-M=%s-ch=%d-nl=%d-nd=%d',dataset{1},rkmethod{1},channels,nlayers,ndata);
        datafilename = [basename,'.mat'];
        fullname = [dirname,'/',datafilename];
        plotname = [plotsdir,'/',basename];
        if ~exist(fullname,'file')
            error('Could not find datafile %s\n',fullname);
        end
        load(fullname,'Ctrls','Method','HBVP','Sit','F_res_it','F_grad_it');
        if cindex == 1
            scalefac = F_res_it(1);
        end
        ax = gca;
        ax.ColorOrderIndex = cindex;
%        loglog(F_res_it / F_res_it(1), '-', 'LineWidth', 2)
%        loglog(F_res_it/scalefac, '-', 'LineWidth', 2)
        loglog(F_res_it/ndata, '-', 'LineWidth', 2)

        hold on;
        cindex=cindex+1;
    end
    set(gca,'FontSize',20)
    
    xlabel('gradient descent iteration');
    ylabel('function value');
    xlim([1, length(F_res_it)+1])
    
    legend(legends{1}, 'Location', 'SouthWest')
    outfilename = sprintf('%s/%s_nl%d',plotsdir,dataset{1},nlayers);
    print([outfilename,'_conv'],'-dpng');
end



