function show_prediction(dataset,rkmethod,channels,nlayers,ndata,seed)
%
%
%
dirname=sprintf('LearnedData-New-s%d',seed);
plotsdir = sprintf('RKfigs-New-s%d/init-transf',seed);
if ~exist(dirname,'dir')
    error('Directory %s does not exist\n',dirname)
end

if ~exist(plotsdir,'dir')
    mkdir(plotsdir)
end

basename = sprintf('Ds=%s-M=%s-ch=%d-nl=%d-nd=%d',dataset,rkmethod,channels,nlayers,ndata);
datafilename = [basename,'.mat'];
fullname = [dirname,'/',datafilename];
plotname = [plotsdir,'/',basename];

if ~exist(fullname,'file')
    error('Could not find datafile %s\n',fullname);
end

load(fullname,'Ctrls','Method','HBVP');
C=HBVP.C;
n=300;
r1 = linspace(-1.5, 1.5, n);
r2 = linspace(-1.5, 1.5, n);
[x1, x2] = meshgrid(r1, r2);
X0=[x1(:),x2(:)];
Ne = size(X0,1);
Xe=[X0 zeros(Ne,channels-2)];
Ctest=Ctrls;
Ctest.Y0=Xe; Ctest.rows=Ne;
S_pred = RKforwardstepper(Ctest,Method,HBVP);
Pred = S_pred.Classifier;
y = 1 * reshape(Pred, n, []);

figure(1);
imagesc(r1, r2, y); hold on; 
colormap(redblue(200));
caxis([0, 1]);
show_dots(Ctrls.Y0',0.3+0.4*C');
axis off
drawnow
movegui(gcf,'west')
hold off
print([plotname,'_ini'],'-dpng');
                
               
figure(3)
S = RKforwardstepper(Ctrls,Method,HBVP);
transpred = HBVP.eta(Xe*Ctrls.W+Ctrls.mu);
z = 1 * reshape(transpred, n, []);
%fprintf('Plotting third figure ...') 
imagesc(r1, r2, z); hold on; 
colormap(redblue(200));
%colorbar;
caxis([0, 1]);
last=nlayers+1;
Ytransformed=S.Y{last};
show_dots(Ytransformed',0.3+0.4*C');
axis off
movegui(gcf,'east')
drawnow
print([plotname,'_transf'],'-dpng');

end