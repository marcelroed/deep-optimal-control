function snaps(dataset,rkmethod,channels,nlayers,ndata,seed)


dirname=sprintf('LearnedData-New-s%d',seed);
plotsdir = sprintf('RKSnaps-New-s%d',seed);
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

S = RKforwardstepper(Ctrls,Method,HBVP);

n=300;
r1 = linspace(-1.5, 1.5, n);
r2 = linspace(-1.5, 1.5, n);
[x1, x2] = meshgrid(r1, r2);
X0=[x1(:),x2(:)];
Ne = size(X0,1);
Xe=[X0 zeros(Ne,channels-2)];
Ctest=Ctrls;
Ctest.Y0=Xe; Ctest.rows=Ne;
%S_pred = RKforwardstepper(Ctest,Method,HBVP);
transpred = HBVP.eta(Xe*Ctrls.W+Ctrls.mu);
z = 1 * reshape(transpred, n, []);




for k=1:nlayers+1
    figure(k), hold off
    imagesc(r1, r2, z); hold on; 
    colormap(redblue(200));
    caxis([0, 1]);
    Ysnap=S.Y{k};
    show_dots(Ysnap',0.3+0.4*C');
    axis off
    drawnow
    print([plotname,'_',num2str(k)],'-dpng');
end