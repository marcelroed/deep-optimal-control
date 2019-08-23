% 
% This script learns parameters of various models and stores them to files
%
channels = 2;
ndata=1000;
seed=4656;
ndata_f=1000;
nlayers_f=50;
channels_f=2;


h=0.1;
tau=0.1;
tau_max=10;
niter=10000; % This is the max number of iterations, but may stop earlier if no progress is made
nsavedit=niter; % 1<=nsavedit<=niter, the number of saved iterations from the convergence process,


% The initial data is generated earlier, typically by gen_and_save_data.m
% and here we read them from file
inidir = 'InitialData-2';
dsnames={'get_data_squares_2d','get_data_donut_2d','get_data_spiral_2d','get_data_donut_1d'};

iname1=sprintf('%s/%s_s%dnl%dch%dnd%d.mat',inidir,dsnames{1},seed,nlayers_f,channels_f,ndata);
iname2=sprintf('%s/%s_s%dnl%dch%dnd%d.mat',inidir,dsnames{2},seed,nlayers_f,channels_f,ndata);
iname3=sprintf('%s/%s_s%dnl%dch%dnd%d.mat',inidir,dsnames{3},seed,nlayers_f,channels_f,ndata);
iname4=sprintf('%s/%s_s%dnl%dch%dnd%d.mat',inidir,dsnames{4},seed,nlayers_f,channels_f,ndata);

dirname=sprintf('LearnedData-New-s%d',seed);
if ~exist(dirname, 'dir')
   mkdir(dirname)
end




% Define some explicit Runge-Kutta schemes
A_eu = 0;
w_eu = 1;
A_ie = [0,0;1,0];
w_ie = [0.5; 0.5];
A_kutta4 = [0,0,0,0;0.5,0,0,0;0,0.5,0,0;0,0,1,0];
w_kutta4 = 1/6*[1;2;2;1];
A_kutta3 = [0,0,0;1/2,0,0;-1,2,0];
w_kutta3 = [1/6;2/3;1/6];
A_ssprk3 = [0,0,0;1,0,0;1/4,1/4,0];
w_ssprk3 = [1/6;1/6;2/3];
A_ralston3 = [0,0,0; 1/2,0,0; 0 3/4 0];
w_ralston3 = 1/9*[2;3;4];



% Run through all three types of data
for dataset =   {'donut1d','squares2d','donut2d','spiral2d'}
    fprintf('Starting dataset %s\n',dataset{1})
%% 
        for nlayers = 15
                fprintf('Starting nlayers=%d\n',nlayers)        
           
            
            
            for rkmethod = {'Euler', 'ImprovedEuler', 'kutta3','kutta4'}

                switch rkmethod{1}
                    case 'Euler'
                        Method=ExplicitRungeKutta(A_eu,w_eu,rkmethod{1});
                    case 'ImprovedEuler'
                        Method=ExplicitRungeKutta(A_ie,w_ie,rkmethod{1});
                    case 'ssprk3'
                        Method=ExplicitRungeKutta(A_ssprk3,w_ssprk3,rkmethod{1});
                    case 'kutta3'
                        Method=ExplicitRungeKutta(A_kutta3,w_kutta3,rkmethod{1});
                    case 'kutta4'
                        Method=ExplicitRungeKutta(A_kutta4,w_kutta4,rkmethod{1});
                end
                fprintf('Starting RK method %s\n',rkmethod{1})    
                F=zeros(niter,1);
                Fn=zeros(niter,1);
                iter=0;
                alpha=tau/2;
                switch dataset{1}
                    case 'squares2d'
                        Idata=InitialDef(iname1,nlayers,channels,ndata);
                    case 'donut2d'
                        Idata=InitialDef(iname2,nlayers,channels,ndata);
                    case 'spiral2d'
                        Idata=InitialDef(iname3,nlayers,channels,ndata);
                    case 'donut1d'
                        Idata=InitialDef(iname4,nlayers,channels,ndata);
                end
                Y0=Idata.Y0;
                C = Idata.C;
                HBVP=HBVPdef(C);
                Ctrls=ControlDef(Idata,h);
                for iter=1:niter
                    Gradient=GradientCalc(Ctrls, Method, HBVP);                   
                    [Ctrls,alpha,normGsq]=backtracking(Ctrls,Gradient,Method,HBVP,min([tau_max,2*alpha]));
                    F(iter,1)=objective(Ctrls, Method, HBVP);
                    Fn(iter,1)=sqrt(normGsq);
                    if mod(iter,2000)==0 || iter==1
                        fprintf('\nMethod=%s, nlayers=%d, dataset=%s\n',rkmethod{1},nlayers,dataset{1})
                        fprintf('Iteration no: %d\n',iter)
                        fprintf('Value of alpha: %7.4f\n',alpha);
                        fprintf('Residual: %8.4f\n',F(iter,1))
                        fprintf('Norm Grad: %8.4e\n',Fn(iter,1));
                    end
                end
                F=F(1:iter,1);
                Fn=Fn(1:iter,1);
                intv=round(iter/nsavedit);
                Sit = (1:intv:iter)';
                F_res_it = F(Sit);
                F_grad_it=Fn(Sit);
                outfilename = sprintf('Ds=%s-M=%s-ch=%d-nl=%d-nd=%d.mat',dataset{1},rkmethod{1},channels,nlayers,ndata);
                save([dirname,'/',outfilename],'Ctrls','Method','HBVP','Sit','F_res_it','F_grad_it');
                fprintf('\nSaved to file %s\n',outfilename)  
            end
            
        end
end