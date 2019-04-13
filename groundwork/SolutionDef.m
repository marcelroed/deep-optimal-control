classdef SolutionDef
    properties
        channels % Number of channels at least equal to cols
        rows  % Number of data points
        cols  % Degrees of freedom in each initial data point, col dim of Y0
        nlayers % Number of steps taken by the integrator
        stepsize % Stepsize of integrator
        Y  % Cell array (nlayers+1,1) holding the Y-values for all steps
        Ys % Cell array (nlayers,nstages) holding all stage values for all steps
        P  % Cell array - dual solution (nlayers+1,1) for all steps
        fPs %  Cell array (nlayers,nstages) - stages of dual solution (derivatives)
        Classifier
    end
    methods
        function S=SolutionDef(stepsize, nstages, Y0, channels, nlayers)
            [N,d]=size(Y0);
            S.channels=channels;
            S.stepsize = stepsize;
            S.nlayers = nlayers;
            S.rows = N;
            S.cols = d;
            S.Y = cell(nlayers+1,1);
            S.Y{1} = [Y0, zeros(N,channels-d)];
            S.Ys = cell(S.nlayers,nstages);
            S.P = cell(S.nlayers+1,1);
            S.fPs = cell(S.nlayers,nstages);
           
        end
    end
end
