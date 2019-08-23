classdef ControlDef
    properties
        Y0  % Matrix (N,channels) of initial data
        channels % Number of channels at least equal to cols
        rows  % Number of data points
        cols  % Degrees of freedom in each initial data point, col dim of Y0
        nlayers % Number of steps taken by the integrator (elements of cell arrays K and b)
        K % Cell array of matrices containing control parameters
        b % Cell array of row vectors containing control biases
        W % Array (channels,1) containing the control parameters of the final projection
        mu % scalar bias for final projection (classifier)
        stepsize
    end
     methods
        function Ctrls=ControlDef(Idata,stepsize)
            [N,d]=size(Idata.Y0);
            Ctrls.Y0 = [Idata.Y0 zeros(N,Idata.channels-d)]; %Store also extra channels with zeros
            Ctrls.channels=Idata.channels;
            Ctrls.nlayers = Idata.nlayers;
            Ctrls.rows = N;
            Ctrls.cols = d;
            Ctrls.K = Idata.K;
            Ctrls.b = Idata.b;
            Ctrls.W = Idata.W;
            Ctrls.mu = Idata.mu;
            Ctrls.stepsize = stepsize;
        end
    end
end
