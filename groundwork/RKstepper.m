function S = RKstepper(Ctrls,M,HBVP)
%
% Computes a forward and backward sweep for a given set of control
% variables
%
% Input
%  Ctrls: Structure with all control variables (and initial data)
%  M: Structure with all parameters defining the RK method
% HBVP: Structure with all functions/vector fields necessary to define the
%       boundary value problem. Contains also the labels.
%


last = Ctrls.nlayers+1;
S=RKforwardstepper(Ctrls,M,HBVP); %Step forward
S.P{last} = HBVP.DJY(S.Y{last},Ctrls.W,Ctrls.mu,HBVP.C); %Set right boundary value for P
S=RKbackwardstepper(Ctrls, M, HBVP, S); % Step backwards to obtain the Ps


