function F = objective(Ctrls, Method,HBVP)
%
% Compute the objective function   F=1/2 * || eta(YN*W+mu) - C ||^2
%
% The RKforwardstepper obtains YN and the classifier eta(YN*W+mu)


S1=RKforwardstepper(Ctrls,Method,HBVP);

F= 1/2 * sum( (S1.Classifier-HBVP.C).^2 );


