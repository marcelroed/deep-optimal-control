function [Problem]=HBVPdef(C)

Problem.C = C;
Problem.sigma = @(x) tanh(x);
Problem.dsigma = @(x) 1./(cosh(x).^2);
Problem.eta = @(x) exp(x)./(exp(x)+1);
Problem.deta = @(x) exp(x)./(1+exp(x)).^2;
Problem.DJW = @(YN,W,mu,C) YN'*(Problem.deta(YN*W+mu).*(Problem.eta(YN*W+mu)-C));
Problem.DJmu = @(YN,W,mu,C) Problem.deta(YN*W+mu)'*(Problem.eta(YN*W+mu)-C);

Problem.DJY = @(Y,W,mu,C) (Problem.deta(Y*W+mu) .* (Problem.eta(Y*W+mu)-C))*W';

Problem.Vf = @(Y,K,b) Problem.sigma(Y*K+b);
Problem.AdjVf = @(Y,P,K,b) -(P .* Problem.dsigma(Y*K+b))*K';

Problem.DVfK = @(Y,P,K,b) Y'*(Problem.dsigma(Y*K+b) .* P);
Problem.DVfb = @(Y,P,K,b) sum(Problem.dsigma(Y*K+b) .* P);



    