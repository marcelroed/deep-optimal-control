% This file can be used to reproduce all figures in
%?Benning, Celledoni, Ehrhardt, Owren & Schönlieb (2019). 
% Deep learning as optimal control problems: models and 
% numerical methods. http://arxiv.org/abs/1904.05657
classdef ODENet < matlab.mixin.Copyable
    % network relation
    % x^{k+1} = alpha^k x^k + h^k * link( K^k x^k + b^k )
    
    % cost function
    % cost = factor * fit(hypothesis( W x^{K+1} + eta ), data)
    
    properties
        dim
        nlayers
        K
        b
        W
        mu
        dt
        alpha
        link
        link_derivative
        hypothesis
        hypothesis_derivative
        fit
        fit_derivative
        factor
    end
    
    methods
        function this = ODENet(dim, nlayers, K, b, W, mu, dt, alpha, link, ...
                link_derivative, hypothesis, hypothesis_derivative, ...
                fit, fit_derivative, factor)
            
            this.nlayers = nlayers;
            this.dim = dim;
                        
            if nargin < 15 || isempty(factor)
                this.factor = 1;
            else
                this.factor = factor;
            end
            
            if nargin < 14 || isempty(fit_derivative)
                [~, this.fit_derivative] = get_cross_entropy();
            else
                this.fit_derivative = fit_derivative;
            end
            
            if nargin < 13 || isempty(fit)
                [this.fit, ~] = get_cross_entropy();
            else
                this.fit = fit;
            end
            
            if nargin < 12 || isempty(hypothesis_derivative)
                this.hypothesis_derivative = ...
                    @(x) 1./(1 + exp(-x)) .* (1 - 1./(1 + exp(-x)));
            else
                this.hypothesis_derivative = hypothesis_derivative;
            end
            
            if nargin < 11 || isempty(hypothesis)
                this.hypothesis = @(x) 1./(1 + exp(-x));
            else
                this.hypothesis = hypothesis;
            end
            
            if nargin < 10 || isempty(link_derivative)
                this.link_derivative = ...
                    @(x) 1./(cosh(x).^2);
            else
                this.link_derivative = link_derivative;
            end
            
            if nargin < 9 || isempty(link)
                this.link = @(x) tanh(x);
            else
                this.link = link;
            end
                        
            if nargin < 8 || isempty(alpha)
                this.alpha = zeros(1, nlayers);
                for k = 1 : nlayers
                    this.alpha(k) = 1;
                end
            else
                this.alpha = alpha;
            end
            
            if nargin < 7 || isempty(dt)
                this.dt = zeros(1, nlayers);
                for k = 1 : nlayers
                    this.dt(k) = 1 / nlayers;
                end
            else
                this.dt = dt;
            end
            
            if nargin < 6 || isempty(mu)
                this.eta = zeros(1);
            else
                this.mu = mu;
            end
            
            if nargin < 5 || isempty(W)
                this.W = zeros(1, dim);
            else
                this.W = W;
            end
            
            if nargin < 4 || isempty(b)
                for k = 1 : nlayers
                    this.b{k} = zeros(dim, 1);
                end
            else
                this.b = b;
            end
            
            if nargin < 3 || isempty(K)
                for k = 1 : nlayers
                    this.K{k} = zeros(dim);
                end
            else
                this.K = K;
            end
            
        end
         
        function [prediction, y, a] = forward(this, features)
            
            N = this.nlayers;
            
            y = cell(1, N+2);
            a = cell(1, N+1);
            
            y{1} = features;
            
            for k = 1 : N
                a{k} = this.K{k} * y{k} + this.b{k};
                y{k+1} = this.alpha(k) * y{k} + this.dt(k) * this.link(a{k});
            end
            
            a{N+1} = this.W * y{N+1} + this.mu;
            y{N+2} = this.hypothesis(a{N+1});
            
            prediction = y{N+2};
        end
        
        function grad = gradient(this, labels, y, a)
            N = this.nlayers;
            
            grad = copy(this);
            
            p = cell(1, N+1);
            
            rho = this.hypothesis_derivative(a{N+1}) .* ...
                this.factor .* this.fit_derivative(y{N+2}, labels);
            
            grad.W = rho * y{N+1}';
            grad.mu = sum(rho);
            
            gamma = cell(1, N);
                        
            for k = N : -1 : 1
                switch k
                    case N
                        p{N} = this.W' * rho;
                    otherwise
                        p{k} = this.alpha(k+1) * p{k+1} ...
                            + this.dt(k+1) * this.K{k+1}' * gamma{k+1};
                end
                
                gamma{k} = this.link_derivative(a{k}) .* p{k};
                
                grad.K{k} = this.dt(k) * gamma{k} * y{k}';
                grad.b{k} = this.dt(k) * sum(gamma{k}, 2);
                
                grad.dt(k) = sum(sum(p{k} .* this.link(a{k})));
                grad.alpha(k) = sum(sum(p{k}));
            end
        end
        
        function F = objective(this, prediction, labels)     
            F = this.factor * this.fit(prediction, labels);
        end
    end
end