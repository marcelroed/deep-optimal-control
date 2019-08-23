% This file can be used to reproduce all figures in
%?Benning, Celledoni, Ehrhardt, Owren & Schönlieb (2019). 
% Deep learning as optimal control problems: models and 
% numerical methods. http://arxiv.org/abs/1904.05657
% 
% It projects inout x onto the unit simplex using [1].
% The implementation is inspired by [2]
%?[1] J. Duchi, S. Shalev-Shwartz, Y. Singer, and T. Chandra, 
% ?Efficient Projections onto the L1 -ball for Learning in High 
% dimensions,? in Proceedings of the 25th International Conference
% on Machine Learning - ICML, 2008, pp. 272?279.
%?[2] L. Bungert, D. A. Coomes, M. J. Ehrhardt, J. Rasch, R. Reisenhofer, 
% and C.-B. Schönlieb, ?Blind Image Fusion for Hyperspectral Imaging
% with the Directional Total Variation,? Inverse Probl., vol. 34, 
% no. 4, p. 044003, 2018.
function px = proj_simplex(x)
    shape_x = size(x);
    x = x(:);
    n = length(x);
    mu = sort(x, 'descend');
    crit = mu - (1./(1:n)') .* (cumsum(mu)-1);
    rho = find(crit>0, 1, 'last');
    theta = 1/rho*(sum(mu(1:rho))-1);
    
    px = max(x-theta,0);
    px = reshape(px, shape_x);