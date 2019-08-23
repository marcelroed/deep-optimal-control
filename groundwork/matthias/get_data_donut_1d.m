% This file can be used to reproduce all figures in
%?Benning, Celledoni, Ehrhardt, Owren & Schönlieb (2019). 
% Deep learning as optimal control problems: models and 
% numerical methods. http://arxiv.org/abs/1904.05657
%
% Returns samples from a 1D donut data set.
%
% It uses the function "pinky" and "gendist", 
% copyright (c) 2012 and 2016, Tristan Ursell
% https://uk.mathworks.com/matlabcentral/fileexchange/35797-generate-random-numbers-from-a-2d-discrete-distribution
function [features, labels, problem] = get_data_donut_1d(n_samples, nx) 
    if nargin < 1
        n_samples = 1000;
    end
    if nargin < 2
        nx = 100;
    end

    z1 = linspace(-1, 1, nx);
    z2 = 0;
    [r1, r2] = meshgrid(z1, z2);
    problem.domain = {z1, z1};
    rad = @(r1, r2) sqrt(r1.^2 + r2.^2);

    mean_rad1 = 0;
    sigma1 = 0.1;
    density1 = exp(-(rad(r1, r2) - mean_rad1).^2 / (2 * sigma1^2));
    density1 = density1 / sum(density1(:));
    problem.densities{1} = density1;

    mean_rad2 = 0.5;
    sigma2 = 0.1;
    density2 = exp(-(rad(r1, r2) - mean_rad2).^2 / (2 * sigma2^2));
    density2 = density2 / (sum(density2(:)));
    problem.densities{2} = density2;


    m1 = ceil(n_samples / 3);
    m2 = n_samples - m1;

    features = zeros(2, n_samples);

    for i = 1 : m1
        [features(1, i), features(2, i)] = pinky(z1, z2, density1);
    end

    for i = m1+1 : n_samples
        [features(1, i), features(2, i)] = pinky(z1, z2, density2);
    end

    labels = [1 * ones(1, m1), 0 * ones(1, m2)];
    i = randperm(n_samples);

    features = features(:, i);
    labels = labels(i);
    features = features + .1 * randn(size(features));

    if nargout < 1
        figure()
        show_dots(features, labels);
        colormap(redblue);
        colorbar;
        caxis([0, 1]);
    end


%% AUX FUNCTIONS
    function [x0,y0]=pinky(Xin,Yin,dist_in,varargin)
    %create column distribution and pick random number
    col_dist=sum(dist_in,1);

    %pick column distribution type
    if nargin==3
        %if no res parameter, simply update X/Yin2
        col_dist=col_dist/sum(col_dist);
        Xin2=Xin;
        Yin2=Yin;
    else
        %generate new, higher res input vectors
        Xin2=linspace(min(Xin),max(Xin),round(res*length(Xin)));
        Yin2=linspace(min(Yin),max(Yin),round(res*length(Yin)));

        %generate interpolated column-sum distribution
        col_dist=interp1(Xin,col_dist,Xin2,'pchip');

        %check to make sure interpolated values are positive
        if any(col_dist<0)
            col_dist=abs(col_dist);
            warning('Interpolation generated negative probability values.')
        end
        col_dist=col_dist/sum(col_dist);
    end

    %generate random value index
    ind1=gendist(col_dist,1,1);

    %save first value
    x0=Xin2(ind1);

    %find corresponding indices and weights in the other dimension
    [val_temp,ind_temp]=sort((x0-Xin).^2);

    if val_temp(1)<eps %if we land on an original value
        row_dist=dist_in(:,ind_temp(1));
    else %if we land inbetween, perform linear interpolation
        low_val=min(ind_temp(1:2));
        high_val=max(ind_temp(1:2));

        Xlow=Xin(low_val);
        Xhigh=Xin(high_val);

        w1=1-(x0-Xlow)/(Xhigh-Xlow);
        w2=1-(Xhigh-x0)/(Xhigh-Xlow);

        row_dist=w1*dist_in(:,low_val) + w2*dist_in(:,high_val);
    end

    %pick column distribution type
    if nargin==3
        row_dist=row_dist/sum(row_dist);
    else
        row_dist=interp1(Yin,row_dist,Yin2,'pchip');
        row_dist=row_dist/sum(row_dist);
    end

    %generate random value index
    ind2=gendist(row_dist,1,1);

    %save first value
    y0=Yin2(ind2);

    
    function T = gendist(P,N,M,varargin)
    if size(P,1)>1
        P=P';
    end

    %normalize P
    Pnorm=[0 P]/sum(P);

    %create cumlative distribution
    Pcum=cumsum(Pnorm);

    %create random matrix
    N=round(N);
    M=round(M);
    R=rand(1,N*M);

    %calculate T output matrix
    V=1:length(P);
    [~,inds] = histc(R,Pcum); 
    T = V(inds);

    %shape into output matrix
    T=reshape(T,N,M);