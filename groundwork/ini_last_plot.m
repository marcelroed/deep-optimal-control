% Ini-last plots
channels=2;
nlayers=15;
ndata=1000;
seed=4656;

for dataset = {'donut1d','donut2d','spiral2d','squares2d'}
    
    for rkmethod = {'Euler','ImprovedEuler','kutta3','kutta4'}

        show_prediction(dataset{1},rkmethod{1},channels,nlayers,ndata,seed)
    end
end