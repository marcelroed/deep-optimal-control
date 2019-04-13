function [C1,alpha,normGsq]=backtracking(Ctrls,Gradient,Method,HBVP,tau)

rho=0.5;
c=0.4;
alpha=tau/rho;
nlayers = Ctrls.nlayers;

fk=objective(Ctrls, Method,HBVP);

normGsq = sum( (Gradient.W).^2 )+ Gradient.mu^2;
for k=1:nlayers
    normGsq=normGsq + sum(sum(Gradient.K{k}.^2)) + Gradient.b{k}*Gradient.b{k}';
end

fnew = 2*fk;

C1=Ctrls; % Make sure that C1 is set at least once
while fnew > fk - alpha*c*normGsq
    C1=Ctrls;
    alpha=rho*alpha;
    for k=1:nlayers
        C1.K{k}=Ctrls.K{k}-alpha*Gradient.K{k};
        C1.b{k}=Ctrls.b{k}-alpha*Gradient.b{k};
    end
    C1.W = Ctrls.W - alpha*Gradient.W;
    C1.mu = Ctrls.mu - alpha*Gradient.mu;
    fnew = objective(C1,Method,HBVP);
end

    
    

