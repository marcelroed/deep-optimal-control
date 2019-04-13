function S=RKforwardstepper(Ctrls,M,HBVP)


nlayers = Ctrls.nlayers;
channels = Ctrls.channels;
h = Ctrls.stepsize;


S=SolutionDef(h, M.s, Ctrls.Y0, channels, nlayers);
last = nlayers+1;
% Forward stepping
F=cell(M.s,1);
for n=1:nlayers
    for i=1:M.s
       S.Ys{n,i}=S.Y{n};
       for j=1:i-1
           S.Ys{n,i}=S.Ys{n,i}+h*M.A(i,j)*F{j};
       end
       F{i} = HBVP.Vf(S.Ys{n,i},Ctrls.K{n},Ctrls.b{n});
    end
    S.Y{n+1}=S.Y{n};
    for i=1:M.s
        S.Y{n+1}=S.Y{n+1}+h*M.w(i)*F{i};
    end
end

S.Classifier = HBVP.eta(S.Y{last}*Ctrls.W+Ctrls.mu);
end