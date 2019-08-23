function G=GradientCalc(Ctrls, M, HBVP)
%
%
%

S = RKstepper(Ctrls, M, HBVP);

channels = Ctrls.channels;
nlayers = Ctrls.nlayers;
C=HBVP.C;
h=S.stepsize;
s = M.s;

last = nlayers+1;
G.K = cell(nlayers,1);
G.b = cell(nlayers,1);
G.W = HBVP.DJW(S.Y{last},Ctrls.W,Ctrls.mu,C);
G.mu = HBVP.DJmu(S.Y{last},Ctrls.W,Ctrls.mu,C);


E = cell(s,1);

for k=1:last-1
    G.K{k}=zeros(channels);
    G.b{k}=zeros(1,channels);
    for i=1:s
        G.K{k} = G.K{k}+h*M.w(i)*HBVP.DVfK(S.Ys{k,i},S.P{k+1},Ctrls.K{k},Ctrls.b{k});
        G.b{k} = G.b{k}+h*M.w(i)*HBVP.DVfb(S.Ys{k,i},S.P{k+1},Ctrls.K{k},Ctrls.b{k});
        E{i}= -HBVP.AdjVf(S.Ys{k,i},S.P{k+1},Ctrls.K{k},Ctrls.b{k});
    end
    
    for m=1:s-1
       for i=1:s-m
           P=zeros(S.rows,channels);
           for j=i+1:s-m+1
               P=P+h*M.At(i,j)*E{j};
           end
           G.K{k} = G.K{k} + h*M.w(i)*HBVP.DVfK(S.Ys{k,i},P,Ctrls.K{k},Ctrls.b{k});
           G.b{k} = G.b{k} + h*M.w(i)*HBVP.DVfb(S.Ys{k,i},P,Ctrls.K{k},Ctrls.b{k});
           E{i} =  -HBVP.AdjVf(S.Ys{k,i},P,Ctrls.K{k},Ctrls.b{k});
       end             
    end              
end



    