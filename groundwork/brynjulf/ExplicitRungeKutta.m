classdef ExplicitRungeKutta
    properties
        s=1;
        A=0;
        w=1;
        At=0;
        wt=1;
        name='Euler';
    end
    methods
        function M = ExplicitRungeKutta(A,w,name)
            if nargin > 2
                M.name = name;
            else
                M.name = 'noname';
            end
            M.A = A;
            M.w = w;
            M.s = size(A,1);
            if M.s <= 0
                error('number of stages must be positive')
            end
            if norm(triu(A))>0
                error('method must be explicit')
            end
            if min(abs(w))==0
                error('all weights must be non-zero')
            end
            M.wt = M.w;
            for i=1:M.s
                for j=1:M.s
                    M.At(i,j)=w(j)/w(i)*A(j,i);
                end
            end
        end
        
            
    end
end
        