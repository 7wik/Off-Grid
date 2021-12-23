function [f_est,w_est,ISTA] = ISTA_grid(y_f,phi,sigma_w,L)
%% CS algorithm /Joint estimation
i = sqrt(-1);
N = length(y_f);
C = zeros(N,L);

f_grid = [0:1/L:1-1/L];
for l=1:L
    xs = exp(-i*2*pi*f_grid(l)*[0:1:N-1]');
    C(:,l) = phi.*xs;
end

lambda = 10/sqrt(2)*sigma_w*sqrt(2*log(L));

%% ISTA

ws = lasso_ista(y_f,C,lambda);

%%
ISTA.f = [];
ISTA.amp = [];

m=0;
for l=1:L
    if norm(ws(l))>=2e-2
        m = m+1;
        ISTA.f(m) = f_grid(l);
        ISTA.amp(m) = norm(ws(l));
    end
end

f_est = ISTA.f;
w_est = ISTA.amp;

