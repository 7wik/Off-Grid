function [f_est,w_est,CS] = CS_grid(y_f,phi,sigma_w,L)
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

%% Joint estimation

cvx_begin sdp quiet
cvx_precision low
variable ws(L,1) complex
variable z(N,1) complex

minimize(lambda*sum(abs(ws)) + z'*z/2)

z == y_f - C*ws;

cvx_end

%%
CS.f = [];
CS.amp = [];

m=0;
for l=1:L
    if norm(ws(l))>=2e-2
        m = m+1;
        CS.f(m) = f_grid(l);
        CS.amp(m) = norm(ws(l));
    end
end

f_est = CS.f;
w_est = CS.amp;

