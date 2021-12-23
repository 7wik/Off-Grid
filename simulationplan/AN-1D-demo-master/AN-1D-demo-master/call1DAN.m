function [x,tau_cvx,amp_cvx] = call1DAN(y,b,lambda)

N = length(b);
iterator = 0;
cvx_begin sdp quiet
variable x(N,1) complex;
variable v(1,1) hermitian;
variable w(N,1) complex;
variable t;
variable u(N-1,1) complex;

minimize((abs(t)/2 + trace(v)/2)*lambda + w'*w/2);
% disp(iterator)
% disp("++++++++++++++++")
% iterator = iterator+1;

[toeplitz([t;u]) x;
    x' v] >= 0;

w == y - diag(b)*x;

cvx_end
% disp("done!!!!!!!!!!!!!!!")

tol = 1e-1;
thresh = 2*pi/N/4;

for l = 1:1
    [poles{l}, coeffs{l}] = callProny(x,tol);
    nPole(l) = length(poles{l});
end

[pole_comb, coeff_comb] = combinePole(poles,coeffs,1,thresh);

channel.poles = poles;
channel.coeffs = coeffs;

tau_cvx = 1-pole_comb(1,:)/2/pi;
for l=1:size(coeff_comb,2)
    amp_cvx(l) = norm([coeff_comb(:,l)]);
end
