function [pole_comb, coeff_comb] = combinePole(poles,coeffs,K,thresh)

pole_set = [];
coeff_set = [];

for k=1:K
    for m = 1:length(poles{k})
        indicator = 0;
        for n = 1:size(pole_set,2)
            if abs(poles{k}(m)-pole_set(k,n))<thresh
                pole_set(k,n) = poles{k}(m);
                coeff_set(k,n) = coeffs{k}(m);
                indicator = 1;
            end
        end
        if indicator ==0
            pole_set = [pole_set,poles{k}(m)*ones(K,1)];
            coeff_add = zeros(K,1);
            coeff_add(k) = coeffs{k}(m);
            coeff_set = [coeff_set, coeff_add];
        end
    end
end

for k=1:K
    for m = 1:size(pole_set,2)
        pole_comb(:,m) = abs(coeff_set(:,m)/sum(abs(coeff_set(:,m))))'*pole_set(:,m);
    end
end
coeff_comb = coeff_set;