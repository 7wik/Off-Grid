avg_ista_off_norm = [0.9977, 0.8606,0.6462,0.6046,0.2545,0.1716,0.0780,0.1616,0.1934,0.1884,0.1877,0.1840,0.1617,0.1717,0.1657,0.3379,0.3430,0.3447,0.3415,0.7977,0.7977,0.8740,0.8740,0.874,0.8740];
annet_norms = [0.65281193, 1.01407925, 0.8470729, 3.19937127, 3.35067937, 0.49751333];
annet_len = length(annet_norms)
ista_len = length(avg_ista_off_norm)
ista_x = 1:1:ista_len
annet_x = 1:1:annet_len
figure;plot(ista_x,avg_ista_off_norm,'--bo');hold on;plot(annet_x,annet_norms,'--rs');
legend('ISTA off grid','AN-NET')

