function [labels,data] = generateTrueGMM(N)
% Generate samples from a 4-component GMM that is the true data
alpha_true = [0.3,0.2,0.2,0.3];
mu_true = [-20 -20 20 20;-20 20 20 -20];
Sigma_true(:,:,1) = [3 1;1 12];
Sigma_true(:,:,2) = [12 1;1 2];
Sigma_true(:,:,3) = [7 1;1 13];
Sigma_true(:,:,4) = [15 1;1 7];
[labels,data] = randGMM(N,alpha_true,mu_true,Sigma_true);

%[d,M] = size(mu_true); % determine dimensionality of samples and number of GMM components
end


