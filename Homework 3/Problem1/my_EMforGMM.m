function [mu,Sigma,alpha,bad_choise] = my_EMforGMM(componentGauss,data)
% Generates N samples from a specified GMM,
% then uses EM algorithm to estimate the parameters
% of a GMM that has the same number of components
% as the true GMM that generates the samples.

close all,
delta = 1e-5; % tolerance for EM stopping criterion
regWeight = 1e-10; % regularization parameter for covariance estimates
d = length(data(:,1));
x = data;
N = length(data); 
%Initialize the GMM to randomly selected samples
alpha = ones(1,componentGauss)/componentGauss;
shuffledIndices = randperm(N);
mu = x(:,shuffledIndices(1:componentGauss)); % pick number of Gauss conponent random samples as initial mean estimates
[~,assignedCentroidLabels] = min(pdist2(mu',x'),[],1); % assign each sample to the nearest mean

for m = 1:componentGauss % use sample covariances of initial assignments as initial covariance estimates
    Sigma(:,:,m) = cov(x(:,find(assignedCentroidLabels==m))') + regWeight*eye(d,d);
end

LearningEstimate = zeros(componentGauss,1);
t = 0; %displayProgress(t,x,alpha,mu,Sigma);

Converged = 0; % Not converged at the beginning
bad_choise = 0;
while ~Converged && ~bad_choise
    for l = 1:componentGauss
        temp(l,:) = repmat(alpha(l),1,N).*evalGaussian(x,mu(:,l),Sigma(:,:,l));
    end
    plgivenx = temp./sum(temp,1);
    alphaNew = mean(plgivenx,2);
    w = plgivenx./repmat(sum(plgivenx,2),1,N);
    muNew = x*w';
    for l = 1:componentGauss
        v = x-repmat(muNew(:,l),1,N);
        u = repmat(w(l,:),d,1).*v;
        SigmaNew(:,:,l) = u*v' + regWeight*eye(d,d); % adding a small regularization term
    end
     
%     Dalpha = sum(abs(alphaNew-alpha'));
%     Dmu = sum(sum(abs(muNew-mu)));
%     DSigma = sum(sum(abs(abs(SigmaNew-Sigma))));
%     Converged = ((Dalpha+Dmu+DSigma)<delta); % Check if converged
    
    for i = 1:componentGauss
            LearningEstimate_new(i) = sum(log(evalGaussian(x,muNew(:,i),SigmaNew(:,:,i))),2);
    end
    
    Converged = (abs(sum(sum(LearningEstimate_new-LearningEstimate'))) < delta);
    observ = abs(sum(sum(LearningEstimate_new-LearningEstimate')));
    LearningEstimate = LearningEstimate_new;
    %plot(t,observ,'g.'); %show the value if it is Converged
    alpha = alphaNew; mu = muNew; Sigma = SigmaNew;
    t = t+1; 
    
    if t > 10000 % set the learning loop in an restriction that if have chosen a bad starting point.
        bad_choise = 1;
    end
    displayProgress(t,x,alpha,mu,Sigma,componentGauss);

end

function displayProgress(t,x,alpha,mu,Sigma,componentGauss)
figure(1),

if size(x,1)==2
    linespec = {'b.','r.','g.','y.','c.','k.'};
    subplot(1,2,1), cla,
    plot(x(1,:),x(2,:),linespec{componentGauss}); 
    xlabel('x_1'), ylabel('x_2'), title('Data and Estimated GMM Contours'),
    axis equal, hold on;
    rangex1 = [min(x(1,:)),max(x(1,:))];
    rangex2 = [min(x(2,:)),max(x(2,:))];
    [x1Grid,x2Grid,zGMM] = contourGMM(alpha,mu,Sigma,rangex1,rangex2);
    contour(x1Grid,x2Grid,zGMM); axis equal, 
    subplot(1,2,2), 
end

logLikelihood = sum(log(evalGMM(x,alpha,mu,Sigma)));
plot(t,logLikelihood,linespec{componentGauss}); hold on,
xlabel('Iteration Index'), ylabel('Log-Likelihood of Data'),
drawnow; %pause(0.1),




