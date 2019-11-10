function [Alpha,Mu,Sigma] = EMofGMM(N_Gaussian,N_sample,dim,data)
    regWeight = 1e-10; % regularization parameter for covariance estimates
    delta = 1e-5; % tolerance for EM stopping criterion

    Alpha = ones(1,N_Gaussian)/N_Gaussian;
    shuffledIndices = randperm(N_sample); % indices = ceil(N_sample*rand(1,N_Gaussian))
    Mu = data(:,shuffledIndices(1:N_Gaussian));% Mu = data(:,indices)% set initial Mu estimates
    [~,assignedCentroidLabels] = min(pdist2(Mu',data'),[],1);

    for m = 1:N_Gaussian % set initial Covariance estimates using sample covariances of initial assignments as 
        Sigma(:,:,m) = cov(data(:,find(assignedCentroidLabels==m))') + regWeight*eye(dim,dim);
    end
    MLE = zeros(N_Gaussian,1);
    converged_flag = 0;
    while ~converged_flag
        %E step: calculate the posterior
        for i=1:N_Gaussian
            tmp(i,:) = repmat(Alpha(i),1,N_sample).*evalGaussian(data,Mu(:,i),Sigma(:,:,i));
        end
        plgivenx = tmp./sum(tmp,1);%??
        alpha_new = mean(plgivenx,2);
        denominator = plgivenx./repmat(sum(plgivenx,2),1,N_sample);
        mu_new = data*denominator';
        for i = 1:N_Gaussian
            v = data-repmat(mu_new(:,i),1,N_sample);
            u = repmat(denominator(i,:),dim,1).*v;
            Sigma_new(:,:,i) = u*v' + regWeight*eye(dim,dim); % adding a small regularization term
        end
%         Dalpha = sum(abs(alpha_new-Alpha));
%         Dmu = sum(sum(abs(mu_new-Mu)));
%         DSigma = sum(sum(abs(abs(Sigma_new-Sigma))));
%         converged_flag = ((Dalpha+Dmu+DSigma)<delta); % Check if converged
        for i=1:N_Gaussian
            MLE_new(i) = sum(log(evalGaussian(data,mu_new(:,i),Sigma_new(:,:,i))),2);
        end
        converged_flag = (sum(MLE_new-MLE')<delta);
        MLE = MLE_new;
        Alpha = alpha_new; Mu = mu_new; Sigma = Sigma_new;% update parameters
    end
%     %draw contour of GMM
%     figure(N_Gaussian),
%     if size(data,1)==2
%         plot(data(1,:),data(2,:),'b.'); 
%         xlabel('x_1'), ylabel('x_2'), title('Data and Estimated GMM Contours'),
%         axis equal, hold on;
%         rangex1 = [1.2*min(data(1,:)),1.2*max(data(1,:))];
%         rangex2 = [1.2*min(data(2,:)),1.2*max(data(2,:))];
%         [x1Grid,x2Grid,zGMM] = contourGMM(Alpha,Mu,Sigma,rangex1,rangex2);
%         contour(x1Grid,x2Grid,zGMM,20); axis equal, 
%     end
end


function g = evalGaussian(x,mu,Sigma)
    % Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
    [n,N] = size(x);
    invSigma = inv(Sigma);
    C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
    E = -0.5*sum((x-repmat(mu,1,N)).*(invSigma*(x-repmat(mu,1,N))),1);
    g = C*exp(E);
end


function gmm = evalGMM(x,alpha,mu,Sigma)
    gmm = zeros(1,size(x,2));
    for m = 1:length(alpha) % evaluate the GMM on the grid
        gmm = gmm + alpha(m)*evalGaussian(x,mu(:,m),Sigma(:,:,m));
    end
end


function [x1Grid,x2Grid,zGMM] = contourGMM(alpha,mu,Sigma,rangex1,rangex2)
    x1Grid = linspace(floor(rangex1(1)),ceil(rangex1(2)),1001);
    x2Grid = linspace(floor(rangex2(1)),ceil(rangex2(2)),901);
    [h,v] = meshgrid(x1Grid,x2Grid);
    GMM = evalGMM([h(:)';v(:)'],alpha, mu, Sigma);
    zGMM = reshape(GMM,901,1001);
end