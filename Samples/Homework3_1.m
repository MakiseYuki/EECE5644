%x_10 = GenerateGMM(2,10);
[label_100,x_100]  = GenerateGMM(2,100);
[label_1000,x_1000] = GenerateGMM(2,1000);
[label_10000,x_10000] = GenerateGMM(2,10000);

K = 10;
MSE = zeros(2,6);

for N_Gaussian = 1:6
    MSE_train = 0;
    MSE_validation = 0;
    N = size(x_100,2);
    
    Seperation = ceil(linspace(0,N,K+1));
    fold_start_end = zeros(K,2);
    for k = 1:K
        fold_start_end(k,:) = [Seperation(k)+1,Seperation(k+1)];
    end
    
    for k=1:10
        [x_train,y_train,x_validation,y_validation] = KFoldCrossValidation(fold_start_end,K,k,x_100,label_100);
        [Alpha_estim,Mu_estim,Sigma_estim] = EMofGMM(N_Gaussian,size(x_train,2),2,x_train);
        label_trian_estim = GetLabel(Alpha_estim,Mu_estim,Sigma_estim,x_train,N_Gaussian);
        label_validation_estim = GetLabel(Alpha_estim,Mu_estim,Sigma_estim,x_validation,N_Gaussian);
        
        MSE_train = MSE_train+calculateMSE(y_train,label_trian_estim);
        MSE_validation = MSE_train+calculateMSE(y_validation,label_validation_estim);
    end
    MSE(:,N_Gaussian)=[MSE_train./10;MSE_validation./10];
end
figure(1), clf,
semilogy(MSE(1,:),'bo'); hold on; semilogy(MSE(2,:),'rx');
xlabel('Number of Gaussian Component'); ylabel(strcat('MSE estimate with ',num2str(K),'-fold cross-validation'));
legend('Training MSE','Validation MSE');


function label_estim = GetLabel(Alpha,Mu,Sigma,X,N_label)
    for i = 1:N_label
        y_estim(i,:) = Alpha(i)*log(evalGaussian(X,Mu(:,i),Sigma(:,:,i)));
    end
    label_tmp = max(y_estim,[],1);
    label_estim = zeros(1,size(X,2));
    for i = 1:size(X,2)
        [label_estim(i), ~]= find(y_estim(:,i) == label_tmp(i));
    end 
end


function [x_train,y_train,x_validation,y_validation] = KFoldCrossValidation(seperate_range,K,k,x_data,y_data)
    data_validation = [seperate_range(k,1):seperate_range(k,2)];
    N = size(x_data,2);
    if k == 1
        data_train = [seperate_range(k,2)+1:N];
    elseif k == K
        data_train = [1:seperate_range(k,1)-1];
    else
        data_train = cat(2,[1:seperate_range(k,1)-1],[seperate_range(k,2)+1:N]);
    end
    x_train = x_data(:,data_train); 
    y_train = y_data(:,data_train); 
    x_validation = x_data(:,data_validation); 
    y_validation = y_data(:,data_validation); 
end


function MSE = calculateMSE(y_true,y_estim)
    MSE = mean((y_true-y_estim).^2);
end


function [labels,x] = GenerateGMM(dim,Nsample)
    alpha_true = [0.20,0.23,0.27,0.3];
    mu_true = [10,10,-10,-10;10,-10,10,-10];
    Sigma_true(:,:,1) = [3 1;1 20];
    Sigma_true(:,:,2) = [7 1;1 2];
    Sigma_true(:,:,3) = [4 1;1 16];
    Sigma_true(:,:,4) = [11 1;6 10];

    alpha_tmp=[0,cumsum(alpha_true)];
    u = rand(1,Nsample); x = zeros(dim,Nsample); labels = zeros(1,Nsample);
    for m = 1:4
        ind = find(alpha_tmp(m)<=u & u<alpha_tmp(m+1));
        z =  randn(dim,length(ind));
        A = Sigma_true(:,:,m)^(1/2);
        x(:,ind)=A*z + repmat(mu_true(:,m),1,length(ind));
        labels(:,ind)=repmat(m,1,length(ind));
    end
end

function g = evalGaussian(x,mu,Sigma)
    % Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
    [n,N] = size(x);
    invSigma = inv(Sigma);
    C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
    E = -0.5*sum((x-repmat(mu,1,N)).*(invSigma*(x-repmat(mu,1,N))),1);
    g = C*exp(E);
end