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