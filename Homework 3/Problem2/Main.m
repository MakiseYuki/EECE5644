clear all; close all;
% x in, y comes out, y is the value from 0 to 1
% w is vector, v is scaler, train the model from the given x
% two class --- positive and negitive
% x as data process to the classifier out put y
% y(x) = P(label = positive|x)

N = 999;
mu = [1 -1;1 -1];
Sigma(:,:,1) = 0.1*[15 2;3 2];
Sigma(:,:,2) = 0.1*[6 7;3 17]; % Sigma should be no diagonal with distinct eigenvalues
classPriors = [0.3,0.7]; thr = [0,cumsum(classPriors)];
d = size(mu,1);
u = rand(1,N); x = zeros(d,N); labels = zeros(1,N);

for m = 1:length(classPriors)
    ind = find(thr(m)<u & u<=thr(m+1));     
    x(:,ind) = randGaussian(length(ind),mu(:,m),Sigma(:,:,m));
    labels(:,ind) = repmat(m,1,length(ind)); % label 1 is negtive 2 is positive
end

figure(1),
plot(x(1,find(labels==1)),x(2,find(labels==1)),'b.'); hold on,
plot(x(1,find(labels==2)),x(2,find(labels==2)),'r.');
title('True Data of two classes')
legend('Class 1 with prior 0.3', 'Class 2 with prior 0.7');

% The following is the process of LDA

x1_indices = find(labels==1);
x2_indices = find(labels==2);
x1 = x(:,x1_indices); 
x2 = x(:,x2_indices);

N1 = length(x1);
N2 = length(x2);

mu1hat = mean(x1,2); S1hat = cov(x1');
mu2hat = mean(x2,2); S2hat = cov(x2');

Sb = (mu1hat-mu2hat)*(mu1hat-mu2hat)';
Sw = S1hat + S2hat;

[V,D] = eig(inv(Sw)*Sb);
[~,indicesFDA] = sort(diag(D),'descend');
w = V(:,indicesFDA(1)); % Fisher LDA projection vector
b = V(:,indicesFDA(1));
dis_1 = w'*x1 + b;
dis_2 = w'*x2 + b;
dis_1 = sum(dis_1);
dis_2 = sum(dis_2);
% 
figure(2),
subplot(2,1,1), 
plot(x1(1,:),x1(2,:),'r*');
hold on;
plot(x2(1,:),x2(2,:),'bo');
title('True data of two classes');
legend('Class 1 with prior 0.3', 'Class 2 with prior 0.7');
axis equal,

subplot(2,1,2), 
plot(dis_1(1,:),zeros(1,N1),'r*');
hold on;
plot(dis_2(1,:),zeros(1,N2),'bo');
title('Fisher LDA projection');
legend('Class 1 with prior 0.3', 'Class 2 with prior 0.7');
axis equal,

% Train the parameter of y(x) = 1/(1+exp(dis_1,dis_2))

y1 = 1./(1 + exp(dis_1)); %negative
y2 = 1./(1 + exp(dis_2)); %positive : modeling 


fun = @(z)-sum(sum(log(1./(1 + exp(z(1)'*x2 + z(2))))));
z0 = [w b];
z = fminsearch(fun,z0);

% 
% newW = [z(1,1,1);z(1,1,2)];
% newB = [z(1,2,1);z(1,2,2)];

dis_t1 = (z(:,1).*x1 + z(:,2))/N1;
dis_t2 = (z(:,1).*x2 + z(:,2))/N2;
dis_t1 = sum(dis_t1);
dis_t2 = sum(dis_t2);

% for i = 1:d
%     
%     t_FDA = find(dis_2(i,:) < 0);
%     t_traning = find(dis_t2(i,:)<0);
%     traning_error = length(t_traning)/N2;
%     errorFDA = length(t_FDA)/N2;
%     
%     figure(3),
%     semilogy(i,traning_error,'r+'), hold on,
%     semilogy(i,errorFDA,'b*'), hold on,
%     axis equal
% end





