clear all;
close all;

%Sample 1
n = 2;
N = 400;
mu(:,1) = [0;0];
mu(:,2) = [3;3];
sigma(:,:,1) = [1 0;0 1];
sigma(:,:,2) = [1 0;0 1];
p = [0.5,0.5];

label = rand(1,N) >= p(1);
Nc = [length(find(label==0)),length(find(label==1))];
x = zeros(n,N);

for l = 0:1
    x(:,label==l) = mvnrnd(mu(:,l+1),sigma(:,:,l+1),Nc(l+1))';
end

figure(1), clf,
plot(x(1,label==0),x(2,label==0),'o'), hold on,
plot(x(1,label==1),x(2,label==1),'+'), axis equal,
legend('Class 0','Class 1'), 
title('Data and their true labels 1'),
xlabel('x_1'), ylabel('x_2'), 

%Sample 2
n = 2;
N = 400;
mu(:,1) = [0;0];
mu(:,2) = [3;3];
sigma(:,:,1) = [3 1;1 0.8];
sigma(:,:,2) = [3 1;1 0.8];
p = [0.5,0.5];

label = rand(1,N) >= p(1);
Nc = [length(find(label==0)),length(find(label==1))];
x = zeros(n,N);

for l = 0:1
    x(:,label==l) = mvnrnd(mu(:,l+1),sigma(:,:,l+1),Nc(l+1))';
end

figure(2), clf,
plot(x(1,label==0),x(2,label==0),'o'), hold on,
plot(x(1,label==1),x(2,label==1),'+'), axis equal,
legend('Class 0','Class 1'), 
title('Data and their true labels 2'),
xlabel('x_1'), ylabel('x_2'), 

%Sample 3
n = 2;
N = 400;
mu(:,1) = [0;0];
mu(:,2) = [2;2];
sigma(:,:,1) = [2 0.5;0.5 1];
sigma(:,:,2) = [2 0.5;0.5 1];
p = [0.5,0.5];

label = rand(1,N) >= p(1);
Nc = [length(find(label==0)),length(find(label==1))];
x = zeros(n,N);

for l = 0:1
    x(:,label==l) = mvnrnd(mu(:,l+1),sigma(:,:,l+1),Nc(l+1))';
end

figure(3), clf,
plot(x(1,label==0),x(2,label==0),'o'), hold on,
plot(x(1,label==1),x(2,label==1),'+'), axis equal,
legend('Class 0','Class 1'), 
title('Data and their true labels 3'),
xlabel('x_1'), ylabel('x_2'), 

%Sample 4
n = 2;
N = 400;
mu(:,1) = [0;0];
mu(:,2) = [3;3];
sigma(:,:,1) = [1 0;0 1];
sigma(:,:,2) = [1 0;0 1];
p = [0.05,0.95];

label = rand(1,N) >= p(1);
Nc = [length(find(label==0)),length(find(label==1))];
x = zeros(n,N);

for l = 0:1
    x(:,label==l) = mvnrnd(mu(:,l+1),sigma(:,:,l+1),Nc(l+1))';
end

figure(4), clf,
plot(x(1,label==0),x(2,label==0),'o'), hold on,
plot(x(1,label==1),x(2,label==1),'+'), axis equal,
legend('Class 0','Class 1'), 
title('Data and their true labels 4'),
xlabel('x_1'), ylabel('x_2'), 

%Sample 5
n = 2;
N = 400;
mu(:,1) = [0;0];
mu(:,2) = [3;3];
sigma(:,:,1) = [3 1;1 0.8];
sigma(:,:,2) = [3 1;1 0.8];
p = [0.05,0.95];

label = rand(1,N) >= p(1);
Nc = [length(find(label==0)),length(find(label==1))];
x = zeros(n,N);

for l = 0:1
    x(:,label==l) = mvnrnd(mu(:,l+1),sigma(:,:,l+1),Nc(l+1))';
end

figure(5), clf,
plot(x(1,label==0),x(2,label==0),'o'), hold on,
plot(x(1,label==1),x(2,label==1),'+'), axis equal,
legend('Class 0','Class 1'), 
title('Data and their true labels 5'),
xlabel('x_1'), ylabel('x_2'), 

%Sample 3
n = 2;
N = 400;
mu(:,1) = [0;0];
mu(:,2) = [2;2];
sigma(:,:,1) = [2 0.5;0.5 1];
sigma(:,:,2) = [2 0.5;0.5 1];
p = [0.05,0.95];

label = rand(1,N) >= p(1);
Nc = [length(find(label==0)),length(find(label==1))];
x = zeros(n,N);

for l = 0:1
    x(:,label==l) = mvnrnd(mu(:,l+1),sigma(:,:,l+1),Nc(l+1))';
end

figure(6), clf,
plot(x(1,label==0),x(2,label==0),'o'), hold on,
plot(x(1,label==1),x(2,label==1),'+'), axis equal,
legend('Class 0','Class 1'), 
title('Data and their true labels 6'),
xlabel('x_1'), ylabel('x_2'), 
