clear all;
close all;

originV = [1.2,1.3,1.4,1.5,1.6];
transedV = zeros(5);
for i = 1:5
    transedV(i) = tranV(originV(i));
end

function transV = tranV(x)
    mu = 1;
    sigma = 3;
    n = 5;
    transV = (2*pi)^(-n/2)*sigma^(-1/2)*exp(((-1/2*(transmission(x-mu)))/sigma)*(x-mu));
end
function trans = transmission(x)
    trans = 1/1*((2*pi))*exp(-x^2/2);
end
% trsfomation for (0,I)

