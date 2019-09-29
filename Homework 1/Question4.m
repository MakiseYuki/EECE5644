clear all;
close all;

x = linspace(-5,5);
y1 = 1/(2*pi)^(1/2)*exp(-(x.^2)/2);
y2 = 1/(2*(2*pi^(1/2)))*exp(-(x-1).^2/(2*2));
plot(x,y1,x,y2)

%min_error = integral(y2,-Inf,0.8586)+integral(y1,0.8586,Inf);