clear all;
close all;

%set two pdf function as parameter=1/2
x = linspace(-5,5);
y1 = @(x)1/4*exp(-abs(x-0)/1);
y2 = @(x)1/8*exp(-abs(x-1)/2);

y3 = 1/(2*pi)^(1/2)*1/4*exp(-abs(x));
y4 = 1/(2*pi)^(1/2)*1/8*exp(-abs(x-1)/2);

plot(x,y3,x,y4)

%We get the plot and integral feom the plor to derive the min_error

min_error = integral(y1,-Inf,-2.374)+integral(y2,-2.374,0.7576)+integral(y1,0.7576,Inf); 