clear all; close all;

fun = @(x)100*(2*x(2) - x(1)^2)^2 + (1 - x(1))^2;
x0 = [0,0];
x = fminsearch(fun,x0)