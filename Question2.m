x = -5:5;
y = log(exp(-abs(x-0)/1))-log(exp(-abs(x-1)/2));

plot(x,y,'g',0,0.5,'g*',1,-1,'g*')
text(0,0.5,'(0,0.5)')
text(1,-1,'(1,-1)')
title('log-likelihood')
xlabel('x')
ylabel('l(x)')
xlim([-5 5])
ylim([-5 5])
