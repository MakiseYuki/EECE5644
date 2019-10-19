clear all;

land_mark = [0 0]';
x = linspace(-2,2);
y = linspace(-2,2);
[X,Y] = meshgrid(x,y);
temp = zeros(length(x),length(y));
for i = 1:length(x)
        for j = 1:length(y)
            point = [X(i,j) Y(i,j)];
            temp(i,j) = distance(point,land_mark);
        end
end


function dis = distance(object_true,k)
    dis = sqrt((object_true(1)-k(1,:)).^2+(object_true(2)-k(2,:)).^2);
end