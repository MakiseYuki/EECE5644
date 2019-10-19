clear all;
xt = 0.9; yt = 0.9;
object_true = [xt yt]';

for sample = 1:4
    k = sample;
    switch k
        case 1
            land_mark = [1 0]';
        case 2
            land_mark = [1 0;-1 0]';
        case 3
            land_mark = [0 1;-0.5*sqrt(3) -0.5;0.5*sqrt(3) -0.5]';
        case 4
            land_mark = [1 0;0 1;-1 0;0 -1]';
    end

    mu_x = 0; mu_y = 0; mu_noise = 0;
    Sigma_x = 0.25; Sigma_y = 0.25; Sigma_noise = 0.3;

    % % Get the priori
    % x = 0; y = 0;
    % e_power = -0.5*[x y]*inv([Sigma_x^2 0;0 Sigma_y^2])*[x;y];
    % p = (2*pi*Sigma_x*Sigma_y)^(-1)*exp(e_power);

    r = zeros(k,1);
    % Prevent from generate the negtive r
    while length(find(r<=0)) > 0
        n = mvnrnd(0,Sigma_noise,k);
        true_distance = distance(object_true,land_mark)';
        r = distance(object_true,land_mark)' + n;
    end

    x = linspace(-2,2);
    y = linspace(-2,2);
    [X,Y] = meshgrid(x,y);

    g = zeros(length(x),length(y));
    kg = zeros(length(x),length(y),k);
    
    g = double(g);

    for i = 1:length(x)
        for j = 1:length(y)
                point = [X(i,j) Y(i,j)]';
                es_r = distance(point,land_mark)';
                gs = estMAP(es_r,true_distance,Sigma_noise,Sigma_x,Sigma_y,point);
                for z = 1:k
                    kg(i,j,z) = gs(z);
                end
                g(i,j) = min(gs);
        end
    end

    


    % x = linspace(-2,2);
    % y = linspace(-2,2);
    % [X,Y] = meshgrid(x,y);
    % 
    % Z = estMAP(r,true_distance,Sigma_noise,Sigma_x,Sigma_y,X,Y,land_mark);
    
    figure(1), subplot(2,2,sample)
        
    plot(xt,yt,'+g'); axis equal, hold on
    plot(land_mark(1,:),land_mark(2,:),'or'); axis equal, hold on, axis([-2 2 -2 2]);
    contour(X,Y,g,20);
    title ("Contour " + sample + " land marks")
    %contour(X,Y,Z);
    
end

    figure(2),
    plot(xt,yt,'+g'); axis equal, hold on
    plot(land_mark(1,:),land_mark(2,:),'or'); axis equal, hold on, axis([-2 2 -2 2]);
    contour(X,Y,kg(:,:,1)); hold on
    contour(X,Y,kg(:,:,2)); hold on
    contour(X,Y,kg(:,:,3)); hold on
    contour(X,Y,kg(:,:,4)); hold on
    title('Contour for each K combination')
    
    
function gm = estMAP(r,mu,Sigma_noise,Sigma_x,Sigma_y,point)
%     L = (r-mu)^2/Sigma_noise^2;
%     P = (x^2/Sigma_x^2) + (y^2/Sigma_y^2);
%     g = P + L;
    
    p = (point(1)^2/Sigma_x^2) + (point(2)^2/Sigma_y^2);
    of =(r-mu).^2/Sigma_noise^2;
    gm = p + of;
    %gm = min(g);
end


function dis = distance(object_true,k)
    dis = sqrt((object_true(1)-k(1,:)).^2+(object_true(2)-k(2,:)).^2);
end