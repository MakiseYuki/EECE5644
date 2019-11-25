%% Example: classification of spirals with Levenberg-Marquardt algorithm
function MLP = ExampleClassificationSpirals()
close('all','force'); clc;
% Parameters
R = 4;
k = 8;
N = R;
S = k*N;
Nt = S*N;
x0 = 0;
y0 = 0;
% Preallocations
X = zeros(N,S);
Y = zeros(N,S);
Regions = zeros(1,S);
% Spiral coordinates
for s = 1:S
   r = floor((s-1)*R/S+1);
   for n = 1:N
        a0 = n/N*2*pi/k;
        a = a0+s*2*pi/S;
        b = a0;
        X(n,s) = b*cos(a)+x0;
        Y(n,s) = b*sin(a)+y0;
        Regions(n,s) = r;
   end
end
Coordinates = [X(:)';Y(:)'];
% Regions
Regions = Regions(:)';
% Probabilities
Probabilities = zeros(R,numel(Regions));
for n = 1:numel(Regions)
    Probabilities(Regions(n),n) = 1;
end
% % Figure
% Figure = figure('Color','w');
% axis('off');
% hold('on');
% Colors = {'m','g','c','y'};
% % Full screen
% jFrame = get(Figure,'JavaFrame');
% drawnow(); pause(0.1);
% jFrame.setMaximized(true);
% % Plot of the coordinates
% for r = 1:R
%     I = find(eq(Regions,r));
%     plot(Coordinates(1,I),Coordinates(2,I),'o',...
%         'MarkerEdgeColor',  'k',...
%         'MarkerFaceColor',  Colors{r},...
%         'MarkerSize',       6);
% end        
% xlim([min(min(X)) max(max(X))]);
% ylim([min(min(Y)) max(max(Y))]);
% axis('equal');
% drawnow();
% % Coordinates of map components
% M = (1+1/N/2)/k*2*pi;
% N = 300;
% V = linspace(-M,+M,N);
% [X,Y] = ndgrid(V+x0,V+y0);      
% Coordinates2 = [X(:)';Y(:)'];
% % Map color
% Colors = [1 0 0;...
%           0 1 0;...
%           0 0 1;...
%           1 1 0];
% Multi-layer perceptron
MLP = ...
    MultiLayerPerceptron('LengthsOfLayers', [2 16 8 R R],...
                         'HiddenActFcn',    'tanh',...
                         'OutputActFcn',    'softmax',...
                         'UpdateFcn',       'default');
% Training options
Options = ...
    struct('TrainingAlgorithm',         'LM',...
           'NumberOfEpochs',            200,...
           'MinimumMSE',                1e-2,...
           'SizeOfBatches',             1,...
           'SplitRatio',                1,...
           'InitialMarquardtParameter', 0.01,...
           'MaximumMarquardtParameter', 1e10,...
           'MarquardtFactor',           10,...
           'BayesianRegularization',    false,...
           'UpdateFcn',                 @Update);                    
% Training
MLP.train(Coordinates,Probabilities,Options);     
     
    % Figure update
    function Continue = Update(MLP)
        
        persistent Coloration Epoch VR
        % Training continuation boolean        
        Continue = true;
        
        switch MLP.TrainingStep
            
            case 'start'
                
                % Creation of the video
                Name = 'Spirals';
                VR = ...
                    VideoRecorder('Filename',     [Name datestr(now,'_dd-mm-yy_HH-MM-SS')],...
                                  'Fileformat',   'MPEG-4',...
                                  'Figure',       Figure);
                              
                % Update of the video
                for i = 1:20
                    VR.add();
                end
                
                return
                
            case 'cancellation'
                % Cancellation of the current training step
                return
           
            case 'Update'
                % Update of the current step except if the epoch is the same
                if MLP.CurrentEpoch == Epoch
                    return
                end
                
            case 'stop'
        
                % Ending of the video
                VR.stop();
                return
        
        end
        % Number of erroneous regions
        MLP.propagate(Coordinates);
        [~,R] = max(MLP.Outputs);
        E = sum(~eq(R,Regions));
        
        if eq(E,0)
            Continue = false; 
        end
        
        % Title update
        title(sprintf('Epoch: %02u, Errors: %u/%u',MLP.CurrentEpoch,E,Nt));
        
        % Most probable regions     
        MLP.propagate(Coordinates2);                
        Probabilities2 = MLP.Outputs;                   
        [~, Regions2] = max(Probabilities2,[],1);        
        
        switch MLP.CurrentEpoch
            case 1
                
                % Creation of the map
                Coloration = ...
                    pcolor(reshape(Coordinates2(1,:),N,N),...
                           reshape(Coordinates2(2,:),N,N),...
                           reshape(Regions2,N,N));
                set(Coloration,...
                    'EdgeColor', 'none',...
                    'FaceAlpha', 0.25);        
                colormap(Colors);
                uistack(Coloration,'bottom');
                
            otherwise
                
                % Update of the map
                set(Coloration,...
                    'Xdata', reshape(Coordinates2(1,:),N,N),...
                    'Ydata', reshape(Coordinates2(2,:),N,N),...
                    'Cdata', reshape(Regions2,N,N));                      
                colormap(Colors);
                
        end
        
        % Update of the video
        drawnow(); 
        for i = 1:20
            VR.add();
        end
        
        % Current epoch
        Epoch = MLP.CurrentEpoch;
    end
end