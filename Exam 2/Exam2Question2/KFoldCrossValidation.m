function [trainedData,trainedLabel,dataValidation,labelValidation] = KFoldCrossValidation(initial_terminal,K,k,data,label)
    validationData = [initial_terminal(k,1):initial_terminal(k,2)]; % set the validation data as k's block
    N = size(data,2);
    if k == 1
        trainData = [initial_terminal(k,2)+1:N]; % first as validation 
    elseif k == K
        trainData = [1:initial_terminal(k,1)-1]; % last as validation
    else
        trainData = cat(2,[1:initial_terminal(k,1)-1],[initial_terminal(k,2)+1:N]);
    end
    trainedData = data(:,trainData); 
    trainedLabel = label(:,trainData); 
    dataValidation = data(:,validationData); 
    labelValidation = label(:,validationData); 
end
