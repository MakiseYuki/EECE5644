%% Activation Function
function fx = Activation_func(fx, unipolarBipolarSelector)
    if (unipolarBipolarSelector == 0)
        fx = 1/(1+exp(-fx)); %
    else
        fx = ln(1+exp(fx)); %
    end
end