function [loss, gradients] = model_Gradients(net, img, label)
    if canUseGPU
        img = gpuArray(img);
    end
    
    % Forward pass
    out1 = activations(net, img,"bfm3a");
    
    % Calculate loss (cross-entropy)
    loss = crossentropy(out1, label);
    
    % Calculate gradients
    gradients = dlgradient(loss, net.LearnableParameters);
end