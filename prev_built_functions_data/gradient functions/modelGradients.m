function [loss, gradients] = modelGradients(net, img, label)
    img=dlarray(single(img),'SSCB');
    
    if canUseGPU
        img = gpuArray(img);
    end
    
    % Forward pass
    out1 = forward(net, img);
    
    % Calculate loss (cross-entropy)
    loss = crossentropy(out1, label);
    
    % Calculate gradients
    gradients = dlgradient(loss, net.LearnableParameters);
end