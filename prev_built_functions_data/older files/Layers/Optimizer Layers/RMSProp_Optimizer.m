% Created by Kuan-Min Lee
% Created date: Mar. 6th, 2024
% All rights reserved to Leelab.ai

% Brief User Introduction:
% I created tis customized convolutional layer class because I can't stand
% with MATLAB's stupid training network function

% Input Arguments:
% momentum: momentum from previous step (iteration) (double variable)
% beta (optional): beta1 for Adam optimizer 
% lr (optional): learning rate (floating number)
% theta: parameters needs to be updated (double variable array)
% dL: gradient of current parameters with respect to the current input (double variable array)

% Output Arguments:
% updated_theta: updated parameters (double variable array)
% out_momentum: updated momentum  (double variable)


function [updated_theta,out_momentum]=RMSProp_Optimizer(momentum,beta,lr,theta,dL)
    theta=gpuArray(theta);
    dL=gpuArray(dL);
    % setup momentum values
    % if not exist, set it to default value
    if ~exist('momentum','var')
        momentum=0;
    end

    % setup beta values
    % if not exist, set it to default value
    if ~exist('beta','var')
        beta=0.9;
    end

    % setup learning rate values
    % if not exist, set it to default value
    if ~exist('lr','var')
        lr=0.001;
    end

    % set up epsilon
    eps=1e-7;

    % calculate the momentum (included bias)
    out_momentum=beta*momentum+(1-beta)*dL*dL;


    % Update parameters
    updated_theta=theta-(lr/(sqrt(out_momentum+eps)))*dL;

end