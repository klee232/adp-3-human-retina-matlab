% Created by Kuan-Min Lee
% Created date: Mar. 6th, 2024
% All rights reserved to Leelab.ai

% Brief User Introduction:
% I created tis customized convolutional layer class because I can't stand
% with MATLAB's stupid training network function

% Input Arguments:
% lr (optional): learning rate (floating number)
% theta: parameters needs to be updated (double variable array)
% dL: gradient of current parameters with respect to the current input (double variable array)

% Outputt Arguments:
% updated_theta: updated parameters (double variable array)


function [updated_theta]=SGD_Optimizer(lr,theta,dL)
    theta=gpuArray(theta);
    dL=gpuArray(dL);
    % setup learning rate values
    % if not exist, set it to default value
    if ~exist('lr','var')
        lr=0.001;
    end

    % Update parameters
    updated_theta=theta-lr*dL;

end