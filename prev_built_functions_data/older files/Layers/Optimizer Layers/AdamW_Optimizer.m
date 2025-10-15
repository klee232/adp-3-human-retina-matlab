% Created by Kuan-Min Lee
% Created date: Mar. 6th, 2024 
% All rights reserved to Leelab.ai

% Brief User Introduction:
% I created tis customized convolutional layer class because I can't stand
% with MATLAB's stupid training network function

% Input Arguments:
% momentum: momentum from previous step (iteration) (double variable)
% lr (optional): learning rate (floating number)
% theta: parameters needs to be updated (double variable array)
% dL: gradient of current parameters with respect to the current input (double variable array)

% Output Arguments:
% updated_theta: updated parameters (double variable array)
% out_momentum: updated momentum  (double variable)

function [updated_theta,out_momentum1,out_momentum2]=AdamW_Optimizer(momentum1,momentum2,beta1,beta2,lr,step,theta,dL)
    theta=gpuArray(theta);
    dL=gpuArray(dL);
    % setup momentum values
    % if not exist, set it to default value
    if ~exist('momentum1','var')
        momentum1=0;
    end
    if ~exist('momentum2','var')
        momentum2=0;
    end

    % setup beta values
    % if not exist, set it to default value
    if ~exist('beta1','var')
        beta1=0.9;
    end
    if ~exist('beta2','var')
        beta2=0.999;
    end

    % setup learning rate values
    % if not exist, set it to default value
    if ~exist('lr','var')
        lr=0.001;
    end

    % set up epsilon
    eps=1e-7;
    % setup lambda value
    lambda=1e-6;


    % calculate the momentums (included bias)
    out_momentum1=beta1*momentum1+(1-beta1)*dL;
    out_momentum2=beta2*momentum2+(1-beta2)*dL*dL;
    % calculate bias corrected momentums
    out_momentum1_hat=out_momentum1/(1-beta1.^step);
    out_momentum2_hat=out_momentum2/(1-beta2.^step);


    % Update parameters
    updated_theta=theta-lr*(out_momentum1_hat+lambda*theta)/(sqrt(out_momentum2_hat)+eps);



end