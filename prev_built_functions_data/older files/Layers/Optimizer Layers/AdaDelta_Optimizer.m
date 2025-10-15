% Created by Kuan-Min Lee
% Created date: Mar. 6th, 2024 
% All rights reserved to Leelab.ai

% Brief User Introduction:
% I created tis customized convolutional layer class because I can't stand
% with MATLAB's stupid training network function

% Input Arguments:
% first_moving_avg: first moving average from previous iteration  (double variable)
% second_moving_avg: second moving average from previous iteration  (double variable)
% decay rate (optional): decay rate (floating number)
% theta: parameters needs to be updated (double variable array)
% dL: gradient of current parameters with respect to the current input (double variable array)

% Output Arguments:
% updated_theta: updated parameters (double variable array)
% out_first_moving_avg: updated first moving average  (double variable)
% out_second_moving_avg: updated second moving average  (double variable)


function [updated_theta,out_first_mov_avg,out_second_mov_avg]=AdaDelta_Optimizer(first_moving_average,second_moving_average,iteration_num,decay_rate,theta,dL)
    theta=gpuArray(theta);
    dL=gpuArray(dL);
    % setup moving average values
    % if not exist, set it to default value
    if ~exist('first_moving_average','var')
        first_moving_average=0;
    end
    if ~exist('second_moving_average','var')
        second_moving_average=0;
    end
    % setup decay rate values
    % if not exist, set it to default value
    if ~exist('decay_rate','var')
        decay_rate=0.9;
    end
    % set up epsilon
    eps=1e-8;

    % calculate the first moving average
    out_first_mov_avg=decay_rate*first_moving_average+(1-decay_rate)*dL*dL;
    out_first_mov_avg=out_first_mov_avg/iteration_num;

    % calcuulate the second moving average and update of parameter
    % update of parameter
    delta_theta=(-1)*(sqrt(second_moving_average+eps))/(sqrt(first_moving_average+eps))*dL;
    out_second_mov_avg=decay_rate*second_moving_average+(1-decay_rate)*delta_theta*delta_theta;
    out_second_mov_avg=out_second_mov_avg/iteration_num;

    % update the parameter
    updated_theta=theta+delta_theta;

end