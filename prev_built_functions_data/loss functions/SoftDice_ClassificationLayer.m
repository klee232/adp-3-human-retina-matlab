% Created by Kuan-Min Lee
% Created date: Mar. 6th, 2024
% All rights reserved to Leelab.ai

% Brief User Introduction:
% I created tis customized convolutional layer class because I can't stand
% with MATLAB's stupid training network function

% Setup Parameter:
% Y: predicted outcome
% T: groundtruth outcome

classdef SoftDice_ClassificationLayer < nnet.layer.ClassificationLayer
    % This layer implements the Soft Dice loss function for training
    % semantic segmentation networks.
    
    properties(Constant)
        % Small constant to prevent division by zero.
        Epsilon = 1e-8;
    end

    methods

        function layer = SoftDice_ClassificationLayer(name)
            % layer =  tverskyPixelClassificationLayer(name, alpha, beta) creates a Tversky
            % pixel classification layer with the specified name and properties alpha and beta.
            
            % Set layer name.          
            layer.Name = name;            
            % Set layer description.
            layer.Description = 'Soft Dice loss';
        end
        
        function loss = forwardLoss(layer, Y, T)
            Y=gpuArray(Y);
            T=gpuArray(T);
            % loss = forwardLoss(layer, Y, T) returns the Soft Dice loss between
            % the predictions Y and the training targets T.  
            % multiply Y with T elementwisely and construct the numerator
            T_opt=double(T);
            T_opt(T_opt>0)=1;
            first_term=Y.*T_opt;
            sum_first=sum(first_term,'all');
            num=2*sum_first+layer.Epsilon;
            % calculate the sum of Y and T and construct the denominator
            sum_Y=sum(Y,'all');
            sum_T=sum(T_opt,'all');
            den=sum_Y+sum_T+layer.Epsilon;
            % calculate Soft Dice Loss
            loss=1-num./den;
        end     

        function dLdY = backwardLoss(layer, Y, T)
            % dLdY = backwardLoss(layer,Y,T) returns the backpropagation
            % outcome of softdice loss
            Y=gpuArray(Y);
            T=gpuArray(double(T));
            T(T>0)=1;
            % conduct the calculation
            YT_mult=Y.*T;
            sum_Y=sum(Y,"all");
            sum_T=sum(T,'all');
            sum_YT=sum(YT_mult,'all');
            % numerator part
            first_term=(-1)*(2*sum_YT+layer.Epsilon)/(sum_Y+sum_T+layer.Epsilon)^2;
            second_term=2*T.*(sum_Y+sum_T+layer.Epsilon)/(sum_Y+sum_T+layer.Epsilon)^2;
            dLdY=(-1)*(first_term+second_term);
        end  

    end
end