% Created by Kuan-Min Lee
% Created date: Mar. 6th, 2024
% All rights reserved to Leelab.ai

% Brief User Introduction:
% I created tis customized convolutional layer class because I can't stand
% with MATLAB's stupid training network function

% Setup Parameter:
% Y: predicted outcome
% T: groundtruth outcome

classdef tverskyPixelClassificationLayer_ppt1 < nnet.layer.ClassificationLayer
    % This layer implements the Tversky loss function for training
    % semantic segmentation networks.
    
    % References
    % Salehi, Seyed Sadegh Mohseni, Deniz Erdogmus, and Ali Gholipour.
    % "Tversky loss function for image segmentation using 3D fully
    % convolutional deep networks." International Workshop on Machine
    % Learning in Medical Imaging. Springer, Cham, 2017.
    % ----------
    
    
    properties(Constant)
        % Small constant to prevent division by zero.
        Epsilon = 1e-8;
    end
    
    properties
        % Default weighting coefficients for False Positives and False
        % Negatives
        Alpha = 0.5;
        Beta = 0.5;
    end

    
    methods
 
        function layer = tverskyPixelClassificationLayer_ppt1(name, alpha, beta)
            % layer =  tverskyPixelClassificationLayer(name, alpha, beta) creates a Tversky
            % pixel classification layer with the specified name and properties alpha and beta.
            
            % Set layer name.          
            layer.Name = name;
            
            layer.Alpha = alpha;
            layer.Beta = beta;
            
            % Set layer description.
            layer.Description = 'Tversky loss';
        end
        
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the Tversky loss between
            % the predictions Y and the training targets T.  
            Y=gpuArray(Y);
            T=gpuArray(T);
            % set up ground truth
            T0=1-T; % truth for background
            T1=T; % truth for vessel
            Y0=Y(:,:,1); % predicted background
            Y1=Y(:,:,2); % predicted vessel


            % calculate true positive
            multi_Y1T1=Y1.*T1;
            True_Positive=sum(multi_Y1T1,'all');
            % calculate fasle positive
            multi_Y1T0=Y1.*T0;
            False_Positive=sum(multi_Y1T0,'all');
            % calculate false negative
            multi_Y0T1=Y0.*T1;
            False_Negative=sum(multi_Y0T1,'all');
            
            % calculate Tversky Loss
            loss=1-((True_Positive)/(True_Positive+layer.Alpha*False_Positive+layer.Beta*False_Negative));
            loss=gpuArray(loss);

        end     

        function dLdY = backwardLoss(layer, Y, T)
            % dLdY = backwardLoss(layer,Y,T) returns the backpropagation outcome of Tversky loss with respect to the predicted outcome from the model  
            % set up ground truth
            Y=gpuArray(Y);
            T=gpuArray(T);
            T0=1-T; % truth for background
            T1=T; % truth for vessel
            Y0=Y(:,:,1); % predicted background
            Y1=Y(:,:,2); % predicted vessel

            % setup dLdY matrix
            [row_Y,col_Y,chn_Y]=size(Y);
            dLdY=zeros(row_Y,col_Y,chn_Y);

            % calculate dLdY0 (with resppect to background)
            multi_Y1T1=Y1.*T1;
            sum_multi_Y1T1=sum(multi_Y1T1,'all');
            multi_Y1T0=Y1.*T0;
            sum_multi_Y1T0=sum(multi_Y1T0,'all');
            multi_Y0T1=Y0.*T1;
            sum_multi_Y0T1=sum(multi_Y0T1,'all');
            g=(sum_multi_Y1T1+layer.Alpha*sum_multi_Y1T0+layer.Beta*sum_multi_Y0T1);
            num=(-1)*layer.Beta*T*sum_multi_Y1T+layer.Epsilon;
            den=power(g,2)+layer.Epsilon;
            dLdY0=num/den;

            % calculate dLdY1 (with resppect to background)
            num=(2)*(T1*g-sum_multi_Y1T1*(Y1+layer.Alpha*Y0));
            den=power(g,2);
            dLdY1=num/den;

            % store in output variable
            dLdY(:,:,1)=dLdY0;
            dLdY(:,:,2)=dLdY1;

            dLdY=gpuArray(dLdY);
        end  

    end
end