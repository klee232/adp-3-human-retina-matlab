% Created by Kuan-Min Lee
% Created date: Jan. 29th, 2024
% All rights reserved to Leelab.ai

% Brief User Introduction:
% I created tis customized convolutional layer class because I can't stand
% with MATLAB's stupid training network function

% Setup Parameter:


classdef tverskyPixelClassificationLayer < nnet.layer.RegressionLayer
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
        Alpha;
        Beta;
    end

    
    methods
        function layer = tverskyPixelClassificationLayer(name, alpha, beta)
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
            Pcnot = 1-Y; % predicted negative
            Gcnot = 1-T; % truth negative
            TP = sum(sum(sum(Y.*T,1),2),3); % true positive detection
            FP = sum(sum(sum(Y.*Gcnot,1),2),3); % false positive detection
            FN = sum(sum(sum(Pcnot.*T,1),2),3); 
            
            numer = TP + layer.Epsilon;
            denom = TP + layer.Alpha*FP + layer.Beta*FN + layer.Epsilon;
            
            % Compute tversky index
            lossTIc = 1 - numer./denom;

            lossTI = sum(lossTIc,4);
            
            % Return average tversky index loss.
            N = size(Y,3);
            loss = sum(lossTI)/N;
        end     

    end
end