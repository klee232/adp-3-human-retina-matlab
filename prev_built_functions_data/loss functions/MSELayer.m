% Created by Kuan-Min Lee
% Created date: Mar. 15th, 2024
% All rights reserved to Leelab.ai

% Brief User Introduction:
% I created tis customized convolutional layer class because I can't stand
% with MATLAB's stupid training network function

% Setup Parameter:


classdef MSELayer < nnet.layer.ClassificationLayer

    methods
 
        function layer = MSELayer(name)
            % layer =  tverskyPixelClassificationLayer(name, alpha, beta) creates a Tversky
            % pixel classification layer with the specified name and properties alpha and beta.
            
            % Set layer name.          
            layer.Name = name;
            
            % Set layer description.
            layer.Description = 'Mean Squared loss';
        end
        
        function loss = forwardLoss(Y, T)
            Y=gpuArray(Y);
            T=gpuArray(T);
            % loss = forwardLoss(layer, Y, T) returns the Mean Absolute loss between
            % the predictions Y and the training targets T.   

            % Preprocessing for Y
            norm_Y=(Y-min(Y,[],'all'))/(max(Y,[],'all')-min(Y,[],'all')); % normalize Y 
            % Binarized Y
            norm_Y(norm_Y<0.5)=0; 
            norm_Y(norm_Y>=0.5)=1;
            
            % Preproccessing for T
            % Binarized T
            T(T>0)=1; 

            % Conduct Mean Absolute Error Calculation
            out_mat=norm_Y-T;
            out_mat=power(out_mat,2);
            loss=sum(out_mat,"all");
            num_pixel=numel(out_mat);
            loss=loss/num_pixel;
        end     


        function dLdY = backwardLoss(Y, T)
            Y=gpuArray(Y);
            T=gpuArray(T);
            % retrieve the loss matrix
            % Preprocessing for Y
            norm_Y=(Y-min(Y,[],'all'))/(max(Y,[],'all')-min(Y,[],'all')); % normalize Y 
            % Binarized Y
            norm_Y(norm_Y<0.5)=0; 
            norm_Y(norm_Y>=0.5)=1;
            % Preproccessing for T
            % Binarized T
            T(T>0)=1; 
            % Conduct Mean Absolute Error Calculation
            out_mat=norm_Y-T;
            num_pixel=numel(out_sign);
            dLdY=(2/num_pixel)*out_mat;
            dLdY=gpuArray(dLdY);
        end  
    end
end