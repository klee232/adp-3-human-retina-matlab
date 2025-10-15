% Created by Kuan-Min Lee
% Created date: Dec. 5th, 2024
% All rights reserved to Leelab.ai

% Brief User Introduction:
% This is the customized neural network layer which serves as the function
% of patch mean removal 

% Input Parameter:
% patch_size: size of patch for patch formation
% input: original input feature

% Output Parameter:
% output: processed array (4D array with size of [depth, patch_size,
% patch_size, number_patch])
classdef fine_block_input_split_layer < nnet.layer.Layer & nnet.layer.Formattable

    properties
        Desired_input
    end
       
    methods

        %% Constructor for the layer
        function layer = fine_block_input_split_layer(name,desired_input)
           layer.Name = name;
           layer.Description="fine block input concatenation layer";
           layer.Desired_input=desired_input;
        end

        %% Forward pass: channel multiplication
        function [output]=predict(layer,input)
            %% set up propagation coefficients
            desired_input_chn=layer.Desired_input;
            num_chn=numel(desired_input_chn);
            output=zeros(size(input,1),size(input,2),num_chn);
            for i_chn=1:num_chn
                location=desired_input_chn(i_chn);
                current_input=input(:,:,location);
                output(:,:,i_chn)=current_input;
            end
            output=dlarray(output,"SSCB");

        end
       
    end
end
