% Created by Kuan-Min Lee
% Created date: Mar. 9th, 2025
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
classdef fine_block_softmax_layer < nnet.layer.Layer & nnet.layer.Formattable    

    methods

        %% Constructor for the layer
        function layer = fine_block_softmax_layer(name)
           %% setup unleanrable parameters
           layer.Name = name;
           layer.Description="softmax layer (along feature)";
        end


        %% Forward pass: conduct softmax along feature dimension
        function [output]=predict(layer,input)
            %% conduct softmax along feature dimension
            % subtract max for numerical stability
            % input_max=max(input,[],3);
            % input_max=repmat(input_max,[1,1,size(input,3),1]);

            exp_input=exp(input);
            sum_exp=sum(exp_input,3);
            sum_exp=repmat(sum_exp, [1,1,size(input,3),1]);
            output=exp_input./sum_exp;
            output=dlarray(output,"SSCB");
        end
       
    end
end
