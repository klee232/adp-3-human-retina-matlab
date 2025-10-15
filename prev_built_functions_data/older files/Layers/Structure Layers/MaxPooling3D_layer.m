% Created by Kuan-Min Lee
% Created date: Jan. 29th, 2024
% All rights reserved to Leelab.ai

% Brief User Introduction:
% I created tis customized convolutional layer class because I can't stand
% with MATLAB's stupid training network function

% Setup Parameter:
% kernel_size (optional): intended kernel size of max pooling
% sttride (optional): intended stride for kernel movement
% Input Parameter:
% input_feats: input feature maps (multi-dimensional array)
% Output Parameter
% output_feats: output feature maps (multi-dimensional array)


classdef MaxPooling3D_layer < nnet.layer.Layer 
    properties
    % define the unlearnable parameters for convolutional 3D layer
        filt_size
        stride
    end
    % define all the functions necessary for the layer
    methods
        % constructor function of the layer
        function layer = MaxPooling3D_layer(name,kernel_size,Stride)
            % layer = MaxPooling3D_layer creates a maxpooling layer with
            % kernel_size
            % set all input properties
            % String parts
            layer.Name=name; % name of current layer
            layer.Description="Customized MaxPooling 3D Layer"; % description of current layer
            % Parameter parts
            % if kernel size value is not set, set it to default value 1
            if ~exist("kernel_size",'var')
                layer.filt_size=2; 
            else
                layer.filt_size=kernel_size; % size of the filter
            end
            % if stride value is not set, set it to default value 1
            if ~exist("stride",'var')
                layer.stride=kernel_size; 
            else
                layer.stride=Stride; % size of the filter
            end
        end
        % forward function of the layer
        function feat_map = predict(layer,input_feats)
            % input: 
            % layer: 
            % in this case, it's the max pooling layer created
            % from the constructor function
            % input_feats: 
            % the feature maps being fed to the model
            % output:
            % feat_map: output feature maps
            % conduct max pooling calculation
            % create output feature map storage
            input_feats=gpuArray(input_feats);
            [row_input,col_input,chn_input]=size(input_feats);
            row_feat=((row_input-layer.filt_size)/layer.stride)+1;
            col_feat=((col_input-layer.filt_size)/layer.stride)+1;
            chn_feat=((chn_input-layer.filt_size)/layer.stride)+1;
            feat_map=zeros(row_feat,col_feat,chn_feat);
            % conduct max pooling 
            for i_chn=1:chn_feat
                for i_row=1:row_feat
                    for i_col=1:col_feat
                        feat_map(i_row,i_col,i_chn)=max(input_feats((i_row-1)*layer.stride+1:(i_row-1)*layer.stride+(layer.stride-1),...
                                                                    (i_col-1)*layer.stride+1:(i_col-1)*layer.stride+(layer.stride-1),...
                                                                    (i_chn-1)*layer.stride+1:(i_chn-1)*layer.stride+(layer.stride-1)),[],'all');
                    end
                end
            end
        end
        % backward function of the layer
        function [dLdx] = backward(layer,input_feats,output_feats,loss_feat)
            % input: 
            % layer: 
            % in this case, it's the convolutional 2D layer created
            % from the constructor function
            % input_feats: 
            % the feature maps being fed to the model
            % loss_feat:
            % the gradient of loss with respect to the input of the
            % previous layer
            % output:
            % dLdx: gradient with respect to the current input
            % create output variable storage
            input_feats=gpuArray(input_feats);
            output_feats=gpuArray(output_feats);
            loss_feat=gpuArray(loss_feat);
            [row_in,col_in,chn_in]=size(input_feats);
            dLdx=zeros(row_in,col_in,chn_in);
            % looping through each output feature element
            [row_out,col_out,chn_out]=size(output_feats);
            for i_chn_out=1:chn_out
                for i_row_out=1:row_out
                    for i_col_out=1:col_out
                        % grab out current output value
                        current_out=output_feats(i_row_out,i_col_out,i_chn_out);
                        % grab out current index range for input
                        start_row=(i_row_out-1)*layer.stride+1;
                        end_row=(i_row_out-1)*layer.stride+(layer.stride-1);
                        start_col=(i_col_out-1)*layer.stride+1;
                        end_col=(i_col_out-1)*layer.stride+(layer.stride-1);
                        start_chn=(i_chn_out-1)*layer.stride+1;
                        end_chn=(i_chn_out-1)*layer.stride+(layer.stride-1);
                        % check every value at current index
                        for i_row_input=start_row:end_row
                            for i_col_input=start_col:end_col
                                for i_chn_input=start_chn:end_chn
                                    current_val=input_feats(i_row_input,i_col_input,i_chn_input);
                                    % if current value equals to current output
                                    % value set it to 1
                                    if current_val==current_out
                                        dLdx(i_row_input,i_col_input,i_chn_input)=1;
                                    else
                                        dLdx(i_row_input,i_col_input,i_chn_input)=0;
                                    end
                                end
                            end
                        end
                    end
                end
            end
            dLdx=loss_feat*dLdx;
            dLdx=gpuArray(dLdx);
        end
    end
end