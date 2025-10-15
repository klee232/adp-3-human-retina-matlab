classdef channel_wise_fullyconnectedLayer < nnet.layer.Layer
    properties (Learnable)
        Weights  % Layer weights (learnable)
        Bias     % Layer bias (learnable)
    end

    properties
        Num_layer
        Num_channel
        Num_unit_channel
    end

    methods
        function layer = channel_wise_fullyconnectedLayer(num_layer, num_channel, num_unit_channel, name)
            % Constructor for the layer
            layer.Name = name;
            layer.Description = "Channel-wise fully connected layer";

            layer.Num_layer=num_layer;
            layer.Num_channel=num_channel;
            layer.Num_unit_channel=num_unit_channel;

            layer.Weights = randn(num_layer, num_unit_channel, num_channel); % Initialize weights
            layer.Bias = randn(num_layer, num_unit_channel); % Initialize biases
        end

        function Z = predict(layer, input)
            % Forward pass (channel-wise fully connected operation)
            % X: input of size (batch, inputSize, numChannels)
            [input_layer, ~, input_channel] = size(input);
            Z = zeros(input_layer, layer.NumUnitsPerChannel, input_channel);
            for i_lyr=1:input_layer
                for i_chn = 1:input_channel
                    current_input=input(i_lyr,:,i_chn);
                    current_input=squeeze(current_input);
                    current_weight=layer.Weights(i_lyr,:,i_chn);
                    current_weight=squeeze(current_weight);
                    current_bias=layer.Bias(i_lyr,:,)
                    Z(:,:,i_chn)=current_input*current_weight + layer.Bias(i_chn);
                end
            end
        end

        function [dLdX, dLdW, dLdB] = backward(layer, X, ~, dLdZ, ~)
            % Backpropagation (gradient computation)
            numChannels = size(X, 3);
            dLdX = zeros(size(X), 'like', X);
            dLdW = zeros(size(layer.Weights), 'like', layer.Weights);
            dLdB = zeros(size(layer.Bias), 'like', layer.Bias);

            for c = 1:numChannels
                dLdX(:,:,c) = dLdZ(:,:,c) * layer.Weights(:,c)';
                dLdW(:,c) = sum(dLdZ(:,:,c) .* X(:,:,c), 1);
                dLdB(c) = sum(dLdZ(:,:,c), 'all');
            end
        end
    end
end
