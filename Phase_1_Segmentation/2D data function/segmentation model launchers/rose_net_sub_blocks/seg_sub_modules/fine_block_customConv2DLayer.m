classdef fine_block_customConv2DLayer < nnet.layer.Layer & nnet.layer.Formattable
    properties (Learnable)
        Weights
        Bias
    end
    
    properties
        FilterSize
        NumFilters
        NumInputFilters
        Padding
    end
    
    methods
        function layer = fine_block_customConv2DLayer(filterSize, numInputFilters,numFilters, name, varargin)
            % Constructor
            layer.Name = name;
            layer.Description = "Custom 2D convolution with weights and bias";
            layer.Type = "CustomConv";

            layer.FilterSize = filterSize;
            layer.NumFilters = numFilters;
            layer.NumInputFilters = numInputFilters;

            % Default: 'same' padding
            if isempty(varargin)
                layer.Padding = 'same';
            else
                layer.Padding = varargin{1};
            end

            % Initialize weights: Xavier/Glorot
            size = [filterSize filterSize numInputFilters numFilters];
            layer.Weights = randn(size, 'single') * 0.01;
            bias=zeros([1 1 numFilters], 'single');
            bias(1,1,ceil(numFilters/2))=1;
            layer.Bias = bias;
        end

        function [output] = predict(layer, input)
            input=dlarray(input,"SSCB");
            output=dlconv(input, layer.Weights, layer.Bias, 'Padding', layer.Padding);
            output=dlarray(output,"SSCB");
        end
    end
end
