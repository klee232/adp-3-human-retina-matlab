classdef BinaryActivationLayer < nnet.layer.Layer
    % BinaryActivationLayer applies a binary threshold to the input.

    methods
        function layer = BinaryActivationLayer(name)
            % Set layer name.
            layer.Name = name;
            % Set layer description.
            layer.Description = "Binary activation with threshold 0.5";
        end
        function Z = predict(layer, X)
            % Forward input data through the layer at prediction time and
            % output the binary activation.
            Z = X;
            Z(Z >= 0.5)=1;
            Z(Z < 0.5)=0;
        end

    end
end
