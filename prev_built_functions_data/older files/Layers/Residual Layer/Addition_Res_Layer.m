classdef Addition_Res_Layer< nnet.layer.Layer
    % BinaryActivationLayer applies a binary threshold to the input.

    methods
        function layer = BinaryActivationLayer(name)
            % Set layer name.
            layer.Name = name;
            % Set layer description.
            layer.Description = "Addition Layer";
        end
        function Z = predict(layer, X,X2)
            % Forward input data through the layer 
            size(X)

            Z = X;
            Z(Z >= 0.5)=1;
            Z(Z < 0.5)=0;
        end

    end
end
