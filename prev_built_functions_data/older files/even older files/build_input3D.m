% Created by Kuan-Min Lee
% Created date: Dec. 15th, 2023
% All rights reserved to Leelab.ai

% Brief User Introduction:
% This the customized 3D input layer for building neural network

% Input Parameter:
% input_feature (numerical array)
% Length: input variable for dimension 1
% Width: input variable for dimension 2
% Channel: input variable for dimension 3
% Output Parameter:
% output_feature (numerical array)


function layer=build_input3D(input_feature,Length,Width,Channel)
    % setup layer
    layer=image3dInputLayer([Length Width Channel],'Name','input');
    % output feature map
%     output_feature=activations(layer,input_feature,1);

end
