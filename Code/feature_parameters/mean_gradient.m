% max_gradient
function [ output_args ] = mean_gradient( value )
shape = size(value);
shape_len = shape(1);
shape_sample = 1;
for i = 2:length(shape)
    shape_sample = shape_sample * shape(i);
end
output_args = zeros([1, shape_sample]);
value_reshape = reshape(value, [shape_len, shape_sample]);
for i = 1:shape_sample
    output_args(1,i) = mean(gradient(value_reshape(:,i)));
end
output_args = reshape(output_args, [1, shape(2:end)]);
end
