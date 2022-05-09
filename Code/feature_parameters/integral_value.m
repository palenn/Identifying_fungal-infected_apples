%integral_value
function [ output_args ] = integral_value(value, fps)
t = 1/fps;
len = size(value, 1);
x = t*(1:len);
output_args = trapz(x, value, 1);
end

