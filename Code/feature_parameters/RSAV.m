% relative stable value
function [ output_args ] = RSAV( value, tstart)
static_value = value(tstart:end,:);
output_args = mean(static_value, 1);

end

