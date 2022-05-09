% load data
path = 'data\raw_data';
file_list = dir([path '/' '*.mat']);
choose = [];
for i = 1:length(file_list)
    load([path '/' file_list(i).name])
    index_1 = 2; % sensor1
    index_2 = 6; % sensor5
    thre = 1; % œ‡À∆µƒ÷µ
    precent_error = 0.1; %  error rate, such as 90% error, is considered a sample error
    start_time = 150;
  
    sub_value = data(start_time:300,index_1) - data(start_time:300,index_2);
    abs_sub_value = abs(sub_value);
    in_thre_num = sum(abs_sub_value < thre)
    %  checked
    choose(i) = in_thre_num/length(sub_value) >= (1-precent_error);
    choose_file(i) = {[path '/' file_list(i).name]};
end