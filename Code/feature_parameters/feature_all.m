% each sensor sequentially extracts the first feature, the second feature, and so on
clear all
data_path = ''; % data path
file_list = dir([data_path, '/x*.mat']);
features = []
end_point = 300;
for i = 1:length(file_list)
    load([data_path '/' file_list(i).name])
    data=data(31:end_point,1:end); 
    inv_data = integral_value(data, 150); % integral value
    max_gra = max_gradient( data ); % maximum gradient value
    mean_gra = mean_gradient(data); % mean gradient value
    var_data = variance_value(data); % variance value
    energy_data = wavelt_energy(data); % energy value
    rsav_data = RSAV(data, 200); % relative stable value
    sample_feature = [inv_data, max_gra, mean_gra, var_data, energy_data, rsav_data];
    features = [features;sample_feature];
end
save('all_features.mat','features')