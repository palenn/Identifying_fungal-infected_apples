% maindir: the original data path
maindir = '';
% objdir: smoothed data storage path
objdir = '';
subdirpath = fullfile( maindir, '*.mat' );
dat = dir( subdirpath );              

for j = 1 : length( dat )
    d = []
    datpath = fullfile( maindir, dat( j ).name);
    
    load(datpath);
    %disp(length(data(:, 9)))
    for k = 2:9
    %  The 9-point smoothing filter is processed here. You can change mean9_1 to mean3_1,
    %  mean5_3, mean7_1, and mean11_1 to obtain 3-point, 5-point, 7-point, and 11-point smoothed
    %  and filtered data as needed.
    data_1 = mean9_1(data(:,k), 3); 
    d = [d, data_1]
    end
    P = reshape(d, [length(d)/8, 8])
    objpath = fullfile(objdir,dat(j).name);
    save(objpath(1:end-4), 'P');
end
