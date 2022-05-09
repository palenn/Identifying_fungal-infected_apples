[filename,filepath]=uigetfile('*.xlsx','open file');
[A]=xlsread(filename);% select the data you want to import
save Aspergillus_niger_features_Mahalanobis.mat;
