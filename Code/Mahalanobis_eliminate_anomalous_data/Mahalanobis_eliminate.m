clc;
clear all;
load data/feature_parameters_data/Aspergillus_niger_features.mat    %load data
data=A;
X=data; % assigned to the X behavior sample, listed as features
X = mapminmax(X',0,1)';
u=mean(X); % average each column
[m,n]=size(X);
for i=1:m
ab=cov(X);% find a covariance matrix
dist(i)=(X(i,:)-u)*inv(ab)*(X(i,:)-u)';% find the Markov distance from each sample to u
end
[a,b]=sort(dist);% sorts the Markov distance, a is the Markov distance, and b is the index
thre=0.02;
T=ceil(m*thre); % set the threshold
Threshold=a(m-T);% set as a threshold value
len=length(a);

index_inlier = 1;
index_outlier = 1;
for i = 1:len % if less than the threshold is the normal point  
    if a(i) < Threshold        
        % inlier
        inlier(index_inlier) = [b(i)];
        index_inlier = index_inlier + 1;
        s=b(i);        
        disp(['正常点序列号：',num2str(s)])
    else
        % outlier
        outlier(index_outlier) = [b(i)];
        index_outlier = index_outlier + 1;
        ns=b(i);
        disp(['离群点序列号：',num2str(ns)]) 
    end
end
inlier_data = data(inlier,:);
outlier_data = data(outlier, :);
% save('x_1m.mat','inlier_data')
az=a'
Dis_out=sqrt(az)
Dis_mean =mean(Dis_out)

Dis_std = std(Dis_out)

Th=(Dis_out-Dis_mean)/Dis_std
bz=b'


