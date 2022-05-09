% wavelt_energy
function [ output_args ] = wavelt_energy( data )
[m,n] = size(data);
for i = 1:n
    E = [];
    c=wpdec(data(:,i),3,'db3');
    for j = 1:2^3 % 3,4��ֽ⣬8,16��ϵ��,
            E(j) = sum(abs(wprcoef(c,[3,j-1])).^2); % �������
    end
    output_args(i) = sum(E)/8;
end
end

