% three-point smooth filtering
function y=mean3_1(x,m)
%  x: data
%  m: number of cycles
 n=length(x);
 a=x;
 for k=1: m
b(1) = (5*a(1) +2*a(2)-a(3))/6;
 for j=2:n-1
     b (j) = (a(j-1) +a(j) +a(j+1))/3;
 end
 b (n) = (5*a(n) +2*a(n-1) -a(n-2))/6;
 a=b;
 end
 y =a;
 end