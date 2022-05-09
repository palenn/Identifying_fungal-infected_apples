% seven-point smooth filtering
function y=mean7_1(x,m)
%  x: data
%  m: number of cycles
n=length(x);
a=x;
for k=1: m
b(1) = (13*a(1) +10*a(2) +4*a(4) +7*a(3) +a(5)-2*a(6)-5*a(7))/28;
 b(2) = (5*a(1)+4*a(2)+3*a(3)+2*a(4)+a(5)-a(7))/14;
 b(3)=(7*a(1)+6*a(2)+5*a(3)+4*a(4)+3*a(5)+2*a(6)+a(7))/28;
 for j=4:n-3
     b (j) = (a(j-3) +a(j-2) +a(j-1)+a(j)+a(j+1)+a(j+2)+a(j+3))/7;
 end
 b (n-2) = (7*a(n)+6*a(n-1)+5*a(n-2)+4*a(n-3)+3*a(n-4)+2*a(n-5)+a(n-6))/28;
 b (n-1) = (5*a(n) +4*a(n-1) +3*a(n-2) +2*a(n-3) +a(n-4)-a(n-6)) /14;
 b (n) = (13*a(n) +10* a(n-1) +7*a(n-2) +4*a(n-3)+a(n-4)-2*a(n-5) -5*a(n-6)) /28;
 a=b;
 end

 y =a;
 end