% eleven-point smooth filtering
function y=mea11_1(x,m)
%  x: data
%  m: number of cycles
n=length(x);
a=x;
disp(n)
for k=1: m
b(1) = (15*a(1)+13*a(2)+10*a(4)+4*a(3)+7*a(5)+a(7)-5*a(6)-3*a(8)-2*a(9))/40;
b(2) = (9*a(1)+7*a(2)+5*a(3)+3*a(4)+2*a(5)+a(6)-a(7)-2*a(8)-3*a(9))/15;
b(3) = (9*a(1)+5*a(2)+3*a(3)+7*a(4)+5*a(5)+4*a(6)-3*a(7)-2*a(2)-a(1))/26;
b(4) = (9*a(1)+8*a(2)+7*a(3)+6*a(4)+5*a(5)+4*a(6)+3*a(7)+2*a(2)+a(1))/28;
b(5) = (9*a(1)+8*a(2)+7*a(3)+6*a(4)+5*a(5)+4*a(6)+3*a(7)+2*a(2)+a(1))/28;
 for j=6:n-5
     b (j) = (a(j-5)+a(j-4)+a(j-3) +a(j-2) +a(j-1)+a(j)+a(j+1)+a(j+2)+a(j+3)+a(j+4)+a(j+5))/9;
 end
 b(n-4) = (21*a(n)+17*a(n-1)+15*a(n-2)+11*a(n-3)-7*a(n-4)-3*a(n-5)-a(n-6))/75;
 b(n-3) = (17*a(n)+13*a(n-1)+9*a(n-2)+7*a(n-3)-3*a(n-4)-2*a(n-5)-a(n-6))/40;
b(n-2) = (7*a(n)+6*a(n-1)+5*a(n-2)+4*a(n-3)+3*a(n-4)+2*a(n-5)-a(n-6))/26;
b(n-1) = (5*a(n) +4*a(n-1) +3*a(n-2) +2*a(n-3) +a(n-4)-a(n-6)) /14;
b(n) = (13*a(n) +10*a(n-1) +7*a(n-2) +4*a(n-3)+a(n-4)-2*a(n-5) -5*a(n-6)) /28;
a=b;
end
y =a;
end