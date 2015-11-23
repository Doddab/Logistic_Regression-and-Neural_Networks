function [acc_new] = test_nn(WT)

a = load('0_tst.txt');
b = load('1_tst.txt');
c = load('2_tst.txt');
d = load('3_tst.txt');
e = load('4_tst.txt');
f = load('5_tst.txt');
g = load('6_tst.txt');
h = load('7_tst.txt');
i = load('8_tst.txt');
j = load('9_tst.txt');
dataset = vertcat(a,b,c,d,e,f,g,h,i,j);
bias=ones(1500,1);

C = horzcat(bias,dataset);

svector = [];
svector = [size(a,1), size(b,1), size(c,1), size(d,1), size(e,1), size(f,1), size(g,1), size(h,1), size(i,1), size(j,1)];
svec = size(svector,2);
counter = 1;
for m = 1:svec
	for n = 1:svector(:,m)
	t(counter,m) = 1;
	counter = counter + 1;
	end
end

W = WT;
dataset = double(C);

acti_m = C*W;

exp_acti_m = exp(acti_m);
tmp = exp_acti_m;

for inc = 1:size(exp_acti_m,1)
    tmp(inc,:) = exp_acti_m(inc,:)./sum(exp_acti_m(inc,:));
end

hits = 0;
[yindex,Ind] = max(tmp,[],2);
[tindex,Ind1] = max(t,[],2);


for in = 1:1500
	if Ind(in,1) == Ind1(in,1)
	hits = hits + 1;
	end
end
acc_old = hits/1500;

acc_new = acc_old;

for i=1:1500
    Ind(i,1) = Ind(i,1) - 1;
end
fid = fopen('C:\Users\HARSHITH\Documents\MATLAB\classes_lr.txt', 'wt');
    fprintf(fid, [repmat('%g\t', 1, size(Ind,2)-1) '%g\n'], Ind.');
    fclose(fid);
end








































