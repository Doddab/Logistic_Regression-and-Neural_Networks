function [W] = train_lr()
a = load('0_reg.txt');
b = load('1_reg.txt');
c = load('2_reg.txt');
d = load('3_reg.txt');
e = load('4_reg.txt');
f = load('5_reg.txt');
g = load('6_reg.txt');
h = load('7_reg.txt');
i = load('8_reg.txt');
j = load('9_reg.txt');
dataset = vertcat(a,b,c,d,e,f,g,h,i,j);
bias=ones(19978,1);

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

W = rand(513,10);
dataset = double(C);

acti_m = C*W;

exp_acti_m = exp(acti_m);
tmp = exp_acti_m;

for inc = 1:size(exp_acti_m,1)
    tmp(inc,:) = exp_acti_m(inc,:)./sum(exp_acti_m(inc,:));
end

eta = 0.001;
grad_e = [];
grad_e = transpose(C) *(tmp - t);
Acc = [];

hits = 0;
[yindex,Ind] = max(tmp,[],2);
[tindex,Ind1] = max(t,[],2);
for in = 1:19978
	if Ind(in,1) == Ind1(in,1)
	hits = hits + 1;
	end
end
acc_old = hits/19978;


for inde = 1 :100
   W_new = W - eta*grad_e;
   W = W_new;
   acti_m = C*W;
   exp_acti_m = exp(acti_m);
   tmp = exp_acti_m;
   

	for inc = 1:size(exp_acti_m,1)
		tmp(inc,:) = exp_acti_m(inc,:)./sum(exp_acti_m(inc,:));
	end
	
	
	grad_e = transpose(C) *(tmp - t);
	
	[yindex,Ind] = max(tmp,[],2);
	[tindex,Ind1] = max(t,[],2);
	hits = 0;
		for in = 1:19978
			if Ind(in,1) == Ind1(in,1)
			hits = hits + 1;
			end
		end
	acc_new = hits/19978;

	if acc_old < acc_new
		eta = 1.1 * eta;
	else
		eta = eta /2;
	end
	acc_old = acc_new;
end

end
















