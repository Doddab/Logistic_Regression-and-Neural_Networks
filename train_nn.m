function [W1,W2] = train_nn()
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

t0 = repmat(0,size(a,1),1);
t1 = repmat(1,size(b,1),1);
t2 = repmat(2,size(c,1),1);
t3 = repmat(3,size(d,1),1);
t4 = repmat(4,size(e,1),1);
t5 = repmat(5,size(f,1),1);
t6 = repmat(6,size(g,1),1);
t7 = repmat(7,size(h,1),1);
t8 = repmat(8,size(i,1),1);
t9 = repmat(9,size(j,1),1);
label = [t0;t1;t2;t3;t4;t5;t6;t7;t8;t9];

r = randperm(size(dataset,1));
train_label = label(r(1:15982),:);
train_data = dataset(r(1:15982),:);
train_data_size=size(train_data,1);

validation_label = label(r(15983:end),:);
validation_data= dataset(r(15983:end),:);
ipn = size(train_data, 2); 

train_label_enc=[];
    for k = 1 : size(train_label,1)
        if(train_label(k))== 0
         trainlabel = [1 0 0 0 0 0 0 0 0 0];
         elseif (train_label(k)== 1)
          trainlabel = [0 1 0 0 0 0 0 0 0 0];
         elseif (train_label(k)== 2)
         trainlabel = [0 0 1 0 0 0 0 0 0 0];
         elseif (train_label(k)== 3)
         trainlabel = [0 0 0 1 0 0 0 0 0 0];
         elseif (train_label(k)== 4)
         trainlabel = [0 0 0 0 1 0 0 0 0 0];
         elseif (train_label(k)== 5)
         trainlabel = [0 0 0 0 0 1 0 0 0 0];
         elseif (train_label(k)== 6)
         trainlabel = [0 0 0 0 0 0 1 0 0 0];
         elseif (train_label(k)== 7)
         trainlabel = [0 0 0 0 0 0 0 1 0 0];
         elseif (train_label(k)== 8)
         trainlabel = [0 0 0 0 0 0 0 0 1 0];
          else
         trainlabel = [0 0 0 0 0 0 0 0 0 1];
         end
         train_label_enc = [train_label_enc; trainlabel];
    end
    
hidden_node = 200;		   
output_node = 10;		   
bias = ones(size(train_data,1),1);
X = horzcat(bias,train_data);
W1 = rand(200,513)*0.2-0.1;
W2 = rand(10,201)*0.5-0.25;

alpha = 0.15;
lambda = 0;
epoch = 100;

for j = 1 : epoch
        acti_hid = X*W1';
        z = double(1./(1.0+exp(-1*acti_hid)));
        bias1 = ones(size(z,1),1);
        b = [bias1 z];
        b1=b*transpose(W2);
        acti_out = double(1./(1.0+exp(-1*b1)));       
		delta_w2 = acti_out - train_label_enc;
		grad_w2 = delta_w2'*b;
		grad_w2_reg= (lambda*W2);
		grad_w2= (grad_w2 + grad_w2_reg)/train_data_size;
        delta_w1 = acti_out - train_label_enc;
		del_wt=delta_w1*W2;
		b_mul= (1-b).*b;
		p_out=b_mul.*del_wt;
		grad_w1_reg = p_out'*X;
		grad_w1=(grad_w1_reg(1:(size(grad_w1_reg,1)-1),:));
    	grad_w1_reg = (lambda*W1);
		grad_w1= (grad_w1 + grad_w1_reg)/train_data_size;
    	W2=(W2-(alpha*grad_w2));
		W1=(W1-(alpha*grad_w1));
end

biastt = ones(size(validation_data,1),1);
validation_data = [biastt, validation_data];    
hid_test = validation_data*W1';
z=double(1./(1.0+exp(-1*hid_test)));
       
biastt1 = ones(size(z,1),1);
b = [biastt1 z];
out_test = b*W2';
out_test1=double(1./(1.0+exp(-1*out_test)));       

p_test=[];
 for k=1:size(validation_data,1)
           class=out_test1(k,:);
            label_class = find(class==(max(max(class))));
            label_class=label_class-1;
            p_test= [p_test ; label_class];         
 end
       
 ans = mean(double(p_test == validation_label));
 
end
 
     
     