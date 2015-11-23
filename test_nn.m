function [ans] = test_nn(WT1, WT2)

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
train_label = label(r(1:1500),:);
train_data = dataset(r(1:1500),:);
train_data_size=size(train_data,1);

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
W1 = WT1;
W2 = WT2;

alpha = 0.15;
lambda = 0;
epoch = 100;

biastt = ones(size(train_data,1),1);
train_data = [biastt, train_data];    
hid_test = train_data*W1';
z=double(1./(1.0+exp(-1*hid_test)));
       
biastt1 = ones(size(z,1),1);
b = [biastt1 z];
out_test = b*W2';
out_test1=double(1./(1.0+exp(-1*out_test)));  

% [d1,I] = max(out_test1,[],2);
% [d2,I1] = max(label,[],2);
% count = 0;
% for i = 1 :1500
%     if(I(i,1) == I1(i,1))
%         count = count + 1;
%     end
% end
% ans = count/1500;
p_test=[];
 for k=1:size(train_data,1)
           class=out_test1(k,:);
            label_class = find(class==(max(max(class))));
            label_class=label_class-1;
            p_test= [p_test ; label_class];         
 end
fid = fopen('C:\Users\HARSHITH\Documents\MATLAB\classes_nn.txt', 'wt');
    fprintf(fid, [repmat('%g\t', 1, size(p_test,2)-1) '%g\n'], p_test.');
    fclose(fid);
       
ans = mean(double(p_test == train_label));


end