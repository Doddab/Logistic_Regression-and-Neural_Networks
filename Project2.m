[W] = train_lr();
[acc_new] = test_lr(W);
[W1,W2] = train_nn();
[ans] = test_nn(W1,W2);

yourubitname = 'harshith';
yournumber = 50134007;
fprintf('My UBIT name is %s\n',yourubitname);
fprintf('My Student number is %d\n',yournumber);
fprintf('Accuracy of Logistic Regression is :%4.2f\n', acc_new*100);
Error = 1-acc_new;
fprintf('Error rate of Logistic Regression is:%4.2f\n', Error*100);
fprintf('Accuracy of Neural Network is :%4.2f\n',ans*100);
Error_nn = 1-ans;
fprintf('Error rate of Neural Network is :%4.2f\n',Error_nn*100);


