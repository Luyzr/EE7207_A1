clc;
clear all;
close all;
%SVM对线性不可分的数据进行处理
%在选择核函数时，尝试用linear以外的rbf,quadratic,polynomial等，观察获得的分类情况
%训练数据
S1 = load('data_train.mat');
train_data = cell2mat(struct2cell(S1));

%训练数据分类情况
S2 = load('label_train.mat');
label_train = cell2mat(struct2cell(S2));
                       
%分类数据（交给老师的
S3 = load('data_test.mat');
test_results = cell2mat(struct2cell(S3)); 

% %%k折交叉验证
% [M,N] = size(train_data);%数据集为一个M*N的矩阵，其中每一行代表一个样本
% indices = crossvalind('Kfold',M,10);%进行随机分包
% accu = zeros(1,10);
% for i = 1:10
%     test = (indices == i); train = ~test;    %分别取第1、2、...、10份为测试集，其余为训练集
%     svmstruct =fitcsvm(train_data(train,:),label_train(train,:),'Standardize',true,'KernelFunction','gaussian','KernelScale','auto');
% 
%     %分类
%     y1 = predict(svmstruct,train_data(test,:));
%     validation_label = label_train(test);
%     same_count = 0;
%     for j=1:length(y1)
%         if y1(j)==validation_label(j)
%             same_count = same_count + 1;
%         end
%     end
%     accu(i) = same_count/length(y1);
% end
% mean_accu = mean(accu)

%训练分类模型
% svmModel = svmtrain(train,group,'kernel_function','rbf','showplot',true);
%svmstruct =fitcsvm(train_data,label_train,'Standardize',true,'KernelFunction','gaussian','KernelScale','auto');

svmstruct = fitcsvm(train_data,label_train,'KernelFunction','gaussian','OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus'))


% 对SVM分类器进行交叉验证。 默认情况下，该软件使用10倍交叉验证。CVSVMModel是一个ClassificationPartitionedModel交叉验证的分类器。
CVSVMModel = crossval(svmstruct);

% 计算样本外分类错误率和准确率。
classLoss = kfoldLoss(CVSVMModel);
disp('验证准确率:')
acc = (1-classLoss)*100

%分类
% y1 = predict(svmstruct,test_results);
% same_count = 0;
% for i=1:length(y1)
%     if y1(i)==label_train(i)
%         same_count = same_count + 1;
%     end
% end
% accu = same_count/length(y1)
