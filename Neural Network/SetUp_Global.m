function [] = SetUp_Global()
global num_tr dim class img_n labels_tr t_k img_test num_test labels_test test_t;

num_tr = 60000;
num_test = 10000;
dim = 785;
class = 10;
%%
% % Load the train data matrix
    Train_img = 'hw2data/training set/train-images-idx3-ubyte';
    Train_label = 'hw2data/training set/train-labels-idx1-ubyte';
    % img matrix is 60000*784, each row is a 784-dimensional vector
    % label matrix is 60000*1, the value in each row vary from [0,9], so there
    [imgs,labels_tr] = readMNIST(Train_img, Train_label, num_tr, 0);

%     load('/Users/yudiwang/Documents/19Winter ECE 271B/HW2/hw2data_label1.mat');
%     load('/Users/yudiwang/Documents/19Winter ECE 271B/HW2/hw2data_mat1.mat');
%     labels_tr = labels;
    
% initialaze the need vector for the looping, dim(img_n)=60000*785
    img_n = zeros(num_tr, dim);
    img_n(:, 1) = ones(num_tr, 1);
    img_n(:, 2:dim) = imgs;


%%
% Build the label matrix, dim(t_k) = 60000*10
    t_k = zeros(num_tr,class);
    for i = 1 : num_tr
        t_k(i,(labels_tr(i, 1)+1)) = 1;
    end
    
% Load the test data matrix
    Test_img = 'hw2data/test set/t10k-images-idx3-ubyte';
    Test_label = 'hw2data/test set/t10k-labels-idx1-ubyte';
    [imgs_t,labels_test] = readMNIST(Test_img, Test_label, num_test, 0);
    
% initialaze the need vector for the looping, dim(img_n)=60000*785
    img_test = zeros(num_test, dim);
    img_test(:, 1) = ones(num_test, 1);
    img_test(:, 2:dim) = imgs_t;
    
    test_t = zeros(num_test,class);
    for i = 1 : num_test
        test_t(i,(labels_test(i, 1)+1)) = 1;
    end
    
    
end

