% Feb.23th 2019 Homework 3 for ECE 271B, Prblem 5
% Code written by Yudi Wang
% Apply boosting to classify digits
%%
%Initial Setup
clc;clear;close all;
num_train = 20000;
num_test = 10000;
dim = 784;
class = 10;
thresh = 0:1/50:1;
% Load the train & test data matrix
[imgs,labels_train] = readMNIST('train-images-idx3-ubyte',...
    'train-labels-idx1-ubyte', num_train, 0);
[imgs_t,labels_test] = readMNIST('t10k-images-idx3-ubyte',...
    't10k-labels-idx1-ubyte', num_test, 0);
% Build 10 binary label matrix, dim(y_train) = 20000*10, dim(y_test) = 20000*10, 
y_train = -1*ones(num_train,class);
y_test = -1*ones(num_test,class);
for c = 1: class
    for i = 1 : num_train
        if((labels_train(i, 1)+1)==c)
            y_train(i,c) = 1;
        end
    end
    for i = 1 : num_test
        if((labels_test(i, 1)+1)==c)
            y_test(i,c) = 1;
        end
    end
end
% weak learner and its twin
u_xjt = @(x,j,t) 1.*(x(:,j)>=t)+(-1).*(x(:,j)<t);% dim(x)=20000*784
u_xjt_twin = @(x,j,t) 1.*(x(:,j)<t)+(-1).*(x(:,j)>=t);% dim(x)=20000*784
% weight for the weak learner
w_t = @(epsilon) 0.5*log((1-epsilon)/epsilon);
iterlist = [5, 10, 50, 100, 250];
%%
iter_time = 250;
g_x_final_train = zeros(num_train,class);
g_x_final_test = zeros(num_test,class);
max_idx = zeros(iter_time,class);
margin_train = zeros(num_train,5,class);
weak_learner = ones(dim,class);
for c = 1 : 10% for each first binary classifier
    y_train_tmp = y_train(:,c);y_test_tmp = y_test(:,c);
    w_x_init = ones(num_train,1)/num_train; w_x = w_x_init; % initial the sample weight
    g_x_res_train = zeros(num_train,1);%the estimation for the ensembled weak learner
    g_x_res_test = zeros(num_test,1);
    train_error = zeros(iter_time,1); 
    test_error = zeros(iter_time,1);     
    k=1;
    weak_learner = 128*ones(dim,1);
    for T = 1:iter_time%Iteration
        disp(T);
        error_min = 1e10;
        j_weak = 0;
        t_weak = 0;
        % Pick up the best weak learner
        for j = 1 : dim
            for t = 0:0.02:1
                temp_train = u_xjt(imgs,j,t);
                error_train_weight = sum(w_x(temp_train~=y_train_tmp));
                error_temp_train =sum(temp_train~=y_train_tmp)/num_train;
                if(error_temp_train > 0.5)%use the twin weak learner
                    error_temp_train = 1 - error_temp_train;
                    temp_train = u_xjt_twin(imgs,j,t);
                    error_train_weight = sum(w_x(temp_train~=y_train_tmp));
                    if(error_train_weight < error_min)%update the best weak learner
                        j_weak = j; t_weak = t; flag = -1;
                        error_min = error_train_weight; 
                        weak_result_train = temp_train;
                        weak_result_test = u_xjt_twin(imgs_t,j,t);

                    end
                else
                    if(error_train_weight < error_min)%update the best weak learner
                        j_weak = j;t_weak = t; flag = 1;
                        error_min = error_train_weight;
                        weak_result_train = temp_train;
                        weak_result_test = u_xjt(imgs_t,j,t);
                    end
                end
            end
        end
        
        if(flag == 1)% Regular learner
            weak_learner(j_weak,1) = 255;
        else % Twin learner
            weak_learner(j_weak,1) = 0;
        end
        epsilon = error_min/sum(w_x);
        w_t_weak = w_t(epsilon);% Weight for the weak learner in the final result
        % for the train error
        g_x_res_train = g_x_res_train + (w_t_weak*weak_result_train);
        train_a=-1*ones(num_train,1);train_a(g_x_res_train>0) = 1;
        train_error(T,1)=sum(train_a~=y_train_tmp)/num_train;
        %for the test error
        g_x_res_test = g_x_res_test + (w_t_weak*weak_result_test);
        test_a=-1*ones(num_test,1);test_a(g_x_res_test>0) = 1;
        test_error(T,1)=sum(test_a~=y_test_tmp)/num_test;
        % Reweight the sample, emphasis the hard sample
        margin_temp = y_train_tmp.*g_x_res_train;
        w_x = exp(-margin_temp);
        [~,max_idx(T,c)]=max(w_x);
        
        if (ismember(T,iterlist))
            margin_train(:,k,c) = margin_temp;
            k=k+1;
        end
    end
    g_x_final_train(:,c) = g_x_res_train;
    g_x_final_test(:,c) = g_x_res_test;
    
    % Plot the train/test error vs. iteration
    figure();
    plot(train_error,'k');
    hold on;
    plot(test_error,'r');
    legend('Train','Test');
    xlabel('Iteration times');
    ylabel('Error Rate');
    title(['AdaBoost - PoE vs. iteration' ,'Binary Classifier ',num2str(c)]);
    saveas(gcf,['AdaBoost - PoE vs. iteration' ,'Binary Classifier ',...
        num2str(c),'.tif']);
    
    % Plot the weak learner
    figure();
    weak_show = reshape(weak_learner,28,28);
    imshow(weak_show,[]);
    title(['Weak learner, Binary Classifier ',num2str(c)]);
    saveas(gcf,['Weak learner, Binary Classifier ', num2str(c),'.tif']);  
end
%%
% Test error for the final classifier

g_x_final_train(:,1:5) = g_x_final_train1(:,1:5);
g_x_final_train(:,6:10) = g_x_final_train2(:,6:10);
g_x_final_test(:,1:5) = g_x_final_test1(:,1:5);
g_x_final_test(:,6:10) = g_x_final_test2(:,6:10);
[~,y_est_train] = max(g_x_final_train, [], 2);
[~,y_est_test] = max(g_x_final_test, [], 2);
final_error_train = sum((y_est_train-1)~=labels_train)/num_train;
final_error_test = sum((y_est_test-1)~=labels_test)/num_test;
disp(["Final train error is", final_error_train]);
disp(["Final test error is ", final_error_test]);

%%
% Each binary classifier - cdf of the margins after {5,10,50,100,250}
% iterations
margin_train(:,:,1:5) = margin_train1(:,:,1:5);
margin_train(:,:,6:10) = margin_train2(:,:,6:10);
for c = 1 : 10
    figure();
    for k = 1:5
        temp_margin = squeeze(margin_train(:,k,c));
        cdfplot(temp_margin);
        hold on;
    end
    legend('t = 5','t = 10','t = 50','t = 100','t = 250');
    title(['CDF-Binary Classifier ',num2str(c)]);
    saveas(gcf,['CDF-Binary Classifier ',num2str(c),'.tif']);
end
%%
% Plot of the index of the example of the largest weight during the
% boosting iteration
max_idx(:,1:5) = max_idx1(:,1:5);
max_idx(:,6:10) = max_idx2(:,6:10);

for c = 1 : 10
    figure();
    idx_temp = squeeze(max_idx(:,c));
    plot(idx_temp);
%     hold on;
    xlabel('Iteration times');
    ylabel('Example index with largest weight');
%     legend('1','2','3','4','5','6','7','8','9','10');
    title(['Example index of largest weight-Binary Classifier ',num2str(c)]);
    saveas(gcf,['Example index of largest weight-Binary Classifier ',num2str(c),'.tif']);
%     title('Example index of largest weight in 10 binary Classifier ');
%     saveas(gcf,'Example index of largest weight in 10 binary Classifier.tif');
end

  
%%
for c = 1:10
    idx_temp = squeeze(max_idx(:,c));
    [n, bin]= hist(idx_temp,unique(idx_temp));
    [~,idx] = sort(-n);
    Freq = bin(idx);
    figure();
    for i = 1:3
        fig =imgs(Freq(i,1),:);
        pic = reshape(fig,28,28);
        subplot(1,3,i);
        imshow(pic,[]);
    end
    title(['Three Heaviest example, Binary Classifier ',num2str(c)]);
    saveas(gcf,['Three Heaviest example, Binary Classifier ',num2str(c),'.tif']);
end







