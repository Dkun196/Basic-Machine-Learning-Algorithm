function [error_tr,w, error_test] = SingleLayer_NN_batch(w, learn_rate)
global num_tr class t_k img_n dim labels_tr img_test num_test labels_test;
%   Calculate the softmax estimation
    a_k = img_n * w;
    est_y = exp(a_k);
    est_y_sum = sum(est_y,2) * ones(1, class);
    est_y_prob = est_y./est_y_sum;

%   Use backpropagation to update w
    diff = t_k - est_y_prob;
    w = w + learn_rate * img_n' * diff;
    
%   Calculate the error rate for the training set
    a_k = img_n * w;
    est_y = exp(a_k);
    est_y_sum = sum(est_y,2) * ones(1, class);
    est_y_prob = est_y./est_y_sum;
    
    [~,idx] = max(est_y_prob,[],2);
    idx = idx - ones(num_tr,1);
    error_tr = sum(idx~=labels_tr);
    error_tr = error_tr/num_tr;

%   Calculate the error rate for the testing set
    est_test = exp(img_test * w);
    est_test_sum = sum(est_test,2) * ones(1, class);
    est_test_prob = est_test./est_test_sum;
    
    [~,idx_test] = max(est_test_prob,[],2);
    idx_test = idx_test - ones(num_test,1);
    error_test = sum(idx_test~=labels_test);
    error_test = error_test/num_test;
end

