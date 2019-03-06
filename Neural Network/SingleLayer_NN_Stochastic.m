function [error_tr,w, error_test] = SingleLayer_NN_Stochastic(w, learn_rate)
global num_tr class t_k img_n dim labels_tr img_test num_test labels_test;
 
%   Calculate the softmax estimation
    est_y = exp(img_n * w);
    est_y_sum = sum(est_y,2) * ones(1, class);
    est_y_prob = est_y./est_y_sum;

%   Use backpropagation to update w
    diff = t_k - est_y_prob;
    
    for n = 1:num_tr
        for k = 1:class
            w(:, k) = w(:,k) + learn_rate * (diff(n,k) * img_n(n, :))';
        end
    end
   
 %   Re - calculate the softmax estimation
        est_y = exp(img_n * w);
        est_y_sum = sum(est_y,2) * ones(1, class);
        est_y_prob = est_y./est_y_sum;

    %   Calculate the error rate for the training set
        [~,idx] = max(est_y_prob,[],2);
        error_tr = (sum((idx-1)~=labels_tr))/num_tr;

    %   Calculate the error rate for the testing set
        est_test = exp(img_test * w);
        est_test_sum = sum(est_test,2) * ones(1, class);
        est_test_prob = est_test./est_test_sum;

        [~,idx_test] = max(est_test_prob,[],2);
        error_test = (sum((idx_test-1)~=labels_test))/num_test;
end



