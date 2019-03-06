function [error_tr, w1, w2, error_test] = TwoLayer_NN_batch_ReLU(w1, w2, learn_rate, lambda, H)
    global num_tr class t_k img_n labels_tr img_test num_test labels_test;   
%   dim(w1) = 785 * 20, dim(w2) = 20 * 10
%   dim(x) = 60000 * 785, dim(y) = 60000 * 20
%   dim(delta_1) = 60000 * 20, dim(delta_2) = 60000 * 10;

    g_tr_temp = img_n * w1;
    g_tr_temp(g_tr_temp < 0) = 0;
    g_tr = ones(num_tr, H+1);
    g_tr(:, 2:H+1) = g_tr_temp;
    u_tr = exp(g_tr * w2);
    est_u_tr = sum(u_tr,2) * ones(1, class);
    est_z_prob = u_tr./est_u_tr;
    
%   Re - calculate the  error rate for the training set   
    [~,idx] = max(est_z_prob,[],2);
    error_tr = sum((idx-1)~=labels_tr)/num_tr;

%   Calculate the error rate for the testing set
    g_test = ones(num_test, H+1);
    g_test(:, 2:H+1) = img_test * w1;
    g_test(g_test < 0) = 0;
    u_test = exp(g_test * w2);
    est_u_test = sum(u_test,2) * ones(1, class);
    est_test_prob = u_test./est_u_test;
    
    [~,idx_test] = max(est_test_prob,[],2);
    error_test = sum((idx_test-1)~=labels_test)/num_test;
    
%   Use backpropagation to update w2 & w1
    delta_2 = est_z_prob - t_k;
    g_tr_grad = g_tr_temp;
    g_tr_grad(g_tr_grad > 0) = 1;
    
    delta_1 = times(g_tr_grad, (delta_2 * w2(2:H+1,:)')); 
    
    update_1 = img_n' * delta_1;
    update_2 = g_tr' * delta_2;
    
    w1 = (1 - lambda)*w1 - learn_rate * update_1;
    w2 = (1 - lambda)*w2 - learn_rate * update_2;
    

   
end

