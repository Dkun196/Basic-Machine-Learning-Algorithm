function [error,w] = SingleLayer_NN(w, learn_rate)
    global num class t_k img_n dim labels;
        a_k = img_n * w;
        est_y = exp(a_k);
        est_y_sum = sum(est_y,2) * ones(1, class);
        est_y_prob = est_y./est_y_sum;
        E_w_grad = zeros(num, class, dim + 1);
        for n = 1:num
            for k = 1:class
                y_kn_temp = est_y_prob(n, k);
                t_k_temp = t_k(n, k);
                x_temp = img_n(n, :);
                E_w_temp = (t_k_temp - y_kn_temp) * x_temp; 
                E_w_grad(n, k, :) = E_w_temp;
            end
        end
        w_old = w;
        for k = 1:class
            E_w_temp = squeeze(E_w_grad(:, k, :));
            w(:, k) = w_old(:,k) + learn_rate * (sum(E_w_temp,1))';
        end
        
        [~,idx] = max(est_y_prob,[],2);
        error = sum(idx~=labels);
        error = error/num;
   
end

