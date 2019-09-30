function Xhat = updateX(Phat, Bhat, Y, N , p, num_of_matrix, lambda, X)
    % step 0.1: calculate Ahat for random initials
    % multiply Phat to A
    Ahat = eye(N);
    for idx = num_of_matrix: -1: 1
        Pi = Phat{idx};
        Ii = eye(N/(2^idx));
        Pi = kron(Ii, Pi);
        Ahat = Pi * Ahat;
    end
    
    %multiply Bhat to Ahat
    for idx = 1:num_of_matrix
        Bi = Bhat{idx};
        % fprintf("size of bi is %d, ", size(Bi)*[1;0])
        Ii = eye(N/(2^idx));
        Bi = kron(Ii, Bi);
        % fprintf("size of Ii is %d, size of Bi is %d, size of Ahat is %d\n", size(Ii)*[1;0], size(Bi)*[1;0], size(Ahat)*[1;0])
        Ahat = Bi * Ahat;
    end
    
    % optimize over X 
    % x_temp = solve X from min\|Y - Ahat*X\|_F + lambda * \|X\|_1
    fprintf('    Updating X\n')
    Aflat = kron(eye(p,p),Ahat);
    yflat = Y(:);
    cvx_begin quiet
        variable Xs(N*p, 1)
        minimize norm(Aflat * Xs - yflat, 2) + lambda * norm(Xs, 1)
    cvx_end
    
    Xhat = reshape(Xs, [N,p]);
    error = norm(Xhat - X, 'fro') / norm(X, 'fro');
    fprintf ("    Relative error in X is %f\n", error);
    fprintf ("\n");
    % Xhat(Xhat<0.00000001) = 0
end

% testing result: only lambda = 0, optimal value is zero
% i = 1;
% lambdas = [0:0.00001:0.0001, 0.001, 0.01, 0.1, 1];
% for lambda = lambdas
%     cvx_begin 
%         variable Xs(N*p, 1)
%         minimize norm(Aflat * Xs - yflat, 2) + lambda * norm(Xs, 1)
%     cvx_end
%     inf_norm(i)=norm(Xs,'inf')
%     Xhat = reshape(Xs, [N,p]);
%     X_est{i} = Xhat;
%     X0 = reshape(X0,[N,p]);
%     error(i) = norm(Xhat - X0, 'fro') / norm(X0, 'fro')
%     i = i + 1
% end
% 
% for i = 1:15
%     temp = X_est{i};
%     temp(temp<0.0000001) = 0;
%     X_sparse{i} = temp;
% end




