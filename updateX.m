function Xhat = updateX(Phat, Bhat, Y, N , p, num_of_matrix, lambda)
    % step 0.1: calculate Ahat for random initials
    % multiply Phat to A
    Ahat = get_A(Bhat,Phat);
    % optimize over X 
    % x_temp = solve X from min\|Y - Ahat*X\|_F + lambda * \|X\|_1
    fprintf('    Updating X\n')
    Aflat = kron(eye(p,p),Ahat);
    yflat = Y(:);
    cvx_begin quiet
        variable Xs(N*p, 1) complex
        minimize norm(Aflat * Xs - yflat, 2) + lambda * norm(Xs, 1)
    cvx_end
    
    Xhat = reshape(Xs, [N,p]);
%     error = norm(Xhat - X, 'fro') / norm(X, 'fro');
%     fprintf ("    Relative error in X is %f\n", error);
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




