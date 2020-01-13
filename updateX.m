function Xhat = updateX(comp, Ahat, Y, N , p, lambda)
    % step 0.1: calculate Ahat for random initials
    % multiply Phat to A
    
    % optimize over X 
    % x_temp = solve X from min\|Y - Ahat*X\|_F + lambda * \|X\|_1
    fprintf('    Updating X\n')
    Aflat = kron(eye(p,p),Ahat);
    yflat = Y(:);
    if comp==true
        cvx_begin quiet
            variable Xs(N*p, 1) complex
            minimize norm(Aflat * Xs - yflat, 2) + lambda * norm(Xs, 1)
        cvx_end
    else
        cvx_begin quiet
            variable Xs(N*p, 1)
            minimize norm(Aflat * Xs - yflat, 2) + lambda * norm(Xs, 1)
        cvx_end
    end
    Xhat = reshape(Xs, [N,p]);
%     error = norm(Xhat - X, 'fro') / norm(X, 'fro');
%     fprintf ("    Relative error in X is %f\n", error);
    fprintf ("\n");
    % Xhat(Xhat<0.00000001) = 0
end

