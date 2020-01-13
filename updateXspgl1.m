function Xhat = updateXspgl1(Ahat, Y, X0, sigma, tau)
    % step 0.1: calculate Ahat for random initials
    % multiply Phat to A
    [N, p] = size(Y);
    % optimize over X 
    % x_temp = solve X from min\|Y - Ahat*X\|_F + lambda * \|X\|_1
    fprintf('    Updating X using spgl1\n')
    Aflat = kron(eye(p,p),Ahat);
    yflat = Y(:);
    options.iterations=500;
    options.verbosity=0;
    Xs = spgl1(Aflat, yflat, [], sigma, X0(:), options);
%     Xs = spg_bpdn(Aflat, yflat, sigma, options);
%     Xs = spg_lasso(Aflat, yflat, tau, options);
    Xhat = reshape(Xs, [N,p]);
%     error = norm(Xhat - X, 'fro') / norm(X, 'fro');
%     fprintf ("    Relative error in X is %f\n", error);
    fprintf ("\n");
    % Xhat(Xhat<0.00000001) = 0
end
