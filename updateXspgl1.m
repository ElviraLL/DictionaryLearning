function Xhat = updateXspgl1(Bhat, Phat, Y, X0, sigma)
    % step 0.1: calculate Ahat for random initials
    % multiply Phat to A
    [N, p] = size(Y);
    Ahat = get_A(Bhat,Phat);
    % optimize over X 
    % x_temp = solve X from min\|Y - Ahat*X\|_F + lambda * \|X\|_1
    fprintf('    Updating X using spgl1\n')
    Aflat = kron(eye(p,p),Ahat);
    yflat = Y(:);
    options.iterations=500;
    options.verbosity=1;
    %Xs=spgl1(Aflat, yflat, [], sigma, X0(:), options);
    Xs = spg_bpdn(Aflat, yflat, sigma, options);
    Xhat = reshape(Xs, [N,p]);
%     error = norm(Xhat - X, 'fro') / norm(X, 'fro');
%     fprintf ("    Relative error in X is %f\n", error);
    fprintf ("\n");
    % Xhat(Xhat<0.00000001) = 0
end
