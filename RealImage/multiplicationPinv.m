function result = multiplicationPinv(Pinvhat)
    % output: result is a matrix of size N * N which is multuplication of B
    % from the smallest size (I kron 2*2 P) to the full size P
    num_of_matrix = size(Pinvhat) * [0;1];
    N = size(Pinvhat{num_of_matrix}) * [1; 0]; % find the first element of size
    result = eye(N);
    for idx = 1:num_of_matrix
        Pi = Pinvhat{idx};
        Ii = eye(N/(2^idx));
        Pi = kron(Ii, Pi);
        result = Pi * result;
    end  
end
