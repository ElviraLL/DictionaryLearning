function result = multiplicationBinv(Binvhat)
    num_of_matrix = size(Binvhat) * [0;1];
    N = size(Binvhat{num_of_matrix}) * [1; 0]; % find the first element of size
    result = eye(N);
    for idx = num_of_matrix:-1:1
        Bi = Binvhat{idx};
        Ii = eye(N/(2^idx));
        Bi = kron(Ii, Bi);
        result = Bi * result;
    end
end
