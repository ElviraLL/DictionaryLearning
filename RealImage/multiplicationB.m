function result = mutiplicationB(B)
    % input:  B is a cell 
    % output: result is a matrix of size N * N which is multuplication of B
    num_of_matrix = size(B) * [0;1];
    N = size(B{num_of_matrix}) * [1; 0]; % find the first element of size
    result = eye(N);
    for idx = 1:num_of_matrix
        Bi = B{idx};
        Ii = eye(N/(2^idx));
        Bi = kron(Ii, Bi);
        result = Bi * result;
    end  
end