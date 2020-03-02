function result = get_ave_sparsity(X)
result = mean((norms(X,1)./norms(X,2)).^2);
end