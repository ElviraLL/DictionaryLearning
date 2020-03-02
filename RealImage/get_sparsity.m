function result = get_sparsity(image)
    result = (norm(image(:),1).^2./norm(image(:),2).^2);
end