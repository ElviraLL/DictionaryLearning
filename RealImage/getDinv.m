function result = getDinv(Binvhat,Pinvhat)
    P = multiplicationPinv(Pinvhat);
    B = multiplicationBinv(Binvhat);
    result = P * B;
end

