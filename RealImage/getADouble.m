function result = getADouble(Bhat1, Bhat2, Phat1, Phat2)
    B1 = multiplicationBinv(Bhat1);
    B2 = multiplicationBinv(Bhat2);
    P1 = multiplicationPinv(Phat1);
    P2 = multiplicationPinv(Phat2);
    % A = (P2*B2) kron (P1 * B1)
    Dinv1 = P1 * B1;
    Dinv2 = P2 * B2;
    result = kron(Dinv2, Dinv1);
end
