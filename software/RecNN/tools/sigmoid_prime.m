function sigmoid_prime = sigmoid_prime(a)
    %a = sigmoid(input);
    sigmoid_prime = a.*(1-a);
    
end