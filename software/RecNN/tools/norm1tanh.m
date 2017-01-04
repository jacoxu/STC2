function out = norm1tanh(x)
    %temp = tanh(x);
    %out = temp./norm(temp);
    out = tanh(x); % don't divide by norm
end