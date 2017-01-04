function [cost grad] = soft_cost(theta, instances, labels, lambda)

[s1 s2] = size(instances);
[~, s4] = size(labels);

mat = reshape(theta(1:end-s4), s4, s2)';
b = theta(end-s4+1:end);
vec = mat'*instances';
if s4 > 1
    sm = softmax(vec + b(:,ones(1,s1)));
else
    sm = sigmoid(vec + b(:,ones(1,s1)));
end
lbl = labels';
lbl_sm = sm - lbl;

cost = 1/2*sum(sum(((lbl_sm').*(lbl_sm'))))/s1 + 1/2*lambda*(theta(1:end-s4)'*theta(1:end-s4));

if s4 > 1
    del = (sm.*lbl_sm) - bsxfun(@times,sum(sm.*lbl_sm), sm);
    gradW = del*instances;
    gradb = sum(del,2);
else
    del = (lbl_sm).*sigmoid_prime(sm);
    gradW = del*instances;
    gradb = sum(del);
    
end
grad = [gradW(:); gradb]/s1 + [lambda*theta(1:end-s4); zeros(s4,1)];
