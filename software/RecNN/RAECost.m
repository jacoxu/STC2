function [cost_total,grad_total] = RAECost(theta, alpha_cat, cat_size, beta, dictionary_length, hiddenSize, ...
    lambda, We_orig, data_cell, labels, freq_orig, sent_freq, f, f_prime)

[~, ~, ~, ~, ~, ~, ~, Wcat, bcat, We] = getW(1, theta, hiddenSize, cat_size, dictionary_length);

szWe = length(We(:));
szbcat = length(bcat(:));
szWcat = length(Wcat(:));
theta1 = theta;
theta1(end-szWe+1:end) = We;
theta1(end-szWe-szbcat-szWcat+1:end-szWe) = [];


theta2 = theta;
theta2(end-szWe+1:end) = We_orig(:) + We(:);
lambda2 = lambda;
lambda(3) = lambda2(4);
lambda(4) = lambda2(3);

% disp('DEBUGGING: DELETE AFTERWARDS')
% data_cell=data_cell(1:2);
% update W using Greedy Unsupervised RAE
[costRAE, gradRAE, allKids] = computeCostAndGradRAE([], theta1, 0, alpha_cat, cat_size, beta, dictionary_length, hiddenSize, ...
    (alpha_cat)*lambda, We_orig , data_cell, labels, freq_orig, f, f_prime);


WegradRAE = gradRAE(end-szWe+1:end);
gradRAE(end-szWe+1:end) = 0;
gradRAE = [gradRAE; zeros(szbcat+szWcat,1)];
gradRAE(end-szWe+1:end) = WegradRAE;

cost_total =  costRAE;
grad_total =  gradRAE;
