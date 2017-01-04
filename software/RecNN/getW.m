function [W1 W2 W3 W4 b1 b2 b3 Wcat bcat We] = getW(Wcat_flag, theta, hiddenSize, cat_size, dictionary_length)
Wcat = [];
bcat = [];
visibleSize = hiddenSize * 2;
if Wcat_flag
    W1 = reshape(theta(1:hiddenSize*visibleSize/2), hiddenSize, visibleSize/2);
    W2 = reshape(theta(hiddenSize*visibleSize/2+1:hiddenSize*visibleSize), hiddenSize, visibleSize/2);
    W3 = reshape(theta(hiddenSize*visibleSize+1:3/2*hiddenSize*visibleSize), visibleSize/2, hiddenSize);
    W4 = reshape(theta(3/2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize/2, hiddenSize);
    b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
    b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:2*hiddenSize*visibleSize+2*hiddenSize);
    b3 = theta(2*hiddenSize*visibleSize+2*hiddenSize+1:2*hiddenSize*visibleSize+3*hiddenSize);
    Wcat = reshape(theta(2*hiddenSize*visibleSize+3*hiddenSize+1: 2*hiddenSize*visibleSize+3*hiddenSize+cat_size*hiddenSize), cat_size, hiddenSize);
    bcat = theta( 2*hiddenSize*visibleSize+3*hiddenSize+cat_size*hiddenSize+1: 2*hiddenSize*visibleSize+3*hiddenSize+cat_size*hiddenSize+cat_size);
    We = reshape(theta(2*hiddenSize*visibleSize+3*hiddenSize+cat_size*hiddenSize+cat_size+1: end),hiddenSize,dictionary_length);
else
    W1 = reshape(theta(1:hiddenSize*visibleSize/2), hiddenSize, visibleSize/2);
    W2 = reshape(theta(hiddenSize*visibleSize/2+1:hiddenSize*visibleSize), hiddenSize, visibleSize/2);
    W3 = reshape(theta(hiddenSize*visibleSize+1:3/2*hiddenSize*visibleSize), visibleSize/2, hiddenSize);
    W4 = reshape(theta(3/2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize/2, hiddenSize);
    b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
    b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:2*hiddenSize*visibleSize+2*hiddenSize);
    b3 = theta(2*hiddenSize*visibleSize+2*hiddenSize+1:2*hiddenSize*visibleSize+3*hiddenSize);
    We = reshape(theta(2*hiddenSize*visibleSize+3*hiddenSize+1: end),hiddenSize,dictionary_length);
end
