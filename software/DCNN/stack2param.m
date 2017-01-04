function [m1,m2,m3,m4,m5,m6,m7,m8] = stack2param(stack, decodeInfo)

num=0;

m1 = reshape(stack(num+1:num+prod(decodeInfo(1,:))),decodeInfo(1,1),decodeInfo(1,2));
num = num+prod(decodeInfo(1,:));

m2 = reshape(stack(num+1:num+prod(decodeInfo(2,:))),decodeInfo(2,1),decodeInfo(2,2));
num = num+prod(decodeInfo(2,:));

m3 = reshape(stack(num+1:num+prod(decodeInfo(3,:))),decodeInfo(3,1),decodeInfo(3,2));
num = num+prod(decodeInfo(3,:));

m4 = reshape(stack(num+1:num+prod(decodeInfo(4,:))),decodeInfo(4,1),decodeInfo(4,2));
num = num+prod(decodeInfo(4,:));

m5 = reshape(stack(num+1:num+prod(decodeInfo(5,:))),decodeInfo(5,1),decodeInfo(5,2));
num = num+prod(decodeInfo(5,:));

m6 = reshape(stack(num+1:num+prod(decodeInfo(6,:))),decodeInfo(6,1),decodeInfo(6,2));
num = num+prod(decodeInfo(6,:));

m7 = reshape(stack(num+1:num+prod(decodeInfo(7,:))),decodeInfo(7,1),decodeInfo(7,2));
num = num+prod(decodeInfo(7,:));

m8 = reshape(stack(num+1:num+prod(decodeInfo(8,:))),decodeInfo(8,1),decodeInfo(8,2));
num = num+prod(decodeInfo(8,:));