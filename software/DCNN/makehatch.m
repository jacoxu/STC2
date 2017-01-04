function A = makehatch(hatch)
%MAKEHATCH Predefined hatch patterns
%  MAKEHATCH(HATCH) returns a matrix with the hatch pattern for HATCH
%   according to the following table:
%      HATCH        pattern
%     -------      ---------
%        /          right-slanted lines
%        \          left-slanted lines
%        |          vertical lines
%        -          horizontal lines
%        +          crossing vertical and horizontal lines
%        x          criss-crossing lines
%        .          single dots
%
%  See also: APPLYHATCH

%  By Ben Hinkle, bhinkle@mathworks.com
%  This code is in the public domain.

n = 6;
A=zeros(n);
switch (hatch)
case '/'
  A = fliplr(eye(n));
case '\'
  A = eye(n);
case '|'
  A(:,1) = 1;
case '-'
  A(1,:) = 1;
case '+'
  A(:,1) = 1;
  A(1,:) = 1;
case 'x'
  A = eye(n) | fliplr(diag(ones(n-1,1),-1));
case '.'
  A(1:2,1:2)=1;
otherwise
  error(['Undefined hatch pattern "' hatch '".']);
end