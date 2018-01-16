function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
grad_tmp = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

s = sigmoid(X*theta);
J = (1/m) * sum( -y.*log(s) - (1 - y) .* log(1 - s)) + ...
(( sum(theta(2:end).**2)) * lambda/(2*m));

grad_tmp(1) = grad(1) + (1/m) * sum((s-y) .* X(:,1)); 
for i=2:size(theta), 
    grad_tmp(i) = grad(i) + (1/m) * sum((s-y) .* X(:,i)) + ...
    (lambda/m)*theta(i) ; 
end
grad = grad_tmp;




% =============================================================

end
