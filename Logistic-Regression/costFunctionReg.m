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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


z = X * theta;
answer = 0;
for i=2:size(theta)
    penalize = theta(i) ^ 2;
    answer = answer + penalize;
end 
optim = lambda/(2*m) * answer;
J = (1/m * ((-y' * log(sigmoid(z))) - ((1-y)'*log(1-sigmoid(z))))) + optim;
%grad(1) = 1/m * ((sigmoid(z(1)) - y(1))' * X(1));
%for j = 2:size(grad)
%    grad(j) = (1/m * ((sigmoid(z(j)) - y(j))' * X(j))) + ((lambda/m) * theta(j));
grad = 1/m * ((sigmoid(z) - y)' * X);
for j = 2:size(theta)
    grad(j) = grad(j) + ((lambda/m) * theta(j));
end
% =============================================================

end
