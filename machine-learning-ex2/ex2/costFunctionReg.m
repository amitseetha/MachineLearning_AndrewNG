function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = size (X,2);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
reg=0;

theta1=theta .^2;
for i=2:n,
reg = reg + theta1(i);
end

for i=1:m,
J=J-y(i)*log(1/(1+exp(-(theta(1)*X(i,1)+theta(2)*X(i,2)+theta(3)*X(i,3)))))-(1-y(i))*(log(1-(1/(1+exp(-(theta(1)*X(i,1)+theta(2)*X(i,2)+theta(3)*X(i,3)))))));
end
J=J/m + (lambda*reg)/(2*m);

for j=1:n,
	P=0;
	for i= 1:m,
		P = P+ ((1/(1+exp(-(theta(1)*X(i,1)+theta(2)*X(i,2)+theta(3)*X(i,3)))))-y(i))*X(i,j);
end
grad(j)=P/m;
if j>1,
	grad(j)=grad(j) + (lambda*theta(j))/m;
end


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
