function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


%first write forward propogation for all nodes here, then you will get h(k). Also modify y vector to give y(k) in terms of 1 or 0
X = [ones(m, 1) X];


%Modifying vector y into a matrix of 1 and 0 as needed
K=y;
y=zeros(m,num_labels);

for i=1:m,
y(i,K(i))=1;
end


%loop start

for i = 1:m,

a2 = sigmoid(Theta1*(X(i,:))');
a2=[1; a2];
a3 = sigmoid(Theta2*(a2));


P=0;
for j=1:num_labels,

P=P-(y(i,j)*log(a3(j))+(1-y(i,j))*log(1-a3(j)));

end
J=J+P;

%backpropagation

delta3 = a3 - y(i,:)'; %Sure this is correct 
delta2 = (Theta2'*delta3) .*(a2 .*(1-a2)); %80 percent sure this is correct
delta2 = delta2(2:end);
Theta1_grad = Theta1_grad + delta2*X(i,:); %40 confidence
Theta2_grad = Theta2_grad + delta3*a2'; %40% confidence

end



J=J/m;

Q=0;
R=0;

for i = 1:size(Theta1,1),

for j = 2:size(Theta1,2),

Q = Q + Theta1(i,j)^2;

end


end

for i = 1:size(Theta2,1),

for j = 2:size(Theta2,2),

R = R + Theta2(i,j)^2;

end


end

J=J+ (lambda/(2*m))*(Q+R);

%backpropagation remains

Theta1_grad = (1/m)*Theta1_grad; %+ lambda*Theta1;
%Theta1_grad(:,1) = (1/m)*Theta1_grad(:,1);

Theta2_grad = (1/m)*Theta2_grad; %+ lambda*Theta2;
%Theta2_grad(:,1) = (1/m)*Theta2_grad(:,1);


%Implement backpropagation

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

Theta1_grad = Theta1_grad + (lambda/m)*Theta1;
Theta1_grad(:,1) = Theta1_grad(:,1) - (lambda/m)*Theta1(:,1);

Theta2_grad = Theta2_grad + (lambda/m)*Theta2;
Theta2_grad(:,1) = Theta2_grad(:,1) - (lambda/m)*Theta2(:,1);

grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
