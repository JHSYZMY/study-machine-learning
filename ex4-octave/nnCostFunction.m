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

X = [ones(m,1) X]; % X:5000x401 Theta_1:25x401 Theta_2:10x26

z2 = Theta1 * X'; %z2: 25x5000
a2 = sigmoid(z2); % hidden_layer¼¤»îÏî

tmp = size(a2,2);
a2 = [ones(1,tmp);a2]; % a2:26x5000

z3 = (Theta2 * a2)'; % z3: 5000x10
h = sigmoid(z3);

temp_y = zeros(m,num_labels); % 5000x10

for c = 1 : m,
    temp_y(c,y(c)) = 1;
end;

J = 1/m*sum(sum(-temp_y.*log(h)-(1-temp_y).*log(1-h)));
J = J + lambda/2/m*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));


delta3=h-temp_y;	%5000*10

delta2=(Theta2)'*(delta3'); % Theta2:10*26 delta: 5000*10 ->delta2:26*5000
delta2=delta2(2:end,:);	%25*5000
delta2=delta2.*sigmoidGradient(z2);	%25*5000.*25*5000

acc_grad1=zeros(size(Theta1));
acc_grad2=zeros(size(Theta2));

acc_grad1=(acc_grad1+delta2*(X));	%25*5000*5000*401=25*401
acc_grad2=(acc_grad2+(delta3')*(a2)');	%(5000*10)'*(26*5000)'=10*26
Theta1_grad=acc_grad1/m;
Theta2_grad=acc_grad2/m;


reg1=lambda/m.*Theta1(:,2:end);
reg2=lambda/m.*Theta2(:,2:end);
reg1=[zeros(size(Theta1,1),1),reg1];
reg2=[zeros(size(Theta2,1),1),reg2];
Theta1_grad=Theta1_grad+reg1;
Theta2_grad=Theta2_grad+reg2;












% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
