function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
theta1 = 0;
theta2 = 0;
for iter = 1:num_iters
        theta1 = theta(1) - alpha * (1/m) * sum(((X * theta) -y) .* X(:,1));
        theta2 = theta(2) - alpha * (1/m) * sum(((X * theta) -y) .* X(:,2));
	theta(1) = theta1;
	theta(2) = theta2;

	%wrong way (iterative):
	%for i=1:2,
	%    summ = 0;
	%    for k=1:m,
	%	h = theta(1)+(theta(2) * X(k,2));
	%	summ = summ + (h-y(k))*X(k,i);
	%    end;
    	%    theta(i) = theta(i) - alpha * (1/m)*summ; 
	%end;

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end
end
