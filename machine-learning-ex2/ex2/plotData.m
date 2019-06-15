function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;
x1=X(:,1);
x2=X(:,2);

for i=1:length(y),
	if y(i,1)==1,
		plot(x1(i,1),x2(i,1),'k+','MarkerSize',7);
	else,
		plot(x1(i,1),x2(i,1),'ro','MarkerSize',7,'MarkerFaceColor','y');
	end
end

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%









% =========================================================================

hold off;
