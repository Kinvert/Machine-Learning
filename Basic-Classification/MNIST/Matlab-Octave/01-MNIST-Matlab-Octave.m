clc; clear all; close all;

function y = sigmoid(x)
  % Forward Pass
  y = 1 ./ (1 + exp(-x));
end

function p = softmax(z)
  % Forward Pass
  p = exp(z) ./ sum(exp(z));
end

d = load("../../../zData/MNIST/mnist.mat");

i = reshape(d.trainX(1,:), 28, 28)';
image(i);

% Normalize
d.trainX = double(d.trainX) / 255.0;
d.testX = double(d.testX) / 255.0;

% One Hot Encoding
%trainY = zeros(size(d.trainY, 1), 10);
%for i = 1:60000
%  thisIdx = d.trainY(i);
%  trainY(i, thisIdx+1) = 1;
%end
%testY = zeros(size(d.testY, 1), 10);
%for i = 1:10000
%  thisIdx = d.testY(i);
%  testY(i, thisIdx+1) = 1;
%end
%save("trainY.mat", "trainY");
%save("testY.mat", "testY");
trainY = load("../../../zData/MNIST/trainY.mat").trainY;
testY = load("../../../zData/MNIST/testY.mat").testY;

numHidden = 64;
numEpochs = 2;
batchSize = 64;
lr = 0.001;

% Initialize the weights and biases
W1 = randn(numHidden, 784) - 0.5;
b1 = randn(numHidden, 1) - 0.5;
W2 = randn(10, numHidden) - 0.5;
b2 = randn(10, 1) - 0.5;

W1 = W1 .* sqrt(1 / (784 + numHidden));
W2 = W2 .* sqrt(1 / (10 + numHidden));

W1 = double(W1);
b1 = double(b1);
W2 = double(W2);
b2 = double(b2);

for i = 1:numEpochs
  for j = 1:batchSize:size(d.trainX, 1)
    % Forward
    a1 = d.trainX(min(j:j+batchSize-1, 59999) + 1, :)';
    z2 = double(W1 * a1 + b1);
    a2 = double(sigmoid(z2));
    z3 = double(W2 * a2 + b2);
    a3 = softmax(z3);

    % Backprop
    %oldLabels = d.trainY(min(j:j+batchSize-1, 59999) + 1);
    oneHotBatch = trainY(min(j:j+batchSize-1, 59999) + 1, :)';
    delta3 = double(a3 - oneHotBatch);
    delta2 = double(a2 .* (1 - a2) .* (W2' * delta3));
    W2 = W2 - lr * delta3 * a2';
    b2 = b2 - lr * delta3;
    W1 = W1 - lr * delta2 * a1';
    b1 = b1 - lr * delta2;
  end
  fprintf("epoch %d of %d\n", i, numEpochs);
end

% Use the trained network to make predictions on the test set
numCorrect = 0
numWrong = 0
predictions = zeros(size(d.testX, 1), 1);
b1 = b1(:, 1);
b2 = b2(:, 1);
for i = 1:size(d.testX, 1)
  % Forward
  a1 = d.testX(i, :)';
  z2 = W1 * a1 + b1;
  a2 = sigmoid(z2);
  z3 = W2 * a2 + b2;
  a3 = sigmoid(z3);
  
  % argmax
  [~, index] = max(a3);
  
  % Matlab 1 based
  predictions(i) = index - 1;
  
  numCorrect = numCorrect + (predictions(i) == d.testY(i));
  numWrong = numWrong + (predictions(i) ~= d.testY(i));
  fprintf("  %d      %d\n", predictions(i), d.testY(i));
end

fprintf("Pred vs Real\n");
fprintf("%d Correct\n", numCorrect);
fprintf("%d Wrong\n", numWrong);
fprintf("%.4f Accuracy\n", numCorrect / 10000.0);