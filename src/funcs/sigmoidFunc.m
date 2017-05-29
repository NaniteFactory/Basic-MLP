function [output] = sigmoid(totalInput)
  output = 1.0 ./ (ones(size(totalInput)) + exp(-1.0 * totalInput));