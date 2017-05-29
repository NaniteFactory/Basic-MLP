function [deltas] = outputDeltas(output,target)
  % deltas, output, target ¸ğµÎ º¤ÅÍ 
  sigmoidDeriv = output .* (ones(size(output)) - output);
  deltas = 2 * (target - output) .* sigmoidDeriv;