function [deltas] = hiddenDeltas(outputDeltas,hiddenOutputs,outputWeights)
  % deltas, outputs은 열벡터. 
  % outputWeights는 hidden->output 가중치 행렬 (각 행은 출력 유닛의 가중치 벡터)
  sigmoidDeriv = hiddenOutputs .* (ones(size(hiddenOutputs)) - hiddenOutputs);
  deltas = (outputWeights' * outputDeltas) .* sigmoidDeriv;