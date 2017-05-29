function [ target ] = initTarget( nPats, nOutputs, labels )
% 타겟 클래스 분류함. 행렬 반환.
target = zeros(nPats, nOutputs);
    for i = 1:nPats
        target(i, labels(i) + 1) = 1;
    end
target = target';
end

