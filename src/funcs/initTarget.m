function [ target ] = initTarget( nPats, nOutputs, labels )
% Ÿ�� Ŭ���� �з���. ��� ��ȯ.
target = zeros(nPats, nOutputs);
    for i = 1:nPats
        target(i, labels(i) + 1) = 1;
    end
target = target';
end

