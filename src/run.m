% �ۼ�: 2012154021 ������ 2017�� 5�� 29�� ������

% BP �н� ���� ��ũ��Ʈ.

fid = fopen('log.txt','a+');
fprintf(fid,'running...\n');

errorsLastEpochTrain = zeros(1, g_nEpochs); % �̹� ��ũ��Ʈ ������ �� ����ũ ���� ���� �� ����ũ�� ������ ���� ��. ����. Ʈ���̴�.
errorsLastEpochTest = zeros(1, g_nEpochs); % �̹� ��ũ��Ʈ ������ �� ����ũ ���� ���� �� ����ũ�� ������ ���� ��. ����. �׽�Ʈ.
startEpoch = g_nEpochsSum + 1; % g_nEpochsSum(��ü����ũ��) �������� �̹� ��ũ��Ʈ���� �����ϴ� ù ��° Ƚ��.

for epoch = 1:g_nEpochs, % �̹� ��ũ��Ʈ���� g_nEpoch�� 100�̶�� ����ũ 100�� ��ŭ �н��� ������ ���̴�.
    sumSqrError = 0.0;
    sumSqrTestError = 0.0;
    outputWGrad = zeros(size(g_outputWeights));
    hiddenWGrad = zeros(size(g_hiddenWeights));
    
    startPatTrain = 1 + (g_nEpochsSum * g_nPatsTrainPerEpoch); % �̹� ��ũ��Ʈ���� �� ��° Ʈ���̴� ���Ϻ��� �н��� �����ϴ��� �� Ʈ���̴� ������ ����.
    startPatTrain = mod(startPatTrain, g_nPatsTrain); % �ε��� ������ ���� ���� ������ �����ִ�ġ�� ���� ���ϰ� �Ѵٴ� �ǹ�. ���⼭ �ִ�ġ�� g_nPatsTrain ��.
    endPatTrain = startPatTrain + g_nPatsTrainPerEpoch;
    if endPatTrain > g_nPatsTrain
        endPatTrain = g_nPatsTrain;
    end
    
    for pat = startPatTrain : endPatTrain, % i. pat. �� i. �� ���Ͽ� ���ؼ�.
        % ������ �н�(pass)
        inp = [g_inputTrain(:,pat)',[1]]';
        hiddenNetInputs = g_hiddenWeights * inp;
        hiddenStates = sigmoidFunc(hiddenNetInputs);
        hidStatesBias = [[hiddenStates]',[1]]';
        outputNetInputs = g_outputWeights * hidStatesBias;
        outputStates = sigmoidFunc(outputNetInputs);

        % ������ �н�(pass)
        targetStates = g_targetTrain(:,pat);
        error = outputStates - targetStates;
        sumSqrError = sumSqrError + dot(error,error);
        outputDel = outputDeltas(outputStates, targetStates);
        outputWGrad = outputWGrad + outputDel * hidStatesBias';
        hiddenDel = hiddenDeltas(outputDel,hidStatesBias,g_outputWeights);
        hiddenWGrad = hiddenWGrad + hiddenDel(1:g_nHidden,:) * inp';
    end
    % epoch�� �������� ����ġ ���� % �н� �κ�
    outputWChange = g_eta * outputWGrad;
    g_outputWeights = g_outputWeights + outputWChange;
    hiddenWChange = g_eta * hiddenWGrad;
    g_hiddenWeights = g_hiddenWeights + hiddenWChange;
  
    %% ���� �н��� ������ �׽�Ʈ
    startPatTest = 1 + (g_nEpochsSum * g_nPatsTestPerEpoch); % �̹� ��ũ��Ʈ���� �� ��° �׽�Ʈ ���Ϻ��� �׽�Ʈ ������ ���� ��Ʈ��ũ�� �ΰ��Ǵ��� �� �׽�Ʈ ������ ����.
    startPatTest = mod(startPatTest, g_nPatsTest);
    endPatTest = startPatTest + g_nPatsTestPerEpoch;
    if endPatTest > g_nPatsTest
        endPatTest = g_nPatsTest;
    end
    
    % �׽�Ʈ �������� ��Ʈ��ũ�� ���� ����
    for pat = startPatTest : endPatTest % i. pat. �� i. �� ���Ͽ� ���ؼ�.
        % ������ �н�(pass)
        inp = [g_inputTest(:,pat)',[1]]';
        hiddenNetInputs = g_hiddenWeights * inp;
        hiddenStates = sigmoidFunc(hiddenNetInputs);
        hidStatesBias = [[hiddenStates]',[1]]';
        outputNetInputs = g_outputWeights * hidStatesBias;
        outputStates = sigmoidFunc(outputNetInputs);
        targetStates = g_targetTest(:,pat);
        error = outputStates - targetStates; % ���� = (Ÿ�� - ���)
        sumSqrTestError = sumSqrTestError + dot(error,error); % (Ÿ�� - ���) ���� �ֵ��� ���� ���ϰ� ������ŭ ���� ���� MSE�ϱ�.
    end

    %% epoch�� �������� ��� ��� ���
    gradSize = norm([hiddenWGrad(:);outputWGrad(:)]);
    g_nEpochsSum = g_nEpochsSum + 1;
    MSE = sumSqrError/g_nPatsTrainPerEpoch;
    TestMSE = sumSqrTestError/g_nPatsTestPerEpoch;
    if g_nEpochsSum == 1
        startError = MSE; % �� ��ũ��Ʈ�� ù ��° ����ũ�� MSE�� ����Ѵ�. �ڿ� �׷����� �׸� �� �ʿ��� ������.
    end
    errorsLastEpochTrain(1,epoch) = MSE;
    errorsLastEpochTest(1,epoch) = TestMSE;
    fprintf(fid,'%d  MSError=%f, MSTestError=%f, |G|=%f\n', ...
        g_nEpochsSum, MSE, TestMSE, gradSize);
end

clf;
if g_nEpochsSum > g_nEpochsMinInGraph
  g_nEpochsInGraph = [1:g_nEpochsSum];
end

% �н� ���� ���տ� ���� �н� Ŀ�� �÷�
g_errorsTrainInGraph(1,startEpoch:g_nEpochsSum) = errorsLastEpochTrain;
subplot(2,1,1), ...
  axis([1 max(g_nEpochsMinInGraph,g_nEpochsSum) 0 startError]),  hold on, ...
  plot(g_nEpochsInGraph(1,1:g_nEpochsSum),g_errorsTrainInGraph(1,1:g_nEpochsSum)),...
  title('Mean Squared Error on the Train Set'), ...
  xlabel('Learning Epoch'), ...
  ylabel('MSE');

% �׽�Ʈ ���� ���տ� ���� �н� Ŀ�� �÷�
g_errorsTestInGraph(1,startEpoch:g_nEpochsSum) = errorsLastEpochTest;
subplot(2,1,2), ...
  axis([1 max(g_nEpochsMinInGraph,g_nEpochsSum) 0 startError]),  hold on, ...
  plot(g_nEpochsInGraph(1,1:g_nEpochsSum),g_errorsTestInGraph(1,1:g_nEpochsSum)), ...
  title('Mean Squared Error on the Test Set'), ...
  xlabel('Learning Epoch'), ...
  ylabel('MSE');

fprintf(fid,'finished.\n');
fclose(fid);