% 작성: 2012154021 문동선 2017년 5월 29일 월요일

% BP 학습 실행 스크립트.

fid = fopen('log.txt','a+');
fprintf(fid,'running...\n');

errorsLastEpochTrain = zeros(1, g_nEpochs); % 이번 스크립트 내에서 한 이포크 수행 직후 그 이포크의 에러율 구한 것. 벡터. 트레이닝.
errorsLastEpochTest = zeros(1, g_nEpochs); % 이번 스크립트 내에서 한 이포크 수행 직후 그 이포크의 에러율 구한 것. 벡터. 테스트.
startEpoch = g_nEpochsSum + 1; % g_nEpochsSum(전체이포크수) 기준으로 이번 스크립트에서 수행하는 첫 번째 횟수.

for epoch = 1:g_nEpochs, % 이번 스크립트에서 g_nEpoch가 100이라면 이포크 100번 만큼 학습을 수행할 것이다.
    sumSqrError = 0.0;
    sumSqrTestError = 0.0;
    outputWGrad = zeros(size(g_outputWeights));
    hiddenWGrad = zeros(size(g_hiddenWeights));
    
    startPatTrain = 1 + (g_nEpochsSum * g_nPatsTrainPerEpoch); % 이번 스크립트에서 몇 번째 트레이닝 패턴부터 학습을 수행하는지 그 트레이닝 패턴의 순번.
    startPatTrain = mod(startPatTrain, g_nPatsTrain); % 인덱스 변수에 대한 모듈로 연산은 상한최대치를 넘지 못하게 한다는 의미. 여기서 최대치는 g_nPatsTrain 다.
    endPatTrain = startPatTrain + g_nPatsTrainPerEpoch;
    if endPatTrain > g_nPatsTrain
        endPatTrain = g_nPatsTrain;
    end
    
    for pat = startPatTrain : endPatTrain, % i. pat. 각 i. 각 패턴에 대해서.
        % 전방향 패스(pass)
        inp = [g_inputTrain(:,pat)',[1]]';
        hiddenNetInputs = g_hiddenWeights * inp;
        hiddenStates = sigmoidFunc(hiddenNetInputs);
        hidStatesBias = [[hiddenStates]',[1]]';
        outputNetInputs = g_outputWeights * hidStatesBias;
        outputStates = sigmoidFunc(outputNetInputs);

        % 역방향 패스(pass)
        targetStates = g_targetTrain(:,pat);
        error = outputStates - targetStates;
        sumSqrError = sumSqrError + dot(error,error);
        outputDel = outputDeltas(outputStates, targetStates);
        outputWGrad = outputWGrad + outputDel * hidStatesBias';
        hiddenDel = hiddenDeltas(outputDel,hidStatesBias,g_outputWeights);
        hiddenWGrad = hiddenWGrad + hiddenDel(1:g_nHidden,:) * inp';
    end
    % epoch의 마지막에 가중치 갱신 % 학습 부분
    outputWChange = g_eta * outputWGrad;
    g_outputWeights = g_outputWeights + outputWChange;
    hiddenWChange = g_eta * hiddenWGrad;
    g_hiddenWeights = g_hiddenWeights + hiddenWChange;
  
    %% 이제 학습이 끝나고 테스트
    startPatTest = 1 + (g_nEpochsSum * g_nPatsTestPerEpoch); % 이번 스크립트에서 몇 번째 테스트 패턴부터 테스트 수행을 위해 네트워크에 인가되는지 그 테스트 패턴의 순번.
    startPatTest = mod(startPatTest, g_nPatsTest);
    endPatTest = startPatTest + g_nPatsTestPerEpoch;
    if endPatTest > g_nPatsTest
        endPatTest = g_nPatsTest;
    end
    
    % 테스트 패턴으로 네트워크의 성능 시험
    for pat = startPatTest : endPatTest % i. pat. 각 i. 각 패턴에 대해서.
        % 전향향 패스(pass)
        inp = [g_inputTest(:,pat)',[1]]';
        hiddenNetInputs = g_hiddenWeights * inp;
        hiddenStates = sigmoidFunc(hiddenNetInputs);
        hidStatesBias = [[hiddenStates]',[1]]';
        outputNetInputs = g_outputWeights * hidStatesBias;
        outputStates = sigmoidFunc(outputNetInputs);
        targetStates = g_targetTest(:,pat);
        error = outputStates - targetStates; % 에러 = (타겟 - 출력)
        sumSqrTestError = sumSqrTestError + dot(error,error); % (타겟 - 출력) 제곱 애들을 전부 더하고 갯수만큼 나눈 것이 MSE니까.
    end

    %% epoch의 마지막에 요약 통계 출력
    gradSize = norm([hiddenWGrad(:);outputWGrad(:)]);
    g_nEpochsSum = g_nEpochsSum + 1;
    MSE = sumSqrError/g_nPatsTrainPerEpoch;
    TestMSE = sumSqrTestError/g_nPatsTestPerEpoch;
    if g_nEpochsSum == 1
        startError = MSE; % 이 스크립트의 첫 번째 이포크의 MSE를 기억한다. 뒤에 그래프로 그릴 때 필요한 정보다.
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

% 학습 패턴 집합에 대한 학습 커브 플롯
g_errorsTrainInGraph(1,startEpoch:g_nEpochsSum) = errorsLastEpochTrain;
subplot(2,1,1), ...
  axis([1 max(g_nEpochsMinInGraph,g_nEpochsSum) 0 startError]),  hold on, ...
  plot(g_nEpochsInGraph(1,1:g_nEpochsSum),g_errorsTrainInGraph(1,1:g_nEpochsSum)),...
  title('Mean Squared Error on the Train Set'), ...
  xlabel('Learning Epoch'), ...
  ylabel('MSE');

% 테스트 패턴 집합에 대한 학습 커브 플롯
g_errorsTestInGraph(1,startEpoch:g_nEpochsSum) = errorsLastEpochTest;
subplot(2,1,2), ...
  axis([1 max(g_nEpochsMinInGraph,g_nEpochsSum) 0 startError]),  hold on, ...
  plot(g_nEpochsInGraph(1,1:g_nEpochsSum),g_errorsTestInGraph(1,1:g_nEpochsSum)), ...
  title('Mean Squared Error on the Test Set'), ...
  xlabel('Learning Epoch'), ...
  ylabel('MSE');

fprintf(fid,'finished.\n');
fclose(fid);