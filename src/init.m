% 작성: 2012154021 문동선 2017년 5월 29일 월요일

% downloadMNISTDatasets
if exist('train-images-idx3-ubyte', 'file') && ...
    exist('train-labels-idx1-ubyte', 'file') && ...
    exist('t10k-images-idx3-ubyte', 'file') && ...
    exist('t10k-labels-idx1-ubyte', 'file')
    disp('file already exists.')
else
    disp('downloading...')
    websave('train-images-idx3-ubyte.gz', 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz')
    disp('downloading...')
    websave('train-labels-idx1-ubyte.gz', 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz')
    disp('downloading...')
    websave('t10k-images-idx3-ubyte.gz', 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz')
    disp('downloading...')
    websave('t10k-labels-idx1-ubyte.gz', 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz')
    disp('extracting...')
    gunzip('*.gz')
    disp('clearing...')
    delete('*.gz')
    disp('download complete.')
end

% loadMNIST
% 학습 및 테스트의 패턴 생성. patterns. pats
disp('loading...')
imgsTrain = loadMNISTImages('train-images-idx3-ubyte');
labelsTrain = loadMNISTLabels('train-labels-idx1-ubyte');
imgsTest = loadMNISTImages('t10k-images-idx3-ubyte');
labelsTest = loadMNISTLabels('t10k-labels-idx1-ubyte');

disp('init...')

% Show the first 10 images
%display_network(imgsTrain(:,1:10)); 
%disp(labelsTrain(1:10));

% 학습 패턴 수 정의
g_nPatsTrain = size(imgsTrain, 2); % 2는 열의 갯수 출력. 즉 트레이닝 패턴 이미지의 수.
g_nPatsTest = size(imgsTest, 2); % 상동. 테스트 패턴 이미지의 수.
g_nInputs = size(imgsTrain, 1); % 1은 행의 갯수 출력. 즉 패턴 이미지의 입력 해상도.
g_nHidden = 25; %floor(g_nInputs / 2); % 은닉층 노드의 수.
g_nOutputs = 10; % 출력 노드의 수.

% 6만 개는 너무 많으니까 한 횟수(이포크)에 20개 정도 학습한다.
g_nPatsTrainPerEpoch = g_nPatsTrain / 3000;
g_nPatsTestPerEpoch = g_nPatsTest / 500;

% 바이어스 가중치를 포함하여, 작은 초기 가중치를 랜덤하게 설정
g_hiddenWeights = 0.5 * (rand(g_nHidden, g_nInputs+1) - ones(g_nHidden, g_nInputs+1) * .5);
g_outputWeights = 0.5 * (rand(g_nOutputs, g_nHidden+1) - ones(g_nOutputs, g_nHidden+1) * .5);

g_inputTrain = imgsTrain;
g_inputTest = imgsTest;

% 목표 패턴 설정(0: 1 0 0 0 , 1: 0 1 0 0, 2: 0 0 1 0, 3: 0 0 0 1, ...)
g_targetTrain = initTarget(g_nPatsTrain, g_nOutputs, labelsTrain);
g_targetTest = initTarget(g_nPatsTest, g_nOutputs, labelsTest);

g_eta = 0.1; % 학습률 

g_nEpochs = 100; % 학습 epoch 수. run이 호출하는 이포크의 수. % const.
g_nEpochsSum = 0; % 지금까지의 학습 epoch 수. run이 호출될 때마다 g_nEpochs에 의하여 증가한다. 

% 학습 커브를 플롯
g_nEpochsMinInGraph = 200; % 에러 플롯 당 이포크 수의 최소. 그래프 그릴 때 좌표축 크기.
g_errorsTrainInGraph = zeros(1,g_nEpochsMinInGraph); % 그래프에서 학습 에러율 변수. 이포크당 에러.
g_errorsTestInGraph = zeros(1,g_nEpochsMinInGraph); % 그래프에서 테스트 에러율 변수. 이포크당 에러.
g_nEpochsInGraph = [1:g_nEpochsMinInGraph]; % 그래프에서 이포크.
