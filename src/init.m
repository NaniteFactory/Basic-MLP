% �ۼ�: 2012154021 ������ 2017�� 5�� 29�� ������

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
% �н� �� �׽�Ʈ�� ���� ����. patterns. pats
disp('loading...')
imgsTrain = loadMNISTImages('train-images-idx3-ubyte');
labelsTrain = loadMNISTLabels('train-labels-idx1-ubyte');
imgsTest = loadMNISTImages('t10k-images-idx3-ubyte');
labelsTest = loadMNISTLabels('t10k-labels-idx1-ubyte');

disp('init...')

% Show the first 10 images
%display_network(imgsTrain(:,1:10)); 
%disp(labelsTrain(1:10));

% �н� ���� �� ����
g_nPatsTrain = size(imgsTrain, 2); % 2�� ���� ���� ���. �� Ʈ���̴� ���� �̹����� ��.
g_nPatsTest = size(imgsTest, 2); % ��. �׽�Ʈ ���� �̹����� ��.
g_nInputs = size(imgsTrain, 1); % 1�� ���� ���� ���. �� ���� �̹����� �Է� �ػ�.
g_nHidden = 25; %floor(g_nInputs / 2); % ������ ����� ��.
g_nOutputs = 10; % ��� ����� ��.

% 6�� ���� �ʹ� �����ϱ� �� Ƚ��(����ũ)�� 20�� ���� �н��Ѵ�.
g_nPatsTrainPerEpoch = g_nPatsTrain / 3000;
g_nPatsTestPerEpoch = g_nPatsTest / 500;

% ���̾ ����ġ�� �����Ͽ�, ���� �ʱ� ����ġ�� �����ϰ� ����
g_hiddenWeights = 0.5 * (rand(g_nHidden, g_nInputs+1) - ones(g_nHidden, g_nInputs+1) * .5);
g_outputWeights = 0.5 * (rand(g_nOutputs, g_nHidden+1) - ones(g_nOutputs, g_nHidden+1) * .5);

g_inputTrain = imgsTrain;
g_inputTest = imgsTest;

% ��ǥ ���� ����(0: 1 0 0 0 , 1: 0 1 0 0, 2: 0 0 1 0, 3: 0 0 0 1, ...)
g_targetTrain = initTarget(g_nPatsTrain, g_nOutputs, labelsTrain);
g_targetTest = initTarget(g_nPatsTest, g_nOutputs, labelsTest);

g_eta = 0.1; % �н��� 

g_nEpochs = 100; % �н� epoch ��. run�� ȣ���ϴ� ����ũ�� ��. % const.
g_nEpochsSum = 0; % ���ݱ����� �н� epoch ��. run�� ȣ��� ������ g_nEpochs�� ���Ͽ� �����Ѵ�. 

% �н� Ŀ�긦 �÷�
g_nEpochsMinInGraph = 200; % ���� �÷� �� ����ũ ���� �ּ�. �׷��� �׸� �� ��ǥ�� ũ��.
g_errorsTrainInGraph = zeros(1,g_nEpochsMinInGraph); % �׷������� �н� ������ ����. ����ũ�� ����.
g_errorsTestInGraph = zeros(1,g_nEpochsMinInGraph); % �׷������� �׽�Ʈ ������ ����. ����ũ�� ����.
g_nEpochsInGraph = [1:g_nEpochsMinInGraph]; % �׷������� ����ũ.
