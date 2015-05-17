function [data_train, data_test, ll_train,  Y_test_vector] ...
    = getDataSmall(N_data, dirName, fold)
% return ll_fun, prediction_fun, score_fun
 
useMex = 1;

[foo, name, foo] = fileparts(dirName);
switch (name)
    case 'test_task'
        nFeatures = 6438;
        nLabels = 3;
        rand('state',fold);
        r = randperm(N_data * 2);
        trainPoints = r(1:N_data);
        testPoints = r(N_data+1:end);
    case 'chunking'
        nFeatures = 29764;
        nLabels = 14;
        rand('state',fold);
        r = randperm(N_data * 2);
        trainPoints = r(1:N_data);
        testPoints = r(N_data+1:end);
    case 'segmentation'
        nFeatures = 1386;
        nLabels = 2;
        rand('state',fold);
        r = randperm(36);%N_data * 2);
        trainPoints = r(1:20);% r(1:N_data);
        testPoints = r(21:36);%r(N_data+1:end);
    case 'basenp'
        nFeatures = 6438;
        nLabels = 3;
        rand('state',fold);
        r = randperm(N_data * 2);
        trainPoints = r(1:N_data);
        testPoints = r(N_data+1:end);
    case 'japanesene'
        nFeatures = 102799;
        nLabels = 17;
        rand('state',fold);
        r = randperm(N_data * 2);
        trainPoints = r(1:N_data);
        testPoints = r(N_data+1:end);
    case  'protein'
        nFeatures=45;
        nLabels=8;
        trainPoints = 1:10;
        testPoints = 21:28;
end
data_train = loadData(dirName, nLabels, trainPoints, nFeatures, useMex);
data_test = loadData(dirName, nLabels, testPoints, nFeatures, useMex);
X_train = data_train.X;
X_test = data_test.X;
Xcont_train = data_train.Xcont;
Xcont_test = data_test.Xcont;

ll_train = @(f) ll_funStructured(f, data_train, useMex);
prediction_fun = @(fStar) predictiveMarginalsN(fStar, data_test);
Y_test_vector = cat(1, data_test.Y{:});
end


%% loadData
function [ dataset ] = loadData(dirName, nLabels, indexData, nFeatures, useMex)
% dataset is a structure with the following fields:
%   N : nb data points
%   nLabels: nb class labels in data
%   T(n) : length of each data point in nb nodes
%   TT : sum(T)
%   X : features of all data points, dim TT * length feature vector
%   dataEnds(n), dataStarts(n): end/ start index of data point n in X
%   Y : cell array of class labels, length N; each array dim T(n) * 1
%   edgeStruct : used by UGM
%   unary{n}(node, class) : index into f
%   binary(class, class) : index into f
%   max : length of f

dataset.N = size(indexData, 2);
dataset.X = sparse(0,0);
dataset.Xcont = sparse(0,0);
dataset.Y = cell(dataset.N ,1);

dataset.edgeStruct = cell(dataset.N ,1);
dataset.unary = cell(dataset.N ,1);

for n =1:dataset.N
    thisXfeatures = readSparse([ dirName filesep int2str(indexData(n)),'.x'],nFeatures);
    
    %%  EVB subsampling features
    ns = min(size(thisXfeatures, 1),2);
    thisXfeatures = thisXfeatures(1:ns,:);
    
    dataset.X = [dataset.X; thisXfeatures];
    
    thisXfeatures = load([ dirName filesep 'word2vec' filesep int2str(indexData(n)),'.x']);
    thisXfeatures(:,1)=[]; 
    
    %% EVB
    thisXfeatures = thisXfeatures(1:ns,:);

    
    dataset.Xcont = [dataset.Xcont; thisXfeatures];
    
    
    nNodes = size(thisXfeatures,1);
    dataset.T(n) = nNodes;
    dataset.Y{n} = load([dirName filesep int2str(indexData(n)) '.y']);
    dataset.Y{n} = dataset.Y{n} + 1; % labels start from 0 in files, we want to start from 1
    
    %% EVB 
    nLabels = 3;
    aaa = dataset.Y{n}; 
    aaa  = aaa(1:ns);
    aaa(aaa>3) = 3;
    dataset.Y{n} = aaa;
    
    dataset.edgeStruct{n}.useMex = useMex;
    dataset.edgeStruct{n}.edgeEnds = int32([1:nNodes-1 ; 2:nNodes])'; % array dim nEdges * 2, where nEdges=nNodes-1; row e, col 1 says at which node edge e starts; row e, col 2 says at which node it ends; in our case edges are ordered from one node to the next
    dataset.edgeStruct{n}.nStates = int32(repmat(nLabels, 1, nNodes)); %array dim 1*nNodes, contains nLabels everywhere (UGM supports variable number of allowed states per node)
    
    dataset.unary{n} = zeros(dataset.T(n), nLabels);
end

dataset.nLabels = nLabels;
dataset.TT = sum(dataset.T);
dataset.dataEnds = cumsum(dataset.T);
dataset.dataStarts = [1, dataset.dataEnds(1:end-1)+1];
dataset.max = 0;

% .unary has the most basic (assumes no ties) shape compatible with
% the linear CRF: f~(n, t, y)
for yt = 1:nLabels
    for n=1:dataset.N
        for t=1:dataset.T(n);
            newFIdx = dataset.max + 1;
            dataset.max = newFIdx;
            dataset.unary{n}(t, yt) = newFIdx;
        end
    end
end
[dataset.binary, dataset.max] = indexRange(dataset.max, [nLabels, nLabels]);

end

%% readSparse
function spdata = readSparse(filename,nFeatures)
data = load(filename);
spdata = spconvert(data);

if size(spdata,2) ~= nFeatures
    spdata(end,nFeatures) = 0;
end
end

function [idxBlock, endIdx] = indexRange(maxIdx, dims)
startIdx = maxIdx + 1;
endIdx = maxIdx + prod(dims);
idxBlock = reshape(startIdx:endIdx, dims);
end



